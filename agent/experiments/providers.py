from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol, Sequence

from .manifest import (
    ManifestError,
    load_dataset_manifest,
    select_case,
    verify_case_data_files,
    verify_case_metadata,
)
from .models import ExperimentSpec, sha256_path


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderUnavailableError(ProviderError):
    """Raised when a requested provider cannot be used."""


class ProviderExecutionError(ProviderError):
    """Raised when a provider does not produce a valid real candidate."""

    def __init__(
        self,
        message: str,
        *,
        artifacts: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.artifacts = dict(artifacts or {})


@dataclass(frozen=True)
class CandidateResult:
    """One provider-produced render and its measured outputs."""

    metrics: Dict[str, float]
    render_count: int
    artifacts: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    test_only: bool = False

    def __post_init__(self) -> None:
        if self.render_count < 1:
            raise ProviderExecutionError("A candidate must report at least one render")
        if not self.metrics:
            raise ProviderExecutionError("A candidate must report measured metrics")
        for name, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProviderExecutionError(f"Metric {name!r} is not numeric")
            if not math.isfinite(float(value)):
                raise ProviderExecutionError(f"Metric {name!r} is not finite")
        if not self.artifacts:
            raise ProviderExecutionError(
                "A candidate must point to at least one real artifact"
            )
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ProviderExecutionError(
                f"Candidate metadata is not JSON serializable: {exc}"
            ) from exc


@dataclass(frozen=True)
class ProviderBatch:
    """One provider call may return a complete iterative trajectory."""

    candidates: Sequence[CandidateResult]
    stop: bool = False
    test_only: bool = False

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ProviderExecutionError("Provider returned no candidates")


@dataclass(frozen=True)
class GenerationRequest:
    spec: ExperimentSpec
    dataset_manifest_path: Path
    output_dir: Path
    call_index: int
    remaining_renders: Optional[int]
    remaining_seconds: Optional[float]
    deadline_monotonic: Optional[float]
    history: Sequence[Dict[str, Any]]
    previous_candidate: Optional[Dict[str, Any]]


class ExperimentProvider(Protocol):
    name: str
    test_only: bool

    def check_available(self) -> None:
        ...

    def generate(
        self,
        request: GenerationRequest,
    ) -> CandidateResult | ProviderBatch:
        ...


def load_provider(
    import_path: str,
    options: Mapping[str, Any],
) -> ExperimentProvider:
    if not import_path:
        raise ProviderUnavailableError(
            "No provider configured; refusing to fabricate a successful run"
        )
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ProviderUnavailableError(
            "Provider must use the import form 'module:ClassOrFactory'"
        )
    try:
        module = importlib.import_module(module_name)
        target = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise ProviderUnavailableError(
            f"Cannot import provider {import_path}: {exc}"
        ) from exc

    try:
        if inspect.isclass(target) or callable(target):
            provider = target(**dict(options))
        else:
            provider = target
    except Exception as exc:
        raise ProviderUnavailableError(
            f"Cannot initialize provider {import_path}: {exc}"
        ) from exc

    if not callable(getattr(provider, "check_available", None)):
        raise ProviderUnavailableError(
            f"Provider {import_path} has no check_available()"
        )
    if not callable(getattr(provider, "generate", None)):
        raise ProviderUnavailableError(f"Provider {import_path} has no generate()")
    if not isinstance(getattr(provider, "name", None), str):
        raise ProviderUnavailableError(f"Provider {import_path} has no string name")
    if not isinstance(getattr(provider, "test_only", None), bool):
        raise ProviderUnavailableError(
            f"Provider {import_path} must declare test_only"
        )
    return provider


@contextmanager
def _temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _render_timeout_seconds(request: GenerationRequest) -> int:
    value = request.spec.method_config.get("render_timeout_seconds", 30)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProviderExecutionError(
            "method_config.render_timeout_seconds must be a positive integer"
        )
    return value


class SingleChainProvider:
    """Adapter for iterative trajectories and independent model-spec samples."""

    name = "phero_viz_single_chain"
    test_only = False

    def __init__(
        self,
        api_key_envs: Optional[Sequence[str]] = None,
        wall_clock_rounds: Optional[int] = None,
    ) -> None:
        self.api_key_envs = tuple(
            api_key_envs
            or (
                "MODEL_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "LLM_API_KEY",
                "OPENAI_API_KEY",
            )
        )
        self.wall_clock_rounds = wall_clock_rounds

    def check_available(self) -> None:
        try:
            from app.services import single_chain_runner
            from app.services.model_client import ModelConfig, ModelClientError
        except ImportError as exc:
            raise ProviderUnavailableError(
                "SingleChainProvider must run with agent/ on PYTHONPATH"
            ) from exc
        single_chain_runner._load_env_file()
        if not any(os.getenv(name) for name in self.api_key_envs):
            joined = ", ".join(self.api_key_envs)
            raise ProviderUnavailableError(
                f"No model credentials found in {joined}; run recorded as failed"
            )
        try:
            ModelConfig.from_env()
        except ModelClientError as exc:
            raise ProviderUnavailableError(str(exc)) from exc

    def generate(
        self,
        request: GenerationRequest,
    ) -> ProviderBatch:
        if (
            request.spec.schedule == "iterative"
            and request.previous_candidate is not None
        ):
            raise ProviderExecutionError(
                "SingleChainProvider cannot resume an iterative trajectory across calls"
            )

        if request.spec.schedule == "best_of_n":
            rounds = 1
        elif request.remaining_renders is not None:
            rounds = request.remaining_renders
        else:
            configured = request.spec.method_config.get(
                "rounds",
                self.wall_clock_rounds,
            )
            if isinstance(configured, bool) or not isinstance(configured, int):
                raise ProviderExecutionError(
                    "Wall-clock single-chain runs require integer method_config.rounds"
                )
            rounds = configured
        if rounds < 1:
            raise ProviderExecutionError("Single-chain rounds must be positive")

        manifest_path = request.dataset_manifest_path
        try:
            manifest_cases = load_dataset_manifest(
                manifest_path,
                dataset_mode=request.spec.dataset_mode,
            )
            selected_case = select_case(
                manifest_cases,
                request.spec.case_id,
            )
            verify_case_metadata(
                selected_case,
                panel_count=request.spec.panel_count,
                split=request.spec.split,
            )
        except ManifestError as exc:
            raise ProviderExecutionError(
                f"Cannot select case {request.spec.case_id!r}: {exc}"
            ) from exc
        if selected_case.panel_count is not None and selected_case.panel_count > 1:
            raise ProviderExecutionError(
                f"case_id {selected_case.case_id!r} has panel_count="
                f"{selected_case.panel_count}; SingleChainProvider cannot run "
                "multi-panel cases, use a multi-panel provider"
            )
        try:
            verify_case_data_files(
                selected_case,
                manifest_path=manifest_path,
            )
        except ManifestError as exc:
            raise ProviderExecutionError(
                f"Cannot verify case data files: {exc}"
            ) from exc
        case = selected_case.payload
        evaluation_expectation = case.get("evaluation_expectation")
        if not isinstance(evaluation_expectation, Mapping):
            raise ProviderExecutionError(
                "SingleChainProvider requires evaluation_expectation; "
                "judge-only fidelity is not accepted for production runs"
            )

        data_path_value = case.get("data_path")
        user_goal = case.get("user_goal")
        chart_family = case.get("chart_family")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (data_path_value, user_goal, chart_family)
        ):
            raise ProviderExecutionError(
                "Manifest case requires data_path, user_goal, and chart_family"
            )
        data_path = Path(str(data_path_value)).expanduser()
        if not data_path.is_absolute():
            data_path = Path(request.spec.dataset_manifest_path).parent / data_path
        data_path = data_path.resolve()
        if not data_path.is_file():
            raise ProviderExecutionError(f"Case data file does not exist: {data_path}")

        sheet = case.get("sheet")
        intent = case.get("intent")
        if intent is not None and not isinstance(intent, dict):
            raise ProviderExecutionError("Manifest case intent must be an object")

        from app.services import single_chain_runner

        core_runs_root = request.output_dir / "core_runs"
        core_runs_root.mkdir(parents=True, exist_ok=True)
        discovered_run_dir: Optional[Path] = None

        def capture_progress(event: str, payload: Dict[str, Any]) -> None:
            nonlocal discovered_run_dir
            if event == "run_directory_ready" and payload.get("path"):
                discovered_run_dir = Path(str(payload["path"])).resolve()

        old_runs_dir = single_chain_runner.RUNS_DIR
        old_client = single_chain_runner._LLM_CLIENT
        old_model_client = single_chain_runner._MODEL_CLIENT
        random.seed(request.spec.seed)
        try:
            import numpy as np

            np.random.seed(request.spec.seed)
        except ImportError:
            pass

        environment = {
            "LLM_MODEL": request.spec.backbone,
            "FORCE_ALL_ROUNDS": "1",
        }
        render_timeout_seconds = _render_timeout_seconds(request)
        if request.remaining_seconds is not None:
            environment["LLM_TIMEOUT"] = str(max(request.remaining_seconds, 1.0))

        try:
            single_chain_runner.RUNS_DIR = core_runs_root
            single_chain_runner._LLM_CLIENT = None
            single_chain_runner._MODEL_CLIENT = None
            with _temporary_environment(environment):
                single_chain_runner.run_chain(
                    str(data_path),
                    str(user_goal),
                    str(chart_family),
                    rounds=rounds,
                    sheet=sheet,
                    intent=dict(intent or {}),
                    progress_callback=capture_progress,
                    initial_generation=str(
                        request.spec.method_config.get(
                            "initial_generation",
                            "model_spec",
                        )
                    ),
                    seed=request.spec.seed + request.call_index - 1,
                    temperature=request.spec.method_config.get("temperature"),
                    memory_mode=str(
                        request.spec.method_config.get(
                            "memory_mode",
                            "none"
                            if request.spec.schedule == "best_of_n"
                            else "full",
                        )
                    ),
                    evaluation_expectation=case.get(
                        "evaluation_expectation"
                    ),
                    metric_config=request.spec.metric_config.get("evaluator"),
                    render_timeout_seconds=render_timeout_seconds,
                )
        except Exception as exc:
            partial = (
                discovered_run_dir
                if discovered_run_dir is not None
                and discovered_run_dir.is_dir()
                else core_runs_root
            )
            raise ProviderExecutionError(
                f"Single-chain execution failed: {exc}",
                artifacts={"partial_core_run": str(partial)},
            ) from exc
        finally:
            single_chain_runner.RUNS_DIR = old_runs_dir
            single_chain_runner._LLM_CLIENT = old_client
            single_chain_runner._MODEL_CLIENT = old_model_client

        if discovered_run_dir is None or not discovered_run_dir.is_dir():
            raise ProviderExecutionError(
                "Core runner did not report a persistent run directory"
            )
        try:
            discovered_run_dir.relative_to(request.output_dir.resolve())
        except ValueError as exc:
            raise ProviderExecutionError(
                "Core runner wrote artifacts outside the allocated provider directory"
            ) from exc

        iteration_paths = sorted(discovered_run_dir.glob("iteration_*.json"))
        if not iteration_paths:
            raise ProviderExecutionError("Core runner produced no iteration records")

        candidates: list[CandidateResult] = []
        for iteration_path in iteration_paths:
            try:
                iteration = json.loads(iteration_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProviderExecutionError(
                    f"Cannot read core iteration {iteration_path}: {exc}"
                ) from exc
            if not isinstance(iteration, dict):
                raise ProviderExecutionError(
                    f"Core iteration is not an object: {iteration_path}"
                )
            stages = iteration.get("stages") or {}
            if isinstance(stages, Mapping):
                for stage_name, stage in stages.items():
                    if not isinstance(stage, Mapping):
                        continue
                    response = stage.get("response")
                    notes = str(stage.get("notes") or "")
                    if (
                        isinstance(response, Mapping)
                        and response.get("error")
                    ) or "llm_error:" in notes:
                        raise ProviderExecutionError(
                            f"Model provider failed in core stage {stage_name}; "
                            "refusing fallback output"
                        )

            raw_metrics = iteration.get("scores")
            if not isinstance(raw_metrics, Mapping):
                raise ProviderExecutionError(
                    f"Core iteration has no measured scores: {iteration_path}"
                )
            if not iteration.get("png_path"):
                raise ProviderExecutionError(
                    f"Core iteration produced no render: {iteration_path}"
                )
            programmatic = iteration.get("programmatic_evaluation")
            if not isinstance(programmatic, Mapping):
                raise ProviderExecutionError(
                    f"Core iteration has no programmatic evaluation: "
                    f"{iteration_path}"
                )
            metrics = {
                str(name): float(value)
                for name, value in raw_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            artifacts = {"iteration": str(iteration_path)}
            round_number = iteration.get("round")
            if isinstance(round_number, int):
                for label, pattern in (
                    ("render", f"figure_round_{round_number}.png"),
                    ("code", f"code_round_{round_number}.py"),
                    ("slots", f"slots_round_{round_number}.json"),
                    (
                        "programmatic_evaluation",
                        f"programmatic_evaluation_round_{round_number}.json",
                    ),
                    ("memory_snapshot", "memory_snapshot.json"),
                    ("memory_trace", "memory_trace.json"),
                    ("memory_compatibility", "pheromones.json"),
                ):
                    artifact = discovered_run_dir / pattern
                    if artifact.is_file():
                        artifacts[label] = str(artifact)
            candidates.append(
                CandidateResult(
                    metrics=metrics,
                    render_count=1,
                    artifacts=artifacts,
                    metadata={
                        "core_round": round_number,
                        "core_run_dir": str(discovered_run_dir),
                        "render_timeout_seconds": render_timeout_seconds,
                        "model_calls": {
                            str(stage_name): dict(
                                stage.get("model_metadata") or {}
                            )
                            for stage_name, stage in stages.items()
                            if isinstance(stage, Mapping)
                            and stage.get("model_metadata")
                        },
                    },
                )
            )
        return ProviderBatch(
            candidates=candidates,
            stop=request.spec.schedule == "iterative",
        )


def _contained_candidate_artifacts(
    raw_artifacts: Any,
    *,
    output_dir: Path,
    required_labels: set[str],
) -> Dict[str, str]:
    if not isinstance(raw_artifacts, Mapping):
        raise ProviderExecutionError(
            "Multi-panel checkpoint has no artifact mapping"
        )
    artifacts: Dict[str, str] = {}
    for label, path in raw_artifacts.items():
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(path, str)
            or not path
        ):
            raise ProviderExecutionError(
                "Multi-panel checkpoint artifact labels and paths must be "
                "non-empty strings"
            )
        artifacts[label] = path
    missing = sorted(required_labels - set(artifacts))
    if missing:
        raise ProviderExecutionError(
            "Multi-panel checkpoint is missing artifacts: "
            + ", ".join(missing)
        )

    root = output_dir.resolve()
    for label, raw_path in artifacts.items():
        path = Path(raw_path)
        if path.is_symlink():
            raise ProviderExecutionError(
                f"Multi-panel checkpoint artifact is a symlink: {label}"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ProviderExecutionError(
                f"Multi-panel checkpoint artifact is missing: {label}"
            ) from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ProviderExecutionError(
                "Multi-panel checkpoint artifact escaped its allocated "
                f"provider directory: {label}={resolved}"
            ) from exc
        artifacts[label] = str(resolved)
    return artifacts


def _reject_failed_multi_panel_iterations(
    artifacts: Mapping[str, str],
) -> None:
    iteration_paths = {
        label: path
        for label, path in artifacts.items()
        if label.startswith("panel.") and label.endswith(".iteration")
    }
    for label, raw_path in sorted(iteration_paths.items()):
        try:
            iteration = json.loads(
                Path(raw_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderExecutionError(
                f"Cannot read {label}: {exc}"
            ) from exc
        if not isinstance(iteration, Mapping):
            raise ProviderExecutionError(f"{label} is not a JSON object")
        stages = iteration.get("stages") or {}
        if not isinstance(stages, Mapping):
            continue
        for stage_name, stage in stages.items():
            if not isinstance(stage, Mapping):
                continue
            response = stage.get("response")
            notes = str(stage.get("notes") or "")
            if (
                isinstance(response, Mapping) and response.get("error")
            ) or "llm_error:" in notes:
                raise ProviderExecutionError(
                    f"Model provider failed in {label} stage {stage_name}; "
                    "refusing fallback output"
                )


def _multi_panel_checkpoint_candidate(
    checkpoint: Mapping[str, Any],
    *,
    expected_round: int,
    total_rounds: int,
    panel_ids: Sequence[str],
    memory_mode: str,
    seed: int,
    render_timeout_seconds: int,
    output_dir: Path,
) -> CandidateResult:
    if checkpoint.get("global_round") != expected_round:
        raise ProviderExecutionError(
            "Multi-panel checkpoints are not a contiguous global-round "
            f"trajectory: expected {expected_round}, got "
            f"{checkpoint.get('global_round')!r}"
        )
    panel_count = len(panel_ids)
    if checkpoint.get("render_count") != panel_count:
        raise ProviderExecutionError(
            f"Global round {expected_round} must report exactly "
            f"{panel_count} panel renders"
        )
    if checkpoint.get("cumulative_render_count") != panel_count * expected_round:
        raise ProviderExecutionError(
            f"Global round {expected_round} has inconsistent cumulative "
            "render accounting"
        )

    panels = checkpoint.get("panels")
    if not isinstance(panels, Mapping) or set(panels) != set(panel_ids):
        raise ProviderExecutionError(
            f"Global round {expected_round} does not contain every panel"
        )
    programmatic = checkpoint.get("programmatic_evaluation")
    if not isinstance(programmatic, Mapping):
        raise ProviderExecutionError(
            f"Global round {expected_round} has no programmatic evaluation"
        )
    panel_fidelity = programmatic.get("panel_fidelity")
    if (
        not isinstance(panel_fidelity, Mapping)
        or set(panel_fidelity) != set(panel_ids)
    ):
        raise ProviderExecutionError(
            f"Global round {expected_round} has incomplete panel fidelity"
        )
    numerator = sum(
        int(item.get("numerator", 0))
        for item in panel_fidelity.values()
        if isinstance(item, Mapping)
    )
    denominator = sum(
        int(item.get("denominator", 0))
        for item in panel_fidelity.values()
        if isinstance(item, Mapping)
    )
    if denominator <= 0:
        raise ProviderExecutionError(
            f"Global round {expected_round} has no applicable fidelity checks"
        )
    metrics: Dict[str, float] = {
        "data_fidelity": numerator / denominator,
        "execution_success": 1.0,
    }
    cohesion_ratio = (programmatic.get("cohesion") or {}).get("ratio")
    if (
        isinstance(cohesion_ratio, (int, float))
        and not isinstance(cohesion_ratio, bool)
    ):
        metrics["series_cohesion"] = float(cohesion_ratio)
    visual_scores: list[float] = []
    for panel in panels.values():
        if not isinstance(panel, Mapping):
            continue
        scores = panel.get("scores")
        score = (
            scores.get("visual_form")
            if isinstance(scores, Mapping)
            else None
        )
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            visual_scores.append(float(score))
    if visual_scores:
        metrics["visual_form"] = sum(visual_scores) / len(visual_scores)

    required_artifacts = {
        "output",
        "render",
        "result",
        "programmatic_evaluation",
        "memory_snapshot",
        "memory_trace",
        "memory_compatibility",
        "schedule_trace",
    }
    for panel_id in panel_ids:
        required_artifacts.update(
            {
                f"panel.{panel_id}.iteration",
                f"panel.{panel_id}.render",
                f"panel.{panel_id}.programmatic_evaluation",
            }
        )
    if memory_mode == "untyped":
        required_artifacts.add("untyped_memory")
    artifacts = _contained_candidate_artifacts(
        checkpoint.get("artifacts"),
        output_dir=output_dir,
        required_labels=required_artifacts,
    )
    _reject_failed_multi_panel_iterations(artifacts)

    served_models = sorted(
        {
            str(metadata.get("model"))
            for panel in panels.values()
            if isinstance(panel, Mapping)
            for metadata in (panel.get("model_calls") or {}).values()
            if isinstance(metadata, Mapping) and metadata.get("model")
        }
    )
    judge_models = sorted(
        {
            str(metadata.get("model"))
            for panel in panels.values()
            if isinstance(panel, Mapping)
            for metadata in [panel.get("judge_model_metadata") or {}]
            if isinstance(metadata, Mapping) and metadata.get("model")
        }
    )
    return CandidateResult(
        metrics=metrics,
        render_count=panel_count,
        artifacts=artifacts,
        metadata={
            "global_round": expected_round,
            "panel_count": panel_count,
            "rounds": total_rounds,
            "memory_mode": memory_mode,
            "seed": seed,
            "render_timeout_seconds": render_timeout_seconds,
            "cumulative_render_count": panel_count * expected_round,
            "served_models": served_models,
            "judge_models": judge_models,
        },
    )


class MultiPanelProvider:
    """Archive each complete global round as one comparable candidate."""

    name = "phero_viz_multi_panel"
    test_only = False

    def __init__(self, wall_clock_rounds: Optional[int] = None) -> None:
        self.wall_clock_rounds = wall_clock_rounds

    def check_available(self) -> None:
        SingleChainProvider().check_available()

    def generate(
        self,
        request: GenerationRequest,
    ) -> CandidateResult | ProviderBatch:
        from app.services.multi_panel_runner import run_multi_panel
        from app.services import single_chain_runner

        manifest_cases = load_dataset_manifest(
            request.dataset_manifest_path,
            dataset_mode=request.spec.dataset_mode,
        )
        selected_case = select_case(manifest_cases, request.spec.case_id)
        verify_case_metadata(
            selected_case,
            panel_count=request.spec.panel_count,
            split=request.spec.split,
        )
        try:
            verify_case_data_files(
                selected_case,
                manifest_path=request.dataset_manifest_path,
            )
        except ManifestError as exc:
            raise ProviderExecutionError(
                f"Cannot verify case data files: {exc}"
            ) from exc
        case = selected_case.payload
        raw_manifest = case.get("multi_panel_manifest")
        if case.get("eligible_for_experiment") is True and raw_manifest is not None:
            raise ProviderExecutionError(
                "Eligible multi-panel cases must use the verified inline panels"
            )
        manifest_base_dir = Path(request.spec.dataset_manifest_path).parent
        if raw_manifest is None:
            if not isinstance(case.get("panels"), list):
                raise ProviderExecutionError(
                    "MultiPanelProvider requires multi_panel_manifest or panels"
                )
            panel_manifest = dict(case)
        elif isinstance(raw_manifest, Mapping):
            panel_manifest = dict(raw_manifest)
        elif isinstance(raw_manifest, str) and raw_manifest.strip():
            path = Path(raw_manifest).expanduser()
            if not path.is_absolute():
                path = (
                    Path(request.spec.dataset_manifest_path).parent / path
                )
            resolved_manifest = path.resolve()
            try:
                panel_manifest = json.loads(
                    resolved_manifest.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ProviderExecutionError(
                    f"Cannot read multi-panel manifest: {exc}"
                ) from exc
            if not isinstance(panel_manifest, dict):
                raise ProviderExecutionError(
                    "Multi-panel manifest must contain an object"
                )
            manifest_base_dir = resolved_manifest.parent
        else:
            raise ProviderExecutionError(
                "multi_panel_manifest must be a path or object"
            )

        if "evaluation_expectation" not in panel_manifest:
            panel_manifest["evaluation_expectation"] = case.get(
                "evaluation_expectation"
            )
        if not isinstance(
            panel_manifest.get("evaluation_expectation"),
            Mapping,
        ):
            raise ProviderExecutionError(
                "MultiPanelProvider requires evaluation_expectation; "
                "judge-only fidelity is not accepted for production runs"
            )
        evaluator_config = request.spec.metric_config.get("evaluator")
        if evaluator_config is not None:
            panel_manifest["metric_config"] = evaluator_config

        panel_count = len(panel_manifest.get("panels") or [])
        if panel_count < 2:
            raise ProviderExecutionError(
                "MultiPanelProvider requires at least two panels"
            )
        if (
            request.spec.panel_count is not None
            and request.spec.panel_count != panel_count
        ):
            raise ProviderExecutionError(
                f"panel_count mismatch: spec={request.spec.panel_count}, "
                f"manifest={panel_count}"
            )
        if request.remaining_renders is not None and (
            request.remaining_renders < panel_count
            or int(request.spec.budget_value) % panel_count
        ):
            raise ProviderExecutionError(
                "Render budget must contain an integer number of complete "
                "multi-panel candidates"
            )

        if request.spec.schedule == "best_of_n":
            rounds = 1
            memory_mode = "none"
        elif request.remaining_renders is not None:
            if request.remaining_renders % panel_count:
                raise ProviderExecutionError(
                    "Render budget must be divisible by panel_count for "
                    "round-robin multi-panel runs"
                )
            rounds = request.remaining_renders // panel_count
            memory_mode = str(
                request.spec.method_config.get("memory_mode", "full")
            ).strip().lower()
        else:
            configured = request.spec.method_config.get(
                "rounds",
                self.wall_clock_rounds,
            )
            if isinstance(configured, bool) or not isinstance(configured, int):
                raise ProviderExecutionError(
                    "Wall-clock multi-panel runs require integer "
                    "method_config.rounds"
                )
            rounds = configured
            memory_mode = str(
                request.spec.method_config.get("memory_mode", "full")
            ).strip().lower()
        if rounds < 1:
            raise ProviderExecutionError("Multi-panel rounds must be positive")

        output_dir = request.output_dir / "multi_panel"
        render_timeout_seconds = _render_timeout_seconds(request)
        environment = {
            "LLM_MODEL": request.spec.backbone,
            "FORCE_ALL_ROUNDS": "1",
        }
        if request.remaining_seconds is not None:
            environment["LLM_TIMEOUT"] = str(
                max(request.remaining_seconds, 1.0)
            )
        old_model_client = single_chain_runner._MODEL_CLIENT
        old_compat_client = single_chain_runner._LLM_CLIENT
        try:
            single_chain_runner._MODEL_CLIENT = None
            single_chain_runner._LLM_CLIENT = None
            with _temporary_environment(environment):
                result = run_multi_panel(
                    panel_manifest,
                    output_dir=output_dir,
                    rounds=rounds,
                    initial_generation=str(
                        request.spec.method_config.get(
                            "initial_generation",
                            "model_spec",
                        )
                    ),
                    seed=request.spec.seed + request.call_index - 1,
                    temperature=request.spec.method_config.get("temperature"),
                    memory_mode=memory_mode,
                    render_timeout_seconds=render_timeout_seconds,
                    base_dir=manifest_base_dir,
                )
        except Exception as exc:
            artifacts = (
                {"partial_multi_panel_run": str(output_dir)}
                if output_dir.is_dir()
                else {}
            )
            raise ProviderExecutionError(
                f"Multi-panel execution failed: {exc}",
                artifacts=artifacts,
            ) from exc
        finally:
            single_chain_runner._MODEL_CLIENT = old_model_client
            single_chain_runner._LLM_CLIENT = old_compat_client
        if result.get("rounds") != rounds:
            raise ProviderExecutionError(
                "Multi-panel run reported an unexpected round count"
            )
        render_counts = result.get("render_counts")
        if (
            not isinstance(render_counts, Mapping)
            or set(render_counts) != {
                str(panel.get("id"))
                for panel in panel_manifest.get("panels") or []
                if isinstance(panel, Mapping)
            }
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != rounds
                for value in render_counts.values()
            )
        ):
            raise ProviderExecutionError(
                "Multi-panel run has inconsistent final render accounting"
            )
        if sum(int(value) for value in render_counts.values()) != (
            panel_count * rounds
        ):
            raise ProviderExecutionError(
                "Multi-panel run did not consume the exact panel-render budget"
            )
        checkpoints = result.get("checkpoints")
        if (
            not isinstance(checkpoints, list)
            or len(checkpoints) != rounds
            or not all(
                isinstance(checkpoint, Mapping)
                for checkpoint in checkpoints
            )
        ):
            raise ProviderExecutionError(
                "Multi-panel run did not produce one checkpoint per global "
                "round; refusing to substitute the final output"
            )
        panel_ids = [
            str(panel["id"])
            for panel in panel_manifest.get("panels") or []
            if isinstance(panel, Mapping)
        ]
        candidate_seed = request.spec.seed + request.call_index - 1
        candidates = tuple(
            _multi_panel_checkpoint_candidate(
                checkpoint,
                expected_round=round_number,
                total_rounds=rounds,
                panel_ids=panel_ids,
                memory_mode=memory_mode,
                seed=candidate_seed,
                render_timeout_seconds=render_timeout_seconds,
                output_dir=request.output_dir,
            )
            for round_number, checkpoint in enumerate(checkpoints, 1)
        )
        if sum(candidate.render_count for candidate in candidates) != (
            panel_count * rounds
        ):
            raise ProviderExecutionError(
                "Multi-panel checkpoint trajectory has inconsistent render "
                "accounting"
            )
        if (
            checkpoints[-1].get("programmatic_evaluation")
            != result.get("programmatic_evaluation")
        ):
            raise ProviderExecutionError(
                "Final multi-panel checkpoint does not match the final result"
            )
        final_render = Path(str(result.get("combined_figure_path") or ""))
        checkpoint_render = Path(
            str(
                (checkpoints[-1].get("artifacts") or {}).get("render")
                or ""
            )
        )
        if (
            not final_render.is_file()
            or not checkpoint_render.is_file()
            or sha256_path(final_render) != sha256_path(checkpoint_render)
        ):
            raise ProviderExecutionError(
                "Final multi-panel checkpoint render does not match the "
                "final result"
            )
        if request.spec.schedule == "best_of_n":
            if len(candidates) != 1:
                raise ProviderExecutionError(
                    "best_of_n must produce one complete multi-panel candidate "
                    "per provider call"
                )
            return candidates[0]
        return ProviderBatch(candidates=candidates, stop=True)


PHEROVIZ_PROVIDER_IMPORT_PATH = (
    "experiments.providers:UnifiedBenchmarkProvider"
)


class PheroVizProvider:
    """Route a mixed benchmark through the matching native provider.

    The canonical experiment-matrix import path is exposed by
    :class:`UnifiedBenchmarkProvider`.
    """

    name = "phero_viz"
    test_only = False

    def __init__(
        self,
        api_key_envs: Optional[Sequence[str]] = None,
        wall_clock_rounds: Optional[int] = None,
    ) -> None:
        self.single_provider = SingleChainProvider(
            api_key_envs=api_key_envs,
            wall_clock_rounds=wall_clock_rounds,
        )
        self.multi_provider = MultiPanelProvider(
            wall_clock_rounds=wall_clock_rounds,
        )

    def check_available(self) -> None:
        self.single_provider.check_available()

    def _route(self, request: GenerationRequest) -> tuple[int, ExperimentProvider]:
        try:
            cases = load_dataset_manifest(
                request.dataset_manifest_path,
                dataset_mode=request.spec.dataset_mode,
            )
            selected = select_case(cases, request.spec.case_id)
            verify_case_metadata(
                selected,
                panel_count=request.spec.panel_count,
                split=request.spec.split,
            )
            verify_case_data_files(
                selected,
                manifest_path=request.dataset_manifest_path,
            )
        except ManifestError as exc:
            raise ProviderExecutionError(
                f"Cannot route case {request.spec.case_id!r}: {exc}"
            ) from exc

        panel_count = selected.panel_count
        if (
            isinstance(panel_count, bool)
            or not isinstance(panel_count, int)
            or panel_count < 1
        ):
            raise ProviderExecutionError(
                f"Case {request.spec.case_id!r} must declare a positive "
                "panel_count for unified routing"
            )
        if panel_count == 1:
            return panel_count, self.single_provider
        return panel_count, self.multi_provider

    def _annotate_candidate(
        self,
        candidate: CandidateResult,
        *,
        panel_count: int,
        delegate: ExperimentProvider,
    ) -> CandidateResult:
        metadata = dict(candidate.metadata)
        if "phero_viz_provider" in metadata:
            raise ProviderExecutionError(
                "Delegate candidate uses reserved metadata key "
                "'phero_viz_provider'"
            )
        metadata["phero_viz_provider"] = {
            "router": self.name,
            "delegate": delegate.name,
            "panel_count": panel_count,
        }
        return CandidateResult(
            metrics=dict(candidate.metrics),
            render_count=candidate.render_count,
            artifacts=dict(candidate.artifacts),
            metadata=metadata,
            test_only=candidate.test_only,
        )

    def generate(
        self,
        request: GenerationRequest,
    ) -> CandidateResult | ProviderBatch:
        panel_count, delegate = self._route(request)
        result = delegate.generate(request)
        if isinstance(result, ProviderBatch):
            return ProviderBatch(
                candidates=tuple(
                    self._annotate_candidate(
                        candidate,
                        panel_count=panel_count,
                        delegate=delegate,
                    )
                    for candidate in result.candidates
                ),
                stop=result.stop,
                test_only=result.test_only,
            )
        return self._annotate_candidate(
            result,
            panel_count=panel_count,
            delegate=delegate,
        )


class UnifiedBenchmarkProvider(PheroVizProvider):
    """Canonical mixed-benchmark provider.

    Matrix import path:
    ``experiments.providers:UnifiedBenchmarkProvider``.
    """


UnifiedPheroVizProvider = PheroVizProvider
