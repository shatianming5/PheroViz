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
    verify_case_metadata,
)
from .models import ExperimentSpec


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


class SingleChainProvider:
    """Honest adapter for the existing single-chain iterative runner.

    The adapter intentionally rejects best-of-N: the current core runner's first
    round is deterministic default-slot generation, not an independent model
    sample. A true best-of-N provider should be supplied as a separate plugin.
    """

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
            manifest_cases = load_dataset_manifest(manifest_path)
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
                            "model",
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
                )
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


class MultiPanelProvider:
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

        manifest_cases = load_dataset_manifest(request.dataset_manifest_path)
        selected_case = select_case(manifest_cases, request.spec.case_id)
        verify_case_metadata(
            selected_case,
            panel_count=request.spec.panel_count,
            split=request.spec.split,
        )
        case = selected_case.payload
        raw_manifest = case.get("multi_panel_manifest")
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
            )
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
            )
        if rounds < 1:
            raise ProviderExecutionError("Multi-panel rounds must be positive")

        output_dir = request.output_dir / "multi_panel"
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
                            "model",
                        )
                    ),
                    seed=request.spec.seed + request.call_index - 1,
                    temperature=request.spec.method_config.get("temperature"),
                    memory_mode=memory_mode,
                    base_dir=manifest_base_dir,
                )
        finally:
            single_chain_runner._MODEL_CLIENT = old_model_client
            single_chain_runner._LLM_CLIENT = old_compat_client
        programmatic = result.get("programmatic_evaluation")
        if not isinstance(programmatic, Mapping):
            raise ProviderExecutionError(
                "Multi-panel run produced no programmatic evaluation"
            )
        panel_fidelity = programmatic.get("panel_fidelity") or {}
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
                "Multi-panel run has no applicable fidelity checks"
            )
        metrics: Dict[str, float] = {
            "data_fidelity": numerator / denominator,
            "execution_success": 1.0,
        }
        cohesion_ratio = (programmatic.get("cohesion") or {}).get("ratio")
        if isinstance(cohesion_ratio, (int, float)):
            metrics["series_cohesion"] = float(cohesion_ratio)
        visual_scores = [
            float((panel["result"].get("scores") or {}).get("visual_form", 0.0))
            for panel in result["panels"].values()
            if isinstance(panel.get("result"), Mapping)
        ]
        if visual_scores:
            metrics["visual_form"] = sum(visual_scores) / len(visual_scores)

        artifacts = {
            "output": str(output_dir),
            "render": str(result["combined_figure_path"]),
            "result": str(result["result_path"]),
            "programmatic_evaluation": str(
                result["programmatic_evaluation_path"]
            ),
            "memory_snapshot": str(result["shared_memory_snapshot_path"]),
            "memory_trace": str(result["shared_memory_trace_path"]),
            "schedule_trace": str(result["schedule_trace_path"]),
        }
        candidate = CandidateResult(
            metrics=metrics,
            render_count=sum(int(value) for value in result["render_counts"].values()),
            artifacts=artifacts,
            metadata={
                "panel_count": panel_count,
                "rounds": rounds,
                "memory_mode": memory_mode,
                "seed": request.spec.seed + request.call_index - 1,
                "served_models": sorted(
                    {
                        str(
                            metadata.get("model")
                        )
                        for panel in result["panels"].values()
                        if isinstance(panel.get("result"), Mapping)
                        for stage in (
                            panel["result"].get("stages") or {}
                        ).values()
                        if isinstance(stage, Mapping)
                        for metadata in [stage.get("model_metadata") or {}]
                        if metadata.get("model")
                    }
                ),
            },
        )
        if request.spec.schedule == "best_of_n":
            return candidate
        return ProviderBatch(candidates=(candidate,), stop=True)
