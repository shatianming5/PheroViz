"""Multi-panel comparison wrapper for single-panel external baselines.

This module deliberately lives apart from ``external_baselines.py`` so the frozen
single-panel external-baseline contract there (its byte hash is pinned by the
portable single-panel subtrack specs) stays untouched. It adds a multi-panel
fan-out provider that runs an existing single-panel baseline once per panel of a
sealed multi-panel case and aggregates the per-panel programmatic evaluations
with the exact arithmetic PheroViz's own multi-panel provider uses, so the
head-to-head ``data_fidelity`` + ``series_cohesion`` are produced by the same
trusted ``app.evaluation`` evaluator on the identical sealed benchmark.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .external_baselines import MatPlotAgentProvider, NvAgentProvider
from .manifest import (
    DatasetCase,
    ManifestError,
    load_dataset_manifest,
    resolve_case_data_path,
    select_case,
    verify_case_data_files,
    verify_case_metadata,
)
from .models import sha256_file, slug_identifier, write_json_atomic
from .providers import (
    CandidateResult,
    GenerationRequest,
    ProviderExecutionError,
)


class MultiPanelExternalBaselineProvider:
    """Fair multi-panel comparison wrapper for single-panel external baselines.

    External chart-generation baselines (MatPlotAgent, nvAgent) natively target a
    single panel, while the C2 headline testbed is multi-panel. This provider runs
    the existing single-panel baseline provider once per panel of a sealed
    multi-panel case, then aggregates the per-panel programmatic evaluations using
    the SAME arithmetic the PheroViz multi-panel provider uses:

      * ``data_fidelity``   = sum(panel_numerator) / sum(panel_denominator)
        (micro-average over per-panel applicable fidelity checks -- identical to
        ``_multi_panel_checkpoint_candidate`` in ``experiments.providers``).
      * ``series_cohesion`` = ``evaluate_cohesion`` over
        ``combine_figure_manifests`` of the per-panel figure manifests against the
        parent case's full multi-panel expectation -- identical to
        ``_combined_programmatic_evaluation`` in ``app.services.multi_panel_runner``.

    Because the panel manifests, fidelity numerators/denominators, and cohesion are
    produced by the *same* trusted ``app.evaluation`` evaluator that scores
    PheroViz, the result is an apples-to-apples ``data_fidelity`` +
    ``series_cohesion`` on the identical sealed benchmark. The sealed multi-panel
    parent case is loaded and fully verified here (panel data SHA-256s checked);
    each per-panel sub-case is a deterministic, unsealed derivation of a verified
    panel (same ``data_path``, ``sheet``, and sealed ``data_sha256``).
    """

    test_only = False

    _INNER_PROVIDERS: Dict[str, Any] = {
        "matplotagent": MatPlotAgentProvider,
        "nvagent": NvAgentProvider,
    }

    def __init__(
        self,
        baseline: str,
        repo_path: Path | str,
        *,
        python_executable: str = sys.executable,
        timeout_seconds: float = 1800.0,
        check_dependencies: bool = True,
        environ: Optional[Mapping[str, str]] = None,
        manifest_data_root: Optional[str] = None,
        runtime_repo_root: Optional[str] = None,
    ) -> None:
        key = str(baseline).strip().lower()
        inner_cls = self._INNER_PROVIDERS.get(key)
        if inner_cls is None:
            raise ProviderExecutionError(
                "Unsupported multi-panel baseline "
                f"{baseline!r}; expected one of "
                f"{sorted(self._INNER_PROVIDERS)}"
            )
        self._inner = inner_cls(
            repo_path,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            check_dependencies=check_dependencies,
            environ=environ,
        )
        self.baseline_key = key
        self.name = f"multipanel_{self._inner.name}"
        self.test_only = bool(getattr(self._inner, "test_only", False))
        self._manifest_data_root = (
            str(manifest_data_root) if manifest_data_root is not None else None
        )
        self._runtime_repo_root = (
            str(runtime_repo_root) if runtime_repo_root is not None else None
        )

    def check_available(self) -> None:
        self._inner.check_available()

    def _parent_case(self, request: GenerationRequest) -> DatasetCase:
        try:
            cases = load_dataset_manifest(
                request.dataset_manifest_path,
                dataset_mode=request.spec.dataset_mode,
                manifest_data_root=self._manifest_data_root,
                runtime_repo_root=self._runtime_repo_root,
            )
            case = select_case(cases, request.spec.case_id)
            verify_case_metadata(
                case,
                panel_count=request.spec.panel_count,
                split=request.spec.split,
            )
            verify_case_data_files(
                case,
                manifest_path=request.dataset_manifest_path,
                manifest_data_root=self._manifest_data_root,
                runtime_repo_root=self._runtime_repo_root,
            )
        except ManifestError as exc:
            raise ProviderExecutionError(str(exc)) from exc
        return case

    def _materialize_panel_manifest(
        self,
        *,
        request: GenerationRequest,
        parent: DatasetCase,
        panel: Mapping[str, Any],
        panel_id: str,
        panel_expectation: Mapping[str, Any],
        full_expectation: Mapping[str, Any],
        child_id: str,
        panel_dir: Path,
    ) -> Path:
        raw_data_path = panel.get("data_path")
        if not isinstance(raw_data_path, str) or not raw_data_path.strip():
            raise ProviderExecutionError(
                f"case_id {parent.case_id!r} panel {panel_id!r} has no data_path"
            )
        resolved_data = resolve_case_data_path(
            raw_data_path,
            manifest_path=request.dataset_manifest_path,
            manifest_data_root=self._manifest_data_root,
            runtime_repo_root=self._runtime_repo_root,
        )
        sub_case: Dict[str, Any] = {
            "case_id": child_id,
            "chart_family": panel.get("chart_family"),
            "input_track": self._inner.definition.input_track,
            "panel_count": 1,
            "split": parent.split,
            "data_path": str(resolved_data),
            "data_sha256": panel.get("data_sha256"),
            "sheet": panel.get("sheet"),
            "user_goal": panel.get("user_goal"),
            "intent": panel.get("intent"),
            "evaluation_expectation": {
                "schema_version": full_expectation.get("schema_version"),
                "panels": [dict(panel_expectation)],
            },
        }
        sub_manifest_path = panel_dir / "panel_manifest.jsonl"
        sub_manifest_path.write_text(
            json.dumps(sub_case, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return sub_manifest_path

    def _run_panel(
        self,
        *,
        request: GenerationRequest,
        parent: DatasetCase,
        panel: Mapping[str, Any],
        panel_id: str,
        panel_expectation: Mapping[str, Any],
        full_expectation: Mapping[str, Any],
        work_dir: Path,
    ) -> tuple[str, CandidateResult]:
        child_id = f"{parent.case_id}__panel-{slug_identifier(panel_id)}"
        panel_dir = work_dir / f"panel__{slug_identifier(panel_id)}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        sub_manifest_path = self._materialize_panel_manifest(
            request=request,
            parent=parent,
            panel=panel,
            panel_id=panel_id,
            panel_expectation=panel_expectation,
            full_expectation=full_expectation,
            child_id=child_id,
            panel_dir=panel_dir,
        )
        sub_spec = replace(
            request.spec,
            run_name=f"multipanelsub__case-{slug_identifier(child_id)}",
            case_id=child_id,
            panel_count=1,
            split=parent.split,
            dataset_manifest_path=str(sub_manifest_path),
            dataset_manifest_hash=sha256_file(sub_manifest_path),
            dataset_mode="legacy",
        )
        inner_output = panel_dir / "inner"
        inner_output.mkdir(parents=True, exist_ok=True)
        sub_request = replace(
            request,
            spec=sub_spec,
            dataset_manifest_path=sub_manifest_path,
            output_dir=inner_output,
            call_index=0,
            remaining_renders=1,
        )
        result = self._inner.generate(sub_request)
        if isinstance(result, CandidateResult):
            return child_id, result
        # Single-panel baseline providers always return a CandidateResult;
        # a ProviderBatch would be a contract violation.
        raise ProviderExecutionError(
            f"panel {panel_id!r} inner provider returned an unexpected batch"
        )

    @staticmethod
    def _read_panel_programmatic(
        panel_id: str,
        result: CandidateResult,
    ) -> Mapping[str, Any]:
        raw_path = result.artifacts.get("programmatic_evaluation")
        if not raw_path:
            raise ProviderExecutionError(
                f"panel {panel_id!r} produced no programmatic_evaluation artifact"
            )
        try:
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderExecutionError(
                f"panel {panel_id!r} programmatic evaluation is unreadable: "
                f"{exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ProviderExecutionError(
                f"panel {panel_id!r} programmatic evaluation is not an object"
            )
        return payload

    def generate(self, request: GenerationRequest) -> CandidateResult:
        try:
            from app.evaluation import (
                combine_figure_manifests,
                evaluate_cohesion,
            )
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ProviderExecutionError(
                "MultiPanelExternalBaselineProvider requires app.evaluation on "
                f"the Python path: {exc}"
            ) from exc

        parent = self._parent_case(request)
        payload = parent.payload
        panels = payload.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ProviderExecutionError(
                f"case_id {parent.case_id!r} has no panels to decompose"
            )
        full_expectation = payload.get("evaluation_expectation")
        if not isinstance(full_expectation, Mapping):
            raise ProviderExecutionError(
                f"case_id {parent.case_id!r} lacks an evaluation_expectation"
            )
        expectation_panels = {
            str(entry.get("panel_id")): entry
            for entry in full_expectation.get("panels", [])
            if isinstance(entry, Mapping)
        }
        metric_config = request.spec.metric_config.get("evaluator")
        if metric_config is not None and not isinstance(metric_config, Mapping):
            raise ProviderExecutionError(
                "metric_config.evaluator must be an object"
            )

        work_dir = request.output_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        panel_ids: list[str] = []
        panel_fidelity: Dict[str, Mapping[str, Any]] = {}
        panel_manifests: Dict[str, Mapping[str, Any]] = {}
        per_panel_meta: Dict[str, Any] = {}

        for index, panel in enumerate(panels):
            if not isinstance(panel, Mapping):
                raise ProviderExecutionError(
                    f"case_id {parent.case_id!r} panel[{index}] is invalid"
                )
            panel_id = str(panel.get("id") or "").strip()
            if not panel_id:
                raise ProviderExecutionError(
                    f"case_id {parent.case_id!r} panel[{index}] has no id"
                )
            if panel_id in panel_fidelity:
                raise ProviderExecutionError(
                    f"case_id {parent.case_id!r} has duplicate panel id "
                    f"{panel_id!r}"
                )
            panel_expectation = expectation_panels.get(panel_id)
            if panel_expectation is None:
                raise ProviderExecutionError(
                    f"case_id {parent.case_id!r} panel {panel_id!r} is absent "
                    "from evaluation_expectation.panels"
                )
            child_id, result = self._run_panel(
                request=request,
                parent=parent,
                panel=panel,
                panel_id=panel_id,
                panel_expectation=panel_expectation,
                full_expectation=full_expectation,
                work_dir=work_dir,
            )
            prog = self._read_panel_programmatic(panel_id, result)
            fidelity = prog.get("fidelity")
            manifest = prog.get("figure_manifest")
            if not isinstance(fidelity, Mapping) or not isinstance(
                manifest, Mapping
            ):
                raise ProviderExecutionError(
                    f"panel {panel_id!r} programmatic evaluation lacks fidelity "
                    "or figure_manifest"
                )
            panel_ids.append(panel_id)
            panel_fidelity[panel_id] = fidelity
            panel_manifests[panel_id] = manifest
            per_panel_meta[panel_id] = {
                "child_case_id": child_id,
                "fidelity_numerator": fidelity.get("numerator"),
                "fidelity_denominator": fidelity.get("denominator"),
                "fidelity_ratio": fidelity.get("ratio"),
                "fidelity_applicable": fidelity.get("applicable"),
                "programmatic_evaluation": result.artifacts.get(
                    "programmatic_evaluation"
                ),
                "render": result.artifacts.get("render"),
                "code": result.artifacts.get("code"),
            }

        numerator = sum(
            int(panel_fidelity[pid].get("numerator", 0)) for pid in panel_ids
        )
        denominator = sum(
            int(panel_fidelity[pid].get("denominator", 0)) for pid in panel_ids
        )
        if denominator <= 0:
            raise ProviderExecutionError(
                f"case_id {parent.case_id!r} has no applicable fidelity checks "
                "across panels"
            )

        try:
            combined_manifest = combine_figure_manifests(
                {pid: panel_manifests[pid] for pid in panel_ids}
            )
            cohesion = evaluate_cohesion(
                combined_manifest,
                full_expectation,
                metric_config,
            )
        except ValueError as exc:
            failure = ProviderExecutionError(
                f"case_id {parent.case_id!r} panel manifests are not composable "
                f"into a single figure: {exc}"
            )
            failure.failure_attribution = "method"
            raise failure from exc

        cohesion_dict = cohesion.to_dict()
        metrics: Dict[str, float] = {
            "data_fidelity": numerator / denominator,
            "execution_success": 1.0,
        }
        cohesion_ratio = cohesion_dict.get("ratio")
        if isinstance(cohesion_ratio, (int, float)) and not isinstance(
            cohesion_ratio, bool
        ):
            metrics["series_cohesion"] = float(cohesion_ratio)

        aggregate = {
            "schema_version": "1.0",
            "case_id": parent.case_id,
            "baseline": self.baseline_key,
            "inner_provider": self._inner.name,
            "external_repo": self._inner.definition.repo_url,
            "external_repo_commit": self._inner.definition.commit,
            "panel_ids": panel_ids,
            "aggregation": "sum_panel_numerator_over_sum_panel_denominator",
            "data_fidelity": {
                "numerator": numerator,
                "denominator": denominator,
                "ratio": numerator / denominator,
            },
            "cohesion": cohesion_dict,
            "panel_fidelity": {
                pid: dict(panel_fidelity[pid]) for pid in panel_ids
            },
            "combined_figure_manifest": combined_manifest.to_dict(),
        }
        aggregate_path = work_dir / "multipanel_programmatic_evaluation.json"
        write_json_atomic(aggregate_path, aggregate)

        artifacts: Dict[str, str] = {
            "multipanel_programmatic_evaluation": str(aggregate_path),
        }
        for pid in panel_ids:
            meta = per_panel_meta[pid]
            evaluation_artifact = meta.get("programmatic_evaluation")
            if evaluation_artifact:
                artifacts[
                    f"panel_{slug_identifier(pid)}_programmatic_evaluation"
                ] = evaluation_artifact
            render_artifact = meta.get("render")
            if render_artifact:
                artifacts[f"panel_{slug_identifier(pid)}_render"] = render_artifact

        metadata = {
            "baseline": self.baseline_key,
            "inner_provider": self._inner.name,
            "external_repo": self._inner.definition.repo_url,
            "external_repo_commit": self._inner.definition.commit,
            "panel_count": len(panel_ids),
            "panel_ids": panel_ids,
            "aggregation": "sum_panel_numerator_over_sum_panel_denominator",
            "cohesion_applicable": bool(cohesion_dict.get("applicable")),
            "panels": per_panel_meta,
        }
        return CandidateResult(
            metrics=metrics,
            render_count=len(panel_ids),
            artifacts=artifacts,
            metadata=metadata,
        )
