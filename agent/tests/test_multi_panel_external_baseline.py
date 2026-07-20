"""Offline unit tests for :class:`MultiPanelExternalBaselineProvider`.

These tests exercise the multi-panel fan-out wrapper without any model calls or
external baseline subprocess. A fake single-panel inner provider renders correct
(or deliberately broken) panels through the *real* ``app.evaluation`` evaluator,
writes the per-panel ``programmatic_evaluation.json`` exactly as the real
single-panel baseline driver does, and returns a ``CandidateResult``. The wrapper
must then reproduce PheroViz's own multi-panel arithmetic:

  * ``data_fidelity``   = sum(panel numerator) / sum(panel denominator)
  * ``series_cohesion`` = ``evaluate_cohesion`` over ``combine_figure_manifests``
    of the per-panel manifests against the parent's full expectation.

The point is to prove the wrapper is apples-to-apples with PheroViz using the
identical trusted evaluator, and that a non-composable set of panels is attributed
to the ``method`` (not the harness).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from app.evaluation import EXPECTATION_SCHEMA_VERSION, evaluate_figure
from experiments.external_baselines_multipanel import (
    MultiPanelExternalBaselineProvider,
)
from experiments.models import sha256_path
from experiments.providers import (
    CandidateResult,
    GenerationRequest,
    ProviderExecutionError,
)
from tests.test_experiment_support import make_spec, write_manifest


# --------------------------------------------------------------------------- #
# Fake single-panel inner provider
# --------------------------------------------------------------------------- #
class _FakeDefinition:
    input_track = "table_instruction"
    repo_url = "https://example.com/fake-baseline.git"
    commit = "0" * 40


class _FakeSinglePanelProvider:
    """Stands in for ``MatPlotAgentProvider``/``NvAgentProvider``.

    It reads the wrapper-materialized single-panel sub-manifest, renders one panel
    through ``evaluate_figure`` (the same evaluator PheroViz uses), and writes the
    resulting ``EvaluationResult.to_dict()`` to ``programmatic_evaluation.json`` --
    byte-for-byte the contract the real baseline driver satisfies.
    """

    name = "fake_single_panel"
    test_only = True

    def __init__(self, *, mode: str = "success") -> None:
        self.definition = _FakeDefinition()
        self.mode = mode
        self.calls: list[GenerationRequest] = []

    def check_available(self) -> None:  # pragma: no cover - trivial
        return None

    def generate(self, request: GenerationRequest) -> CandidateResult:
        self.calls.append(request)
        manifest_path = Path(request.dataset_manifest_path)
        line = manifest_path.read_text(encoding="utf-8").strip().splitlines()[0]
        case = json.loads(line)
        expectation = case["evaluation_expectation"]
        panel = expectation["panels"][0]
        panel_id = panel["panel_id"]
        series = panel["series"][0]

        frame = pd.read_csv(case["data_path"])
        x_values = list(frame[series["x"]])
        y_values = list(frame[series["y"]])
        # Break fidelity for panel-b under the "wrong" mode.
        if self.mode == "wrong" and panel_id.endswith("b"):
            y_values = [value + 100.0 for value in y_values]

        fig, ax = plt.subplots()
        ax.set_gid(panel_id)
        ax.plot(x_values, y_values, label=series["label"], color="#1f77b4")
        if panel.get("xlabel"):
            ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel.get("ylabel", "Value (mg)"))
        ax.legend()
        try:
            result = evaluate_figure(fig, frame, expectation)
        finally:
            plt.close(fig)

        payload = result.to_dict()
        if self.mode == "uncomposable":
            # Duplicate the single panel axis so the combined manifest contains
            # two axes with role == "panel" -- combine_figure_manifests must
            # reject it, and the wrapper must attribute the failure to "method".
            axes = payload["figure_manifest"]["axes"]
            axes.append(dict(axes[0]))

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prog_path = output_dir / "programmatic_evaluation.json"
        prog_path.write_text(json.dumps(payload), encoding="utf-8")
        render_path = output_dir / "render.png"
        render_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        ratio = payload["fidelity"].get("ratio")
        return CandidateResult(
            metrics={
                "data_fidelity": float(ratio) if ratio is not None else 0.0,
                "execution_success": 1.0,
            },
            render_count=1,
            artifacts={
                "programmatic_evaluation": str(prog_path),
                "render": str(render_path),
            },
            metadata={"panel_id": panel_id},
        )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _multi_panel_expectation() -> dict:
    panels = []
    for index, panel_id in enumerate(("panel-a", "panel-b")):
        panels.append(
            {
                "panel_id": panel_id,
                "axis_index": index,
                "ylabel": "Value (mg)",
                "y_unit": "mg",
                "y_scale": "linear",
                "series": [
                    {
                        "series_id": "signal",
                        "kind": "line",
                        "label": "signal",
                        "x": "x",
                        "y": "y",
                    }
                ],
            }
        )
    return {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": panels,
        "panel_groups": [
            {
                "group_id": "shared-measurement",
                "panels": ["panel-a", "panel-b"],
                "series": ["signal"],
                "checks": {
                    "shared_y_scale": True,
                    "shared_y_unit": True,
                    "legend_deduplicated": True,
                    "palette_consistent": True,
                },
            }
        ],
    }


def _build_parent_case(workspace: Path) -> dict:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    panels = []
    for panel_id in ("panel-a", "panel-b"):
        data_path = workspace / f"{panel_id}.csv"
        frame.to_csv(data_path, index=False)
        panels.append(
            {
                "id": panel_id,
                "chart_family": "line",
                "data_path": str(data_path),
                "data_sha256": sha256_path(data_path),
                "sheet": None,
                "intent": "trend",
                "user_goal": f"Plot {panel_id}",
            }
        )
    return {
        "case_id": "c2-parent",
        "panel_count": 2,
        "split": "test",
        "input_track": "table_instruction",
        "eligible_for_experiment": True,
        "panels": panels,
        "evaluation_expectation": _multi_panel_expectation(),
    }


def _make_provider(mode: str) -> MultiPanelExternalBaselineProvider:
    provider = MultiPanelExternalBaselineProvider.__new__(
        MultiPanelExternalBaselineProvider
    )
    inner = _FakeSinglePanelProvider(mode=mode)
    provider._inner = inner
    provider.baseline_key = "matplotagent"
    provider.name = f"multipanel_{inner.name}"
    provider.test_only = True
    provider._manifest_data_root = None
    provider._runtime_repo_root = None
    return provider


def _make_request(workspace: Path) -> GenerationRequest:
    manifest_path = write_manifest(workspace, [_build_parent_case(workspace)])
    spec = make_spec(
        workspace,
        run_name="multipanel-external",
        case_id="c2-parent",
        panel_count=2,
        split="test",
        budget_value=1,
        selection_metric="data_fidelity",
    )
    output_dir = workspace / "provider-output"
    output_dir.mkdir()
    return GenerationRequest(
        spec=spec,
        dataset_manifest_path=manifest_path,
        output_dir=output_dir,
        call_index=1,
        remaining_renders=1,
        remaining_seconds=None,
        deadline_monotonic=None,
        history=(),
        previous_candidate=None,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_all_panels_correct_micro_average_is_one(tmp_path):
    provider = _make_provider("success")
    request = _make_request(tmp_path)

    result = provider.generate(request)

    assert result.render_count == 2
    assert len(provider._inner.calls) == 2
    # Core wrapper contract: pooled (micro-average) fidelity over both panels.
    assert result.metrics["data_fidelity"] == pytest.approx(1.0)
    assert result.metrics["execution_success"] == 1.0
    # Cohesion is evaluated by the real app.evaluation cohesion evaluator over the
    # combined manifest; a per-panel fan-out legitimately need not score a perfect
    # 1.0 (e.g. duplicated per-panel legends). Assert it is a finite ratio in
    # [0, 1] and was actually computed against the parent's panel groups.
    cohesion = result.metrics["series_cohesion"]
    assert 0.0 <= cohesion <= 1.0
    assert result.metadata["cohesion_applicable"] is True

    aggregate_path = Path(result.artifacts["multipanel_programmatic_evaluation"])
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    # Micro-average identity: ratio == numerator / denominator.
    fidelity = aggregate["data_fidelity"]
    assert fidelity["numerator"] == fidelity["denominator"]
    assert fidelity["denominator"] > 0
    assert fidelity["ratio"] == pytest.approx(1.0)
    assert aggregate["panel_ids"] == ["panel-a", "panel-b"]
    assert (
        aggregate["aggregation"]
        == "sum_panel_numerator_over_sum_panel_denominator"
    )
    # The combined manifest that cohesion was scored on has exactly one panel
    # axis per parent panel (proves the per-panel manifests composed correctly).
    combined_axes = aggregate["combined_figure_manifest"]["axes"]
    panel_axes = [ax for ax in combined_axes if ax.get("role") == "panel"]
    assert {ax["panel_id"] for ax in panel_axes} == {"panel-a", "panel-b"}


def test_micro_average_penalizes_broken_panel(tmp_path):
    provider = _make_provider("wrong")
    request = _make_request(tmp_path)

    result = provider.generate(request)

    assert result.render_count == 2
    # panel-a full match, panel-b broken -> strictly between 0 and 1.
    fidelity = result.metrics["data_fidelity"]
    assert 0.0 < fidelity < 1.0

    aggregate_path = Path(result.artifacts["multipanel_programmatic_evaluation"])
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    panel_fidelity = aggregate["panel_fidelity"]
    # Micro-average must equal pooled numerator / pooled denominator, which is
    # NOT the mean of the two per-panel ratios in general.
    numerator = sum(int(panel_fidelity[p]["numerator"]) for p in ("panel-a", "panel-b"))
    denominator = sum(
        int(panel_fidelity[p]["denominator"]) for p in ("panel-a", "panel-b")
    )
    assert fidelity == pytest.approx(numerator / denominator)


def test_non_composable_panels_attributed_to_method(tmp_path):
    provider = _make_provider("uncomposable")
    request = _make_request(tmp_path)

    with pytest.raises(ProviderExecutionError) as excinfo:
        provider.generate(request)

    assert getattr(excinfo.value, "failure_attribution", None) == "method"
    assert "not composable" in str(excinfo.value)
