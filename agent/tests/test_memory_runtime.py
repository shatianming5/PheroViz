from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

import app.services.single_chain_runner as single_runner
from app.services.model_client import ModelClientError, ModelResponse
from app.services.multi_panel_runner import run_multi_panel
from app.services.pheromones import ConstraintRecord, PersistentMemory, Scope


def _disable_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "MODEL_API_BASE",
        "ANTHROPIC_BASE_URL",
        "LLM_API_BASE",
        "OPENAI_BASE_URL",
        "MODEL_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "VLM_API_KEY",
        "LLM_MODEL",
        "VLM_MODEL",
        "FORCE_ALL_ROUNDS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(single_runner, "_ENV_LOADED", True)
    monkeypatch.setattr(single_runner, "_MODEL_CLIENT", None)
    monkeypatch.setattr(single_runner, "_LLM_CLIENT", None)
    monkeypatch.setenv("PHEROVIZ_RENDER_TIMEOUT", "120")

    def forbidden_client() -> Any:
        raise AssertionError("offline first round must not construct ModelClient")

    monkeypatch.setattr(single_runner, "_get_model_client", forbidden_client)


def _write_csv(path: Path, x_name: str, y_name: str) -> None:
    pd.DataFrame({x_name: ["A", "B", "C"], y_name: [1, 2, 3]}).to_csv(
        path, index=False
    )


class FakeStageClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        prompt = str(messages[-1]["content"])
        if "(L1)" in prompt:
            slots = {"spec.theme_defaults": "return spec"}
        elif "(L2)" in prompt:
            slots = {"data.prepare": "return df"}
        elif "(L3)" in prompt:
            slots = {"marks.line.main": "ax.plot([], []); return []"}
        elif "(L4)" in prompt:
            slots = {"axes.labels": "return None"}
        else:
            raise AssertionError(f"unknown stage prompt: {prompt[:80]}")
        self.calls.append({"messages": messages, **kwargs})
        return ModelResponse(
            value={"slots": slots, "notes": "fake model"},
            model="fake",
            request_id=f"request-{len(self.calls)}",
            usage={},
            stop_reason="end_turn",
            latency_seconds=0.0,
        )


def _stub_render_and_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_execute(
        py_code: str,
        df: Any,
        intent: dict[str, Any],
        ctx: dict[str, Any],
        out_png: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        del py_code, df, intent, timeout_s
        output = Path(out_png)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (255, 255, 255)).save(output)
        return {
            "ok": True,
            "png_path": str(output),
            "stderr": "",
            "ctx": dict(ctx),
        }

    def fake_judge(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "visual_form": 0.2,
            "data_fidelity": 0.2,
            "diagnostics": [],
        }

    monkeypatch.setattr(single_runner, "execute_script", fake_execute)
    monkeypatch.setattr(single_runner, "judge", fake_judge)


def test_offline_single_chain_writes_memory_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    data_path = tmp_path / "panel.csv"
    _write_csv(data_path, "category", "value")
    run_dir = tmp_path / "run"

    result = single_runner.run_chain(
        str(data_path),
        "offline chart",
        "bar",
        rounds=1,
        run_dir=run_dir,
    )

    assert Path(result["png_path"]).is_file()
    assert Path(result["artifact_path"]).is_file()
    assert Path(result["memory"]["snapshot_path"]).is_file()
    assert Path(result["memory"]["trace_path"]).is_file()
    compatibility_path = Path(result["memory"]["compatibility_path"])
    assert compatibility_path.name == "pheromones.json"
    assert compatibility_path.is_file()
    assert isinstance(json.loads(compatibility_path.read_text(encoding="utf-8")), list)
    assert result["memory"]["write"]["constraint_ids"]
    assert result["memory"]["write"]["patch_ids"]
    snapshot = json.loads(
        Path(result["memory"]["snapshot_path"]).read_text(encoding="utf-8")
    )
    assert snapshot["constraints"]
    assert snapshot["patch_templates"]


def test_single_chain_persists_programmatic_fidelity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    data_path = tmp_path / "panel.csv"
    _write_csv(data_path, "category", "value")
    expectation = {
        "schema_version": "1.1.0",
        "panels": [
            {
                "panel_id": "panel-0",
                "axis_index": 0,
                "series": [
                    {
                        "series_id": "value",
                        "kind": "bar",
                        "x": "category",
                        "value": "value",
                    }
                ],
            }
        ],
        "panel_groups": [],
    }

    result = single_runner.run_chain(
        str(data_path),
        "programmatic chart",
        "bar",
        rounds=1,
        intent={"x": "category", "y": "value"},
        run_dir=tmp_path / "programmatic-run",
        evaluation_expectation=expectation,
    )

    assert result["programmatic_evaluation"]["fidelity"]["ratio"] == 1.0
    assert result["scores"]["data_fidelity"] == 1.0
    assert Path(result["programmatic_evaluation_path"]).is_file()


def test_multi_panel_round_robin_reuses_shared_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, "category", "value")
    _write_csv(right, "label", "amount")
    manifest = {
        "panel_group": "figure-1",
        "rounds": 1,
        "panels": [
            {
                "id": "left",
                "data_path": str(left),
                "user_goal": "left panel",
                "chart_family": "bar",
            },
            {
                "id": "right",
                "data_path": str(right),
                "user_goal": "right panel",
                "chart_family": "bar",
            },
        ],
    }

    result = run_multi_panel(manifest, output_dir=tmp_path / "multi")
    assert result["render_counts"] == {"left": 1, "right": 1}
    assert Path(result["pheromones_path"]).is_file()
    schedule = json.loads(
        Path(result["schedule_trace_path"]).read_text(encoding="utf-8")
    )
    assert [entry["panel_id"] for entry in schedule] == ["left", "right"]

    right_result = result["panels"]["right"]["result"]
    reused = set(right_result["memory"]["reused_record_ids"])
    assert reused
    shared = PersistentMemory.load(result["shared_memory_snapshot_path"])
    left_ids = {
        record.id
        for record in shared.constraints.values()
        if record.provenance.get("panel_id") == "left"
        and record.scope.value == "panel_group"
    }
    assert reused & left_ids
    assert right_result["stages"]["L1"]["payload"]["memory_context"]["constraints"]


def test_multi_panel_programmatic_cohesion_and_composite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, "category", "value")
    _write_csv(right, "category", "value")
    panels = []
    expectations = []
    for panel_id, path in (("left", left), ("right", right)):
        panels.append(
            {
                "id": panel_id,
                "data_path": str(path),
                "user_goal": f"{panel_id} panel",
                "chart_family": "bar",
                "intent": {"x": "category", "y": "value"},
            }
        )
        expectations.append(
            {
                "panel_id": panel_id,
                "axis_index": 0,
                "series": [
                    {
                        "series_id": "value",
                        "kind": "bar",
                        "x": "category",
                        "value": "value",
                    }
                ],
            }
        )
    manifest = {
        "panel_group": "figure-programmatic",
        "rounds": 1,
        "layout": {"columns": 2},
        "panels": panels,
        "evaluation_expectation": {
            "schema_version": "1.1.0",
            "panels": expectations,
            "panel_groups": [
                {
                    "group_id": "shared-scale",
                    "panels": ["left", "right"],
                    "checks": {"shared_y_scale": True},
                }
            ],
        },
    }

    result = run_multi_panel(
        manifest,
        output_dir=tmp_path / "multi-programmatic",
    )

    assert Path(result["combined_figure_path"]).is_file()
    assert Path(result["programmatic_evaluation_path"]).is_file()
    assert result["programmatic_evaluation"]["cohesion"]["ratio"] == 1.0
    assert set(result["programmatic_evaluation"]["panel_fidelity"]) == {
        "left",
        "right",
    }


def test_ephemeral_memory_is_shared_within_round_and_cleared_between_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    _stub_render_and_judge(monkeypatch)
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, "category", "value")
    _write_csv(right, "category", "value")
    client = FakeStageClient()
    manifest = {
        "panel_group": "ephemeral-figure",
        "rounds": 2,
        "initial_generation": "model",
        "memory_mode": "ephemeral",
        "panels": [
            {
                "id": "left",
                "data_path": str(left),
                "user_goal": "left",
                "chart_family": "line",
            },
            {
                "id": "right",
                "data_path": str(right),
                "user_goal": "right",
                "chart_family": "line",
            },
        ],
    }

    result = run_multi_panel(
        manifest,
        output_dir=tmp_path / "ephemeral",
        model_client=client,  # type: ignore[arg-type]
    )
    left_rounds = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in result["panels"]["left"]["artifacts"]
    ]
    right_rounds = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in result["panels"]["right"]["artifacts"]
    ]

    assert right_rounds[0]["memory"]["reused_record_ids"]
    assert left_rounds[1]["memory"]["reused_record_ids"] == []
    assert right_rounds[1]["memory"]["reused_record_ids"]
    shared_trace = json.loads(
        Path(result["shared_memory_trace_path"]).read_text(encoding="utf-8")
    )
    assert sum(event["event"] == "memory_cleared" for event in shared_trace) == 1


def test_requested_programmatic_evaluation_never_reuses_stale_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    data_path = tmp_path / "panel.csv"
    _write_csv(data_path, "category", "value")

    def failed_execute(
        py_code: str,
        df: Any,
        intent: dict[str, Any],
        ctx: dict[str, Any],
        out_png: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        del py_code, df, intent, out_png, timeout_s
        stale_ctx = dict(ctx)
        stale_ctx["_programmatic_evaluation"] = {
            "fidelity": {"ratio": 1.0}
        }
        stale_ctx["_programmatic_evaluation_round_token"] = 0
        return {
            "ok": False,
            "png_path": None,
            "stderr": "intentional render failure",
            "ctx": stale_ctx,
        }

    monkeypatch.setattr(single_runner, "execute_script", failed_execute)
    expectation = {
        "schema_version": "1.1.0",
        "panels": [
            {
                "panel_id": "panel-0",
                "series": [
                    {
                        "series_id": "value",
                        "kind": "bar",
                        "x": "category",
                        "value": "value",
                    }
                ],
            }
        ],
        "panel_groups": [],
    }
    with pytest.raises(RuntimeError, match="did not produce a result"):
        single_runner.run_chain(
            str(data_path),
            "must fail",
            "bar",
            rounds=1,
            run_dir=tmp_path / "failed",
            evaluation_expectation=expectation,
        )


def test_later_round_model_failure_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    data_path = tmp_path / "panel.csv"
    _write_csv(data_path, "category", "value")
    monkeypatch.setenv("FORCE_ALL_ROUNDS", "1")

    def missing_client() -> Any:
        raise ModelClientError("Missing model configuration")

    monkeypatch.setattr(single_runner, "_get_model_client", missing_client)
    with pytest.raises(ModelClientError, match="Missing model configuration"):
        single_runner.run_chain(
            str(data_path),
            "requires a second round",
            "bar",
            rounds=2,
            run_dir=tmp_path / "run",
        )


def test_model_prompt_receives_memory_context() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        def generate_json(
            self,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> ModelResponse:
            del kwargs
            self.messages = messages
            return ModelResponse(
                value={
                    "slots": {"spec.theme_defaults": "return spec"},
                    "notes": "ok",
                },
                model="fake",
                request_id="request-1",
                usage={},
                stop_reason="end_turn",
                latency_seconds=0.0,
            )

    client = FakeClient()
    single_runner._llm_generate_slots(
        "L1",
        {
            "slot_keys": ["spec.theme_defaults"],
            "data_profile": {},
            "intent": {},
            "spec": {},
            "memory_context": {
                "constraints": {
                    "theme.palette_global": {
                        "value": "viridis",
                        "hard": False,
                    }
                },
                "eligible_patch_templates": [],
            },
        },
        model_client=client,  # type: ignore[arg-type]
    )
    prompt = str(client.messages[-1]["content"])
    assert "Persistent memory context" in prompt
    assert "theme.palette_global" in prompt


def test_l1_structured_slots_are_compiled_without_eval() -> None:
    class StructuredL1Client:
        def generate_json(self, messages, **kwargs):
            del messages, kwargs
            return ModelResponse(
                value={
                    "slots": {
                        "spec.compose": {
                            "canvas": {"width": 640, "height": 480, "dpi": 100},
                            "theme": {"palette_global": "tab10"},
                            "layout": {},
                            "scales": {},
                            "overlays": [],
                        },
                        "spec.theme_defaults": {"font": "Arial"},
                    },
                    "notes": "structured L1",
                },
                model="fake",
                request_id="structured-l1",
                usage={},
                stop_reason="end_turn",
                latency_seconds=0.0,
            )

    result = single_runner._llm_generate_slots(
        "L1",
        {
            "slot_keys": ["spec.compose", "spec.theme_defaults"],
            "data_profile": {},
            "intent": {},
            "spec": {},
            "memory_context": {},
        },
        model_client=StructuredL1Client(),  # type: ignore[arg-type]
    )

    assert result["slots"]["spec.compose"].startswith("return {")
    assert "theme.update" in result["slots"]["spec.theme_defaults"]


def test_l1_empty_theme_string_preserves_spec() -> None:
    class EmptyThemeClient:
        def generate_json(self, messages, **kwargs):
            del messages, kwargs
            return ModelResponse(
                value={
                    "slots": {
                        "spec.compose": "return spec",
                        "spec.theme_defaults": "{}",
                    }
                },
                model="fake",
                request_id="empty-theme",
                usage={},
                stop_reason="end_turn",
                latency_seconds=0.0,
            )

    result = single_runner._llm_generate_slots(
        "L1",
        {
            "slot_keys": ["spec.compose", "spec.theme_defaults"],
            "data_profile": {},
            "intent": {},
            "spec": {},
            "memory_context": {},
        },
        model_client=EmptyThemeClient(),  # type: ignore[arg-type]
    )
    assert result["slots"]["spec.theme_defaults"] == "return spec"


def test_model_initial_generation_calls_all_stages_with_sampling_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    _stub_render_and_judge(monkeypatch)
    data_path = tmp_path / "panel.csv"
    _write_csv(data_path, "category", "value")
    client = FakeStageClient()

    result = single_runner.run_chain(
        str(data_path),
        "model candidate",
        "line",
        rounds=1,
        run_dir=tmp_path / "run",
        model_client=client,  # type: ignore[arg-type]
        initial_generation="model",
        seed=17,
        temperature=0.65,
        memory_mode="none",
    )

    assert len(client.calls) == 4
    assert {call["seed"] for call in client.calls} == {17}
    assert {call["temperature"] for call in client.calls} == {0.65}
    assert all(
        stage["prompt"] != "DEFAULT_V2"
        for stage in result["stages"].values()
    )
    assert result["run_config"] == {
        "initial_generation": "model",
        "seed": 17,
        "temperature": 0.65,
        "memory_mode": "none",
    }


@pytest.mark.parametrize(
    ("mode", "expected_context_key", "writes_constraints", "writes_patches"),
    [
        ("none", None, False, False),
        ("ephemeral", "empty_typed", True, True),
        ("untyped", "untyped_log", False, False),
        ("constraints", "constraints", True, False),
        ("patches", "eligible_patch_templates", False, True),
        ("full", "constraints", True, True),
    ],
)
def test_memory_modes_control_real_reads_and_writes(
    mode: str,
    expected_context_key: str | None,
    writes_constraints: bool,
    writes_patches: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_network(monkeypatch)
    _stub_render_and_judge(monkeypatch)
    monkeypatch.setenv("FORCE_ALL_ROUNDS", "1")
    data_path = tmp_path / f"{mode}.csv"
    _write_csv(data_path, "category", "value")
    client = FakeStageClient()
    supplied_memory = PersistentMemory()
    if mode == "none":
        supplied_memory.add_constraint(
            ConstraintRecord(
                id="must-not-be-read",
                scope=Scope.FIGURE,
                level=1,
                slot="theme.palette_global",
                value="plasma",
                hard=False,
                provenance={"source": "previous-run"},
                validated_executable=True,
            )
        )

    result = single_runner.run_chain(
        str(data_path),
        f"{mode} memory",
        "line",
        rounds=2,
        run_dir=tmp_path / mode,
        model_client=client,  # type: ignore[arg-type]
        initial_generation="model",
        seed=3,
        temperature=0.2,
        memory_mode=mode,
        memory=supplied_memory,
    )

    assert len(client.calls) == 8
    context = result["stages"]["L1"]["payload"]["memory_context"]
    if expected_context_key is None:
        assert context == {}
    elif expected_context_key == "empty_typed":
        assert context["constraints"] == {}
        assert context["eligible_patch_templates"] == []
    else:
        assert context[expected_context_key]
    if mode == "patches":
        assert context["constraints"] == {}
    if mode == "constraints":
        assert context["eligible_patch_templates"] == []
    if mode == "full":
        assert context["eligible_patch_templates"]

    write = result["memory"]["write"]
    assert bool(write["constraint_ids"]) is writes_constraints
    assert bool(write["patch_ids"]) is writes_patches
    if mode == "untyped":
        assert write["untyped_entries_written"] == 4
        assert Path(result["memory"]["untyped_path"]).is_file()
    if mode == "none":
        assert set(supplied_memory.constraints) == {"must-not-be-read"}
        assert not supplied_memory.patches
