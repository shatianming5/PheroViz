from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from app.services.model_client import ModelClientError, ModelResponse


judge_module = importlib.import_module("app.services.judge")


class FakeVisionClient:
    def evaluate_image_json(self, prompt, image_path, **kwargs):
        assert "scientific chart" in prompt
        assert Path(image_path).is_file()
        assert kwargs["model"] == "claude-sonnet-4.6"
        return ModelResponse(
            value={
                "scores": {
                    "visual_form": 0.8,
                    "data_fidelity": 0.4,
                },
                "diagnostics": [
                    {
                        "slot": "axes.title",
                        "key": "title",
                        "hint": "shorten title",
                        "sev": 1,
                    }
                ],
                "notes": "fixture",
            },
            model="claude-sonnet-4-6",
            request_id="vision-1",
            usage={"input_tokens": 10},
            stop_reason="end_turn",
            latency_seconds=0.1,
        )


def test_judge_uses_shared_multimodal_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "chart.png"
    Image.new("RGB", (16, 16), "white").save(image)
    monkeypatch.setenv("VLM_MODEL", "claude-sonnet-4.6")
    monkeypatch.setattr(
        judge_module.ModelClient,
        "from_env",
        classmethod(lambda cls, model=None: FakeVisionClient()),
    )

    result = judge_module.judge(
        str(image),
        "",
        pd.DataFrame({"x": [1], "y": [2]}),
        {"overlays": [{"x": "x", "y": "y"}]},
    )

    assert result["visual_form"] == 0.8
    assert result["data_fidelity"] == 0.4
    assert result["diagnostics"][0]["slot"] == "axes.title"
    assert result["model_metadata"]["model"] == "claude-sonnet-4-6"


def test_required_vlm_failure_is_not_silently_heuristic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "chart.png"
    Image.new("RGB", (16, 16), "white").save(image)
    monkeypatch.setenv("VLM_MODEL", "claude-sonnet-4.6")
    monkeypatch.setenv("VLM_REQUIRED", "1")

    class FailingClient:
        def evaluate_image_json(self, *args, **kwargs):
            raise ModelClientError("intentional outage")

    monkeypatch.setattr(
        judge_module.ModelClient,
        "from_env",
        classmethod(lambda cls, model=None: FailingClient()),
    )
    with pytest.raises(ModelClientError, match="Required VLM judge failed"):
        judge_module.judge(
            str(image),
            "",
            pd.DataFrame({"x": [1], "y": [2]}),
            {"overlays": [{"x": "x", "y": "y"}]},
        )
