from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests
from PIL import Image

from app.services.model_client import (
    ModelClient,
    ModelClientError,
    ModelConfig,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}", response=self)

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def config(**overrides: Any) -> ModelConfig:
    values = {
        "base_url": "http://example.test:4142",
        "api_key": "secret",
        "model": "gpt-5.6-sol",
        "retries": 0,
        "reasoning_effort": "max",
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_generate_json_uses_real_auth_and_model_safe_payload() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "id": "request-1",
                    "model": "gpt-5.6-sol",
                    "choices": [{"message": {"content": '{"ok": true}'}}],
                    "usage": {"input_tokens": 3},
                }
            )
        ]
    )
    result = ModelClient(config(), session=session).generate_json(
        [{"role": "user", "content": "json"}]
    )

    assert result.value == {"ok": True}
    call = session.calls[0]
    assert call["url"] == "http://example.test:4142/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["headers"]["x-api-key"] == "secret"
    assert call["json"]["reasoning_effort"] == "max"
    assert "temperature" not in call["json"]
    assert result.stop_reason is None


def test_generate_json_parses_fenced_json() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "choices": [
                        {"message": {"content": "```json\n{\"value\": 4}\n```"}}
                    ]
                }
            )
        ]
    )
    result = ModelClient(config(reasoning_effort=None), session=session).generate_json(
        [{"role": "user", "content": "json"}]
    )
    assert result.value == {"value": 4}


def test_generate_json_uses_last_complete_object() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "```json\n{\"rubric\": true}\n```"
                                    "{\"visual_form\": 0.5,"
                                    " \"diagnostics\": [\"ok\"]}"
                                )
                            }
                        }
                    ]
                }
            )
        ]
    )
    result = ModelClient(config(reasoning_effort=None), session=session).generate_json(
        [{"role": "user", "content": "json"}]
    )
    assert result.value == {
        "visual_form": 0.5,
        "diagnostics": ["ok"],
    }


def test_generate_json_accepts_line_comments_outside_strings() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "```json\n"
                                    "{\n"
                                    '  "url": "https://example.test/a//b",\n'
                                    '  "slots": {\n'
                                    '    "spec.compose": {"overlays": []} '
                                    "// model rationale\n"
                                    "  }\n"
                                    "}\n"
                                    "```"
                                )
                            }
                        }
                    ]
                }
            )
        ]
    )

    result = ModelClient(
        config(reasoning_effort=None),
        session=session,
    ).generate_json([{"role": "user", "content": "json"}])

    assert result.value == {
        "url": "https://example.test/a//b",
        "slots": {"spec.compose": {"overlays": []}},
    }


def test_sampling_parameters_are_model_compatible() -> None:
    session = FakeSession(
        [
            FakeResponse(
                {
                    "choices": [{"message": {"content": '{"ok": true}'}}],
                    "copilot_usage": {"total_nano_aiu": 12},
                }
            )
        ]
    )
    result = ModelClient(config(), session=session).generate_json(
        [{"role": "user", "content": "json"}],
        model="gpt-4o-mini",
        temperature=0.7,
        seed=3,
    )
    payload = session.calls[0]["json"]
    assert payload["temperature"] == 0.7
    assert payload["seed"] == 3
    assert "reasoning_effort" not in payload
    assert result.usage == {"total_nano_aiu": 12}


def test_vision_uses_anthropic_image_format(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image)
    session = FakeSession(
        [
            FakeResponse(
                {
                    "id": "vision-1",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": '{"color": "red"}'}],
                }
            )
        ]
    )
    result = ModelClient(config(), session=session).evaluate_image_json(
        "identify", image, model="claude-sonnet-4.6"
    )

    assert result.value == {"color": "red"}
    call = session.calls[0]
    assert call["url"] == "http://example.test:4142/v1/messages"
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    source = call["json"]["messages"][0]["content"][0]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert source["data"]


def test_vision_uses_detected_media_type_not_suffix(tmp_path: Path) -> None:
    image = tmp_path / "mislabeled.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image, format="WEBP")
    session = FakeSession(
        [
            FakeResponse(
                {
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": '{"ok": true}'}],
                }
            )
        ]
    )
    ModelClient(config(), session=session).evaluate_image_json(
        "inspect",
        image,
        model="claude-sonnet-4.6",
    )
    source = session.calls[0]["json"]["messages"][0]["content"][0]["source"]
    assert source["media_type"] == "image/webp"


def test_invalid_json_fails_explicitly() -> None:
    session = FakeSession(
        [FakeResponse({"choices": [{"message": {"content": "not json"}}]})]
    )
    with pytest.raises(ModelClientError, match="JSON"):
        ModelClient(config(), session=session).generate_json(
            [{"role": "user", "content": "json"}]
        )


def test_http_failure_is_not_success() -> None:
    session = FakeSession([FakeResponse({"error": "denied"}, status=401)])
    with pytest.raises(ModelClientError, match="HTTP 401.*denied"):
        ModelClient(config(), session=session).generate_json(
            [{"role": "user", "content": "json"}]
        )


def test_missing_environment_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "MODEL_API_BASE",
        "ANTHROPIC_BASE_URL",
        "LLM_API_BASE",
        "OPENAI_BASE_URL",
        "MODEL_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ModelClientError, match="Missing model configuration"):
        ModelConfig.from_env()


def test_api_root_is_not_duplicated() -> None:
    session = FakeSession(
        [FakeResponse({"choices": [{"message": {"content": json.dumps({"ok": 1})}}]})]
    )
    ModelClient(config(base_url="http://example.test/v1"), session=session).generate_json(
        [{"role": "user", "content": "json"}]
    )
    assert session.calls[0]["url"] == "http://example.test/v1/chat/completions"
