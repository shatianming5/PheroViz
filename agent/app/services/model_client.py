from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests


class ModelClientError(RuntimeError):
    """Raised when a model request cannot produce a valid response."""


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 180.0
    connect_timeout: float = 10.0
    retries: int = 2
    max_tokens: int = 2048
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls, *, model: str | None = None) -> "ModelConfig":
        base_url = (
            os.getenv("MODEL_API_BASE")
            or os.getenv("ANTHROPIC_BASE_URL")
            or os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        ).strip()
        api_key = (
            os.getenv("MODEL_API_KEY")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        selected_model = (model or os.getenv("LLM_MODEL") or "").strip()
        missing = [
            name
            for name, value in (
                ("MODEL_API_BASE/ANTHROPIC_BASE_URL/LLM_API_BASE", base_url),
                ("MODEL_API_KEY/ANTHROPIC_AUTH_TOKEN/LLM_API_KEY", api_key),
                ("LLM_MODEL", selected_model),
            )
            if not value
        ]
        if missing:
            raise ModelClientError(f"Missing model configuration: {', '.join(missing)}")

        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=selected_model,
            timeout=_env_float("LLM_TIMEOUT", 180.0, minimum=1.0),
            connect_timeout=_env_float("LLM_CONNECT_TIMEOUT", 10.0, minimum=1.0),
            retries=_env_int("LLM_RETRY", 2, minimum=0),
            max_tokens=_env_int("LLM_MAX_TOKENS", 2048, minimum=1),
            reasoning_effort=(os.getenv("LLM_REASONING_EFFORT") or "").strip() or None,
        )


@dataclass(frozen=True)
class ModelResponse:
    value: dict[str, Any]
    model: str
    request_id: str | None
    usage: dict[str, Any]
    latency_seconds: float


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(float(raw), minimum)
    except ValueError as exc:
        raise ModelClientError(f"{name} must be numeric, got {raw!r}") from exc


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(int(raw), minimum)
    except ValueError as exc:
        raise ModelClientError(f"{name} must be an integer, got {raw!r}") from exc


def _api_root(base_url: str) -> str:
    return base_url if base_url.rstrip("/").endswith("/v1") else f"{base_url.rstrip('/')}/v1"


def _parse_json_text(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ModelClientError("Model response did not contain a JSON object") from exc
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as nested_exc:
            raise ModelClientError("Model response contained invalid JSON") from nested_exc
    if not isinstance(value, dict):
        raise ModelClientError("Model response JSON must be an object")
    return value


class ModelClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self._session = session or requests.Session()

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        session: requests.Session | None = None,
    ) -> "ModelClient":
        return cls(ModelConfig.from_env(model=model), session=session)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }

    def generate_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ModelResponse:
        selected_model = model or self.config.model
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens or self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        effort = reasoning_effort or self.config.reasoning_effort
        if effort and selected_model.startswith("gpt-5"):
            payload["reasoning_effort"] = effort
        if temperature is not None and not selected_model.startswith("gpt-5"):
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed

        started = time.monotonic()
        data = self._post_json(
            f"{_api_root(self.config.base_url)}/chat/completions",
            payload,
            extra_headers=None,
        )
        choices = data.get("choices") or []
        if not choices:
            raise ModelClientError("Model response did not include choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            raise ModelClientError("Model response content was empty")
        return ModelResponse(
            value=_parse_json_text(content),
            model=str(data.get("model") or selected_model),
            request_id=_optional_str(data.get("id")),
            usage=dict(data.get("usage") or data.get("copilot_usage") or {}),
            latency_seconds=time.monotonic() - started,
        )

    def evaluate_image_json(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        selected_model = model or os.getenv("VLM_MODEL") or self.config.model
        path = Path(image_path)
        if not path.is_file():
            raise ModelClientError(f"Image does not exist: {path}")
        media_type = _image_media_type(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": selected_model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        started = time.monotonic()
        data = self._post_json(
            f"{_api_root(self.config.base_url)}/messages",
            payload,
            extra_headers={"anthropic-version": "2023-06-01"},
        )
        content = data.get("content") or []
        text = "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if not text.strip():
            raise ModelClientError("Vision response content was empty")
        return ModelResponse(
            value=_parse_json_text(text),
            model=str(data.get("model") or selected_model),
            request_id=_optional_str(data.get("id")),
            usage=dict(data.get("usage") or data.get("copilot_usage") or {}),
            latency_seconds=time.monotonic() - started,
        )

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        extra_headers: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        headers = self._headers
        if extra_headers:
            headers.update(extra_headers)
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=dict(payload),
                    timeout=(self.config.connect_timeout, self.config.timeout),
                )
                response.raise_for_status()
            except requests.HTTPError as exc:
                status = getattr(response, "status_code", None)
                detail = _response_detail(response)
                error = ModelClientError(
                    f"Model endpoint HTTP {status or 'error'}: {detail}"
                )
                if status is not None and 400 <= status < 500 and status != 429:
                    raise error from exc
                last_error = error
            except requests.RequestException as exc:
                last_error = exc
            else:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise ModelClientError("Model endpoint returned invalid JSON") from exc
                if not isinstance(data, dict):
                    raise ModelClientError("Model endpoint returned non-object JSON")
                if data.get("error"):
                    raise ModelClientError(f"Model endpoint error: {data['error']}")
                return data
            if attempt >= self.config.retries:
                break
            time.sleep(min(2**attempt, 5))
        raise ModelClientError(f"Model request failed: {last_error}") from last_error


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    try:
        return media_types[suffix]
    except KeyError as exc:
        raise ModelClientError(f"Unsupported image type: {suffix or '<none>'}") from exc


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        text = json.dumps(payload, ensure_ascii=False)
    except (ValueError, TypeError):
        text = str(getattr(response, "text", "") or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000] or "no response body"
