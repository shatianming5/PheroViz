from __future__ import annotations

import http.client
import json
import os
import random
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from experiments.transformers_server import (
    SHUTDOWN_HEADER,
    ServerConfig,
    ServerLimits,
    TransformersTextEngine,
    create_server,
    is_loopback_host,
    load_transformers_engine,
    validate_bind_host,
    write_state_file,
)


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self, *, prompt_tokens: int = 4, decoded: str = "not-json") -> None:
        self.prompt_tokens = prompt_tokens
        self.decoded = decoded
        self.chat_messages: list[dict[str, str]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        self.chat_messages = [dict(message) for message in messages]
        return "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert prompt
        assert kwargs == {
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        return {
            "input_ids": [list(range(self.prompt_tokens))],
            "attention_mask": [[1] * self.prompt_tokens],
        }

    def decode(self, tokens: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        assert len(tokens) >= 1
        return self.decoded


class FakeModel:
    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(eos_token_id=0)
        self.config = SimpleNamespace(eos_token_id=0)
        self.calls: list[dict[str, Any]] = []
        self.device: str | None = None
        self.evaluated = False

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.calls.append(dict(kwargs))
        prompt = list(kwargs["input_ids"][0])
        count = min(int(kwargs["max_new_tokens"]), 2)
        return [prompt + [100 + index for index in range(count)]]

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.evaluated = True
        return self


class FakeCuda:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return False

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def ipc_collect(self) -> None:
        return None


class FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.cuda = FakeCuda()
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        )

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


def make_engine(
    *,
    limits: ServerLimits | None = None,
    prompt_tokens: int = 4,
    decoded: str = "not-json",
) -> tuple[TransformersTextEngine, FakeTokenizer, FakeModel, FakeTorch]:
    tokenizer = FakeTokenizer(prompt_tokens=prompt_tokens, decoded=decoded)
    model = FakeModel()
    torch = FakeTorch()
    engine = TransformersTextEngine(
        tokenizer=tokenizer,
        model=model,
        served_model_name="local-qwen",
        device="cpu",
        dtype_name="float32",
        limits=limits or ServerLimits(
            max_request_bytes=4096,
            max_context_tokens=32,
            max_output_tokens=8,
            default_max_tokens=4,
            request_timeout_seconds=5.0,
            generation_lock_timeout_seconds=1.0,
        ),
        torch_module=torch,
    )
    return engine, tokenizer, model, torch


@contextmanager
def running_server(
    *,
    limits: ServerLimits | None = None,
    decoded: str = "not-json",
    shutdown_token: str = "shutdown-secret",
) -> Iterator[tuple[Any, TransformersTextEngine, FakeTokenizer, FakeModel, FakeTorch]]:
    engine, tokenizer, model, torch = make_engine(
        limits=limits,
        decoded=decoded,
    )
    config = ServerConfig(
        host="127.0.0.1",
        port=0,
        served_model_name="local-qwen",
        api_key="api-secret",
        limits=engine.limits,
    )
    server = create_server(config, engine, shutdown_token=shutdown_token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, engine, tokenizer, model, torch
    finally:
        if thread.is_alive():
            server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        engine.close()


def request(
    server: Any,
    method: str,
    path: str,
    *,
    payload: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection(
        server.server_address[0],
        server.server_address[1],
        timeout=5,
    )
    body = None if payload is None else json.dumps(payload)
    request_headers = dict(headers or {})
    if body is not None:
        request_headers.setdefault("Content-Type", "application/json")
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, json.loads(raw)


def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer api-secret"}


def test_health_models_and_auth() -> None:
    with running_server() as (server, *_):
        status, health = request(server, "GET", "/health")
        assert status == 200
        assert health["text_only"] is True

        status, error = request(server, "GET", "/v1/models")
        assert status == 401
        assert error["error"]["code"] == "invalid_api_key"

        status, error = request(
            server,
            "POST",
            "/v1/chat/completions",
            payload={
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert status == 401
        assert error["error"]["code"] == "invalid_api_key"

        status, models = request(
            server,
            "GET",
            "/v1/models",
            headers={"x-api-key": "api-secret"},
        )
        assert status == 200
        assert models["data"][0]["id"] == "local-qwen"


def test_chat_sampling_seed_and_json_prompt_without_fabrication() -> None:
    with running_server(decoded="plain-not-json") as (
        server,
        _engine,
        tokenizer,
        model,
        torch,
    ):
        random.seed(999)
        status, response = request(
            server,
            "POST",
            "/v1/chat/completions",
            headers=auth_headers(),
            payload={
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "produce data"}],
                "max_tokens": 3,
                "temperature": 0.7,
                "seed": 17,
                "reasoning_effort": "high",
                "response_format": {"type": "json_object"},
            },
        )

        assert status == 200
        assert response["model"] == "local-qwen"
        assert response["choices"][0]["message"]["content"] == "plain-not-json"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["usage"] == {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        }
        assert response["ignored_parameters"] == ["reasoning_effort"]
        assert tokenizer.chat_messages[0]["role"] == "system"
        assert "valid JSON object" in tokenizer.chat_messages[0]["content"]
        call = model.calls[0]
        assert call["max_new_tokens"] == 3
        assert call["do_sample"] is True
        assert call["temperature"] == 0.7
        assert torch.seeds == [17]
        assert torch.cuda.seeds == [17]


def test_greedy_sampling_omits_temperature() -> None:
    with running_server() as (server, _engine, _tokenizer, model, _torch):
        status, _ = request(
            server,
            "POST",
            "/v1/chat/completions",
            headers=auth_headers(),
            payload={
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
            },
        )
        assert status == 200
        assert model.calls[0]["do_sample"] is False
        assert "temperature" not in model.calls[0]


def test_request_context_output_and_text_only_limits() -> None:
    limits = ServerLimits(
        max_request_bytes=300,
        max_context_tokens=3,
        max_output_tokens=2,
        default_max_tokens=1,
        request_timeout_seconds=5.0,
        generation_lock_timeout_seconds=1.0,
    )
    with running_server(limits=limits) as (server, *_):
        status, error = request(
            server,
            "POST",
            "/v1/chat/completions",
            headers=auth_headers(),
            payload={
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert status == 413
        assert error["error"]["code"] == "context_length_exceeded"

        status, error = request(
            server,
            "POST",
            "/v1/chat/completions",
            headers=auth_headers(),
            payload={
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
            },
        )
        assert status == 400
        assert error["error"]["code"] == "max_tokens_limit"

        status, error = request(
            server,
            "POST",
            "/v1/chat/completions",
            headers=auth_headers(),
            payload={
                "model": "local-qwen",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "url": "file:///etc/passwd"}],
                    }
                ],
            },
        )
        assert status == 400
        assert error["error"]["code"] == "text_messages_only"

        oversized = {"messages": [{"role": "user", "content": "x" * 500}]}
        status, error = request(
            server,
            "POST",
            "/v1/chat/completions",
            headers=auth_headers(),
            payload=oversized,
        )
        assert status == 413
        assert error["error"]["code"] == "request_too_large"


def test_shutdown_requires_random_token() -> None:
    with running_server(shutdown_token="shutdown-secret") as (server, *_):
        status, error = request(server, "POST", "/shutdown")
        assert status == 403
        assert error["error"]["code"] == "invalid_shutdown_token"

        status, response = request(
            server,
            "POST",
            "/shutdown",
            headers={SHUTDOWN_HEADER: "shutdown-secret"},
        )
        assert status == 200
        assert response == {"status": "shutting_down"}


def test_remote_bind_gate() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    validate_bind_host("127.0.0.1", allow_remote=False)
    with pytest.raises(ValueError, match="allow-remote"):
        validate_bind_host("0.0.0.0", allow_remote=False)
    validate_bind_host("0.0.0.0", allow_remote=True)


def test_state_file_is_private_and_does_not_require_api_key(tmp_path: Path) -> None:
    state_path = write_state_file(
        tmp_path / "server-state.json",
        {
            "pid": 123,
            "port": 8000,
            "shutdown_token": "shutdown-only",
        },
    )
    assert oct(state_path.stat().st_mode & 0o777) == "0o600"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["shutdown_token"] == "shutdown-only"
    assert "api_key" not in state


class FakeLoader:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, path: str, **kwargs: Any) -> Any:
        self.calls.append((path, dict(kwargs)))
        return self.value


class FakeTransformers:
    def __init__(self, architecture: str) -> None:
        self.AutoConfig = FakeLoader(
            SimpleNamespace(
                architectures=[architecture],
                model_type="qwen3_5" if architecture.startswith("Qwen") else "llama",
            )
        )
        self.AutoTokenizer = FakeLoader(FakeTokenizer())
        self.AutoModelForImageTextToText = FakeLoader(FakeModel())
        self.AutoModelForCausalLM = FakeLoader(FakeModel())


@pytest.mark.parametrize(
    ("architecture", "expected_loader"),
    [
        ("Qwen3_5ForConditionalGeneration", "image"),
        ("LlamaForCausalLM", "causal"),
    ],
)
def test_loader_selects_qwen35_or_causal_model(
    architecture: str,
    expected_loader: str,
    tmp_path: Path,
) -> None:
    transformers = FakeTransformers(architecture)
    torch = FakeTorch()
    engine = load_transformers_engine(
        model_path=tmp_path,
        served_model_name="served",
        device="cpu",
        dtype="float32",
        limits=ServerLimits(),
        transformers_module=transformers,
        torch_module=torch,
    )
    try:
        if expected_loader == "image":
            assert transformers.AutoModelForImageTextToText.calls
            assert not transformers.AutoModelForCausalLM.calls
        else:
            assert transformers.AutoModelForCausalLM.calls
            assert not transformers.AutoModelForImageTextToText.calls
        loader = (
            transformers.AutoModelForImageTextToText
            if expected_loader == "image"
            else transformers.AutoModelForCausalLM
        )
        assert loader.calls[0][1]["local_files_only"] is True
        assert loader.calls[0][1]["trust_remote_code"] is False
        assert engine.model.device == "cpu"
        assert engine.model.evaluated
    finally:
        engine.close()
