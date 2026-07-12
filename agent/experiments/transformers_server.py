from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hmac
import http.server
import ipaddress
import json
import math
import os
import random
import secrets
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


SHUTDOWN_HEADER = "X-PheroViz-Shutdown-Token"
_ALLOWED_ROLES = {"system", "user", "assistant"}
_ALLOWED_REQUEST_KEYS = {
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "seed",
    "reasoning_effort",
    "response_format",
    "stream",
}
_JSON_PROMPT = (
    "Return exactly one valid JSON object and no Markdown or surrounding text. "
    "Do not claim that the server validates or repairs the JSON."
)


class APIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str,
        code: str,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class ServerLimits:
    max_request_bytes: int = 1_048_576
    max_context_tokens: int = 16_384
    max_output_tokens: int = 2_048
    default_max_tokens: int = 512
    request_timeout_seconds: float = 300.0
    generation_lock_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "max_request_bytes",
            "max_context_tokens",
            "max_output_tokens",
            "default_max_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.default_max_tokens > self.max_output_tokens:
            raise ValueError("default_max_tokens cannot exceed max_output_tokens")
        for name in (
            "request_timeout_seconds",
            "generation_lock_timeout_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    served_model_name: str
    api_key: str = field(repr=False, compare=False)
    limits: ServerLimits = ServerLimits()

    def __post_init__(self) -> None:
        if not self.served_model_name.strip():
            raise ValueError("served_model_name must be non-empty")
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if not 0 <= int(self.port) <= 65535:
            raise ValueError("port must be in 0..65535")


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_bind_host(host: str, *, allow_remote: bool) -> None:
    if not is_loopback_host(host) and not allow_remote:
        raise ValueError(
            f"Refusing non-loopback bind {host!r}; pass --allow-remote explicitly"
        )


def _safe_json_dumps(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_state_file(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _safe_json_dumps(dict(payload))
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, output)
        os.chmod(output, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _sequence_length(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape):
        return int(shape[-1])
    if isinstance(value, Sequence) and value:
        first = value[0]
        if isinstance(first, Sequence):
            return len(first)
        return len(value)
    raise APIError(
        500,
        "Tokenizer returned an unsupported input representation",
        error_type="server_error",
        code="tokenizer_contract_error",
    )


def _first_sequence(value: Any) -> Any:
    try:
        return value[0]
    except (IndexError, KeyError, TypeError) as exc:
        raise APIError(
            500,
            "Model returned no generated sequence",
            error_type="server_error",
            code="model_contract_error",
        ) from exc


def _token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape):
        return int(shape[-1]) if len(shape) > 1 else int(shape[0])
    return len(value)


def _move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _eos_ids(tokenizer: Any, model: Any) -> set[int]:
    values: set[int] = set()
    for candidate in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    ):
        if isinstance(candidate, int):
            values.add(candidate)
        elif isinstance(candidate, Sequence) and not isinstance(candidate, str):
            values.update(int(item) for item in candidate if isinstance(item, int))
    return values


class TransformersTextEngine:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        served_model_name: str,
        device: str,
        dtype_name: str,
        limits: ServerLimits,
        torch_module: Any | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.served_model_name = served_model_name
        self.device = device
        self.dtype_name = dtype_name
        self.limits = limits
        self.torch = torch_module
        self.generation_lock = threading.Lock()
        self.closed = False

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if not self.closed else "stopped",
            "model": self.served_model_name,
            "device": self.device,
            "dtype": self.dtype_name,
            "text_only": True,
        }

    def models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-transformers",
                }
            ],
        }

    def _validate_payload(
        self, payload: Mapping[str, Any]
    ) -> tuple[list[dict[str, str]], int, float | None, int | None, bool]:
        unknown = sorted(set(payload) - _ALLOWED_REQUEST_KEYS)
        if unknown:
            raise APIError(
                400,
                f"Unsupported request fields: {', '.join(unknown)}",
                error_type="invalid_request_error",
                code="unsupported_fields",
            )
        requested_model = payload.get("model")
        if requested_model not in (None, self.served_model_name):
            raise APIError(
                404,
                "Requested model is not served by this process",
                error_type="invalid_request_error",
                code="model_not_found",
                param="model",
            )
        if payload.get("stream") not in (None, False):
            raise APIError(
                400,
                "Streaming is not supported",
                error_type="invalid_request_error",
                code="stream_not_supported",
                param="stream",
            )
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise APIError(
                400,
                "messages must be a non-empty array",
                error_type="invalid_request_error",
                code="invalid_messages",
                param="messages",
            )
        messages: list[dict[str, str]] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise APIError(
                    400,
                    f"messages[{index}] must be an object",
                    error_type="invalid_request_error",
                    code="invalid_message",
                    param="messages",
                )
            extra = set(raw_message) - {"role", "content"}
            role = raw_message.get("role")
            content = raw_message.get("content")
            if extra or role not in _ALLOWED_ROLES or not isinstance(content, str):
                raise APIError(
                    400,
                    f"messages[{index}] must contain only a supported role and text content",
                    error_type="invalid_request_error",
                    code="text_messages_only",
                    param="messages",
                )
            messages.append({"role": str(role), "content": content})

        raw_max_tokens = payload.get("max_tokens", self.limits.default_max_tokens)
        if (
            isinstance(raw_max_tokens, bool)
            or not isinstance(raw_max_tokens, int)
            or not 1 <= raw_max_tokens <= self.limits.max_output_tokens
        ):
            raise APIError(
                400,
                f"max_tokens must be in 1..{self.limits.max_output_tokens}",
                error_type="invalid_request_error",
                code="max_tokens_limit",
                param="max_tokens",
            )

        raw_temperature = payload.get("temperature")
        temperature: float | None
        if raw_temperature is None:
            temperature = None
        elif isinstance(raw_temperature, bool) or not isinstance(
            raw_temperature, (int, float)
        ):
            raise APIError(
                400,
                "temperature must be numeric",
                error_type="invalid_request_error",
                code="invalid_temperature",
                param="temperature",
            )
        else:
            temperature = float(raw_temperature)
            if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
                raise APIError(
                    400,
                    "temperature must be finite and in [0, 2]",
                    error_type="invalid_request_error",
                    code="invalid_temperature",
                    param="temperature",
                )

        raw_seed = payload.get("seed")
        if raw_seed is not None and (
            isinstance(raw_seed, bool) or not isinstance(raw_seed, int)
        ):
            raise APIError(
                400,
                "seed must be an integer",
                error_type="invalid_request_error",
                code="invalid_seed",
                param="seed",
            )

        response_format = payload.get("response_format")
        json_mode = False
        if response_format is not None:
            if not isinstance(response_format, Mapping) or set(response_format) != {
                "type"
            }:
                raise APIError(
                    400,
                    "response_format must contain only type",
                    error_type="invalid_request_error",
                    code="invalid_response_format",
                    param="response_format",
                )
            format_type = response_format.get("type")
            if format_type == "json_object":
                json_mode = True
            elif format_type != "text":
                raise APIError(
                    400,
                    "response_format.type must be text or json_object",
                    error_type="invalid_request_error",
                    code="invalid_response_format",
                    param="response_format",
                )
        reasoning_effort = payload.get("reasoning_effort")
        if reasoning_effort is not None and not isinstance(reasoning_effort, str):
            raise APIError(
                400,
                "reasoning_effort must be a string when provided",
                error_type="invalid_request_error",
                code="invalid_reasoning_effort",
                param="reasoning_effort",
            )
        return messages, raw_max_tokens, temperature, raw_seed, json_mode

    def chat_completion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise APIError(
                503,
                "Model runtime is shutting down",
                error_type="server_error",
                code="runtime_stopped",
            )
        messages, max_tokens, temperature, seed, json_mode = self._validate_payload(
            payload
        )
        prompt_messages = list(messages)
        if json_mode:
            if prompt_messages and prompt_messages[0]["role"] == "system":
                prompt_messages[0] = {
                    "role": "system",
                    "content": (
                        f"{prompt_messages[0]['content'].rstrip()}\n\n{_JSON_PROMPT}"
                    ),
                }
            else:
                prompt_messages.insert(
                    0,
                    {"role": "system", "content": _JSON_PROMPT},
                )

        try:
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
        except APIError:
            raise
        except Exception as exc:
            raise APIError(
                400,
                "Tokenizer rejected the text messages",
                error_type="invalid_request_error",
                code="tokenization_failed",
                param="messages",
            ) from exc
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise APIError(
                500,
                "Tokenizer did not return input_ids",
                error_type="server_error",
                code="tokenizer_contract_error",
            )
        prompt_tokens = _sequence_length(encoded["input_ids"])
        if prompt_tokens > self.limits.max_context_tokens:
            raise APIError(
                413,
                f"Context has {prompt_tokens} tokens; limit is {self.limits.max_context_tokens}",
                error_type="invalid_request_error",
                code="context_length_exceeded",
                param="messages",
            )
        batch = _move_batch(encoded, self.device)

        acquired = self.generation_lock.acquire(
            timeout=self.limits.generation_lock_timeout_seconds
        )
        if not acquired:
            raise APIError(
                503,
                "Generation queue is busy",
                error_type="server_error",
                code="generation_busy",
            )
        try:
            if seed is not None:
                random.seed(seed)
                if self.torch is not None:
                    self.torch.manual_seed(seed)
                    cuda = getattr(self.torch, "cuda", None)
                    if cuda is not None and callable(
                        getattr(cuda, "manual_seed_all", None)
                    ):
                        cuda.manual_seed_all(seed)
            generate_kwargs: dict[str, Any] = {
                **batch,
                "max_new_tokens": max_tokens,
                "do_sample": bool(temperature and temperature > 0),
            }
            if temperature is not None and temperature > 0:
                generate_kwargs["temperature"] = temperature
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            try:
                generated = self.model.generate(**generate_kwargs)
            except Exception as exc:
                raise APIError(
                    500,
                    "Local model generation failed",
                    error_type="server_error",
                    code="generation_failed",
                ) from exc
        finally:
            self.generation_lock.release()

        sequence = _first_sequence(generated)
        completion_ids = sequence[prompt_tokens:]
        completion_tokens = _token_count(completion_ids)
        try:
            text = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )
        except Exception as exc:
            raise APIError(
                500,
                "Tokenizer failed to decode model output",
                error_type="server_error",
                code="decode_failed",
            ) from exc
        eos = _eos_ids(self.tokenizer, self.model)
        last_token: int | None = None
        if completion_tokens:
            try:
                last_token = int(completion_ids[-1])
            except (TypeError, ValueError):
                last_token = None
        finish_reason = (
            "stop"
            if last_token is not None and last_token in eos
            else "length"
            if completion_tokens >= max_tokens
            else "stop"
        )
        response: dict[str, Any] = {
            "id": f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": str(text)},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if payload.get("reasoning_effort") is not None:
            response["ignored_parameters"] = ["reasoning_effort"]
        return response

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        model = self.model
        tokenizer = self.tokenizer
        self.model = None
        self.tokenizer = None
        del model
        del tokenizer
        gc.collect()
        if self.torch is not None:
            cuda = getattr(self.torch, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)):
                if cuda.is_available():
                    if callable(getattr(cuda, "empty_cache", None)):
                        cuda.empty_cache()
                    if callable(getattr(cuda, "ipc_collect", None)):
                        cuda.ipc_collect()


def _resolve_device(torch_module: Any, requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            return "cuda"
        backends = getattr(torch_module, "backends", None)
        mps = getattr(backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if normalized in {"cpu", "mps", "cuda"}:
        return normalized
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        return normalized
    raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")


def _resolve_dtype(torch_module: Any, requested: str) -> tuple[Any | None, str]:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return None, "auto"
    names = {
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
    }
    if normalized not in names:
        raise ValueError("dtype must be auto, float16, bfloat16, or float32")
    return getattr(torch_module, names[normalized]), normalized


def load_transformers_engine(
    *,
    model_path: str | Path,
    served_model_name: str,
    device: str,
    dtype: str,
    limits: ServerLimits,
    transformers_module: Any | None = None,
    torch_module: Any | None = None,
) -> TransformersTextEngine:
    local_path = Path(model_path).expanduser().resolve(strict=True)
    if not local_path.is_dir():
        raise ValueError("model_path must be a local Transformers model directory")
    if transformers_module is None:
        try:
            import transformers as transformers_module
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required in the model server environment"
            ) from exc
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("torch is required in the model server environment") from exc

    resolved_device = _resolve_device(torch_module, device)
    torch_dtype, dtype_name = _resolve_dtype(torch_module, dtype)
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    config = transformers_module.AutoConfig.from_pretrained(
        str(local_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    architectures = {
        str(value).replace(".", "_").lower()
        for value in (getattr(config, "architectures", None) or [])
    }
    model_type = str(getattr(config, "model_type", "")).replace(".", "_").lower()
    is_qwen35 = (
        model_type in {"qwen3_5", "qwen35"}
        or any(
            "qwen3_5forconditionalgeneration" in architecture
            or "qwen35forconditionalgeneration" in architecture
            for architecture in architectures
        )
    )
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        str(local_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if is_qwen35:
        model_loader = getattr(
            transformers_module,
            "AutoModelForImageTextToText",
            None,
        )
        if model_loader is None:
            raise RuntimeError(
                "Installed transformers lacks AutoModelForImageTextToText required by Qwen3.5"
            )
    else:
        model_loader = transformers_module.AutoModelForCausalLM
    model = model_loader.from_pretrained(str(local_path), **load_kwargs)
    model.to(resolved_device)
    model.eval()
    return TransformersTextEngine(
        tokenizer=tokenizer,
        model=model,
        served_model_name=served_model_name,
        device=resolved_device,
        dtype_name=dtype_name,
        limits=limits,
        torch_module=torch_module,
    )


class TransformersHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        config: ServerConfig,
        engine: TransformersTextEngine,
        shutdown_token: str,
    ) -> None:
        self.config = config
        self.engine = engine
        self.shutdown_token = shutdown_token
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="transformers-generation",
        )
        super().__init__(server_address, TransformersRequestHandler)

    def server_close(self) -> None:
        super().server_close()
        self.executor.shutdown(wait=True, cancel_futures=True)


class TransformersRequestHandler(http.server.BaseHTTPRequestHandler):
    server: TransformersHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "PheroVizTransformers/1.0"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        message = format % args
        print(
            f"[transformers-server] client={self.client_address[0]} {message}",
            flush=True,
        )

    def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = _safe_json_dumps(payload)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return

    def _error(self, error: APIError) -> None:
        self._write_json(error.status, error.payload())

    def _authorized(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        bearer = ""
        if authorization.lower().startswith("bearer "):
            bearer = authorization[7:]
        candidate = bearer or self.headers.get("x-api-key", "")
        return bool(candidate) and hmac.compare_digest(
            candidate,
            self.server.config.api_key,
        )

    def _require_auth(self) -> None:
        if not self._authorized():
            self.close_connection = True
            raise APIError(
                401,
                "Invalid or missing API key",
                error_type="authentication_error",
                code="invalid_api_key",
            )

    def _read_json_body(self) -> dict[str, Any]:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise APIError(
                415,
                "Content-Type must be application/json",
                error_type="invalid_request_error",
                code="invalid_content_type",
            )
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise APIError(
                411,
                "Content-Length is required",
                error_type="invalid_request_error",
                code="length_required",
            )
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise APIError(
                400,
                "Invalid Content-Length",
                error_type="invalid_request_error",
                code="invalid_content_length",
            ) from exc
        if length < 0 or length > self.server.config.limits.max_request_bytes:
            raise APIError(
                413,
                "Request body exceeds the configured limit",
                error_type="invalid_request_error",
                code="request_too_large",
            )
        self.connection.settimeout(self.server.config.limits.request_timeout_seconds)
        try:
            body = self.rfile.read(length)
        except socket.timeout as exc:
            raise APIError(
                408,
                "Timed out reading request body",
                error_type="request_timeout",
                code="request_timeout",
            ) from exc
        if len(body) != length:
            raise APIError(
                400,
                "Request body ended before Content-Length bytes were received",
                error_type="invalid_request_error",
                code="incomplete_body",
            )
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIError(
                400,
                "Request body must be valid UTF-8 JSON",
                error_type="invalid_request_error",
                code="invalid_json",
            ) from exc
        if not isinstance(value, dict):
            raise APIError(
                400,
                "Request JSON must be an object",
                error_type="invalid_request_error",
                code="invalid_json_type",
            )
        return value

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._write_json(200, self.server.engine.health())
                return
            if self.path == "/v1/models":
                self._require_auth()
                self._write_json(200, self.server.engine.models())
                return
            raise APIError(
                404,
                "Endpoint not found",
                error_type="invalid_request_error",
                code="not_found",
            )
        except APIError as error:
            self._error(error)

    def do_POST(self) -> None:
        try:
            if self.path == "/shutdown":
                token = self.headers.get(SHUTDOWN_HEADER, "")
                if not token or not hmac.compare_digest(
                    token,
                    self.server.shutdown_token,
                ):
                    raise APIError(
                        403,
                        "Invalid shutdown token",
                        error_type="authentication_error",
                        code="invalid_shutdown_token",
                    )
                self._write_json(200, {"status": "shutting_down"})
                threading.Thread(
                    target=self.server.shutdown,
                    name="transformers-shutdown",
                    daemon=True,
                ).start()
                return
            if self.path != "/v1/chat/completions":
                raise APIError(
                    404,
                    "Endpoint not found",
                    error_type="invalid_request_error",
                    code="not_found",
                )
            self._require_auth()
            payload = self._read_json_body()
            future = self.server.executor.submit(
                self.server.engine.chat_completion,
                payload,
            )
            try:
                response = future.result(
                    timeout=self.server.config.limits.request_timeout_seconds
                )
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise APIError(
                    504,
                    "Local generation exceeded the request timeout",
                    error_type="request_timeout",
                    code="generation_timeout",
                ) from exc
            self._write_json(200, response)
        except APIError as error:
            self.close_connection = True
            self._error(error)
        except Exception:
            self.close_connection = True
            self._error(
                APIError(
                    500,
                    "Unexpected server error",
                    error_type="server_error",
                    code="internal_error",
                )
            )


def create_server(
    config: ServerConfig,
    engine: TransformersTextEngine,
    *,
    shutdown_token: str | None = None,
) -> TransformersHTTPServer:
    return TransformersHTTPServer(
        (config.host, config.port),
        config=config,
        engine=engine,
        shutdown_token=shutdown_token or secrets.token_urlsafe(32),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a local text-only Transformers model with an OpenAI-compatible API."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--api-key-env", default="OPEN_MODEL_API_KEY")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--max-request-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-context-tokens", type=int, default=16_384)
    parser.add_argument("--max-output-tokens", type=int, default=2_048)
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--generation-lock-timeout", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_bind_host(args.host, allow_remote=args.allow_remote)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"API key environment variable {args.api_key_env!r} is missing or empty"
        )
    limits = ServerLimits(
        max_request_bytes=args.max_request_bytes,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        default_max_tokens=args.default_max_tokens,
        request_timeout_seconds=args.request_timeout,
        generation_lock_timeout_seconds=args.generation_lock_timeout,
    )
    engine = load_transformers_engine(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        device=args.device,
        dtype=args.dtype,
        limits=limits,
    )
    config = ServerConfig(
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        api_key=api_key,
        limits=limits,
    )
    shutdown_token = secrets.token_urlsafe(32)
    server = create_server(config, engine, shutdown_token=shutdown_token)
    bound_host, bound_port = server.server_address[:2]
    state_path = Path(args.state_file)
    started_at = int(time.time())
    state = {
        "status": "ready",
        "pid": os.getpid(),
        "host": str(bound_host),
        "port": int(bound_port),
        "served_model_name": args.served_model_name,
        "device": engine.device,
        "dtype": engine.dtype_name,
        "text_only": True,
        "shutdown_header": SHUTDOWN_HEADER,
        "shutdown_token": shutdown_token,
        "started_at": started_at,
    }
    write_state_file(state_path, state)
    print(
        f"[transformers-server] ready pid={os.getpid()} host={bound_host} "
        f"port={bound_port} state={state_path}",
        flush=True,
    )

    def request_shutdown(signum: int, frame: Any) -> None:
        del signum, frame
        threading.Thread(
            target=server.shutdown,
            name="transformers-signal-shutdown",
            daemon=True,
        ).start()

    previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        engine.close()
        state["status"] = "stopped"
        state["stopped_at"] = int(time.time())
        state["gpu_release_requested"] = True
        write_state_file(state_path, state)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
