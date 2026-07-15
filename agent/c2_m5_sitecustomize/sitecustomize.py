"""Credential-free HTTPS and response-size guard for the frozen M5 downloader."""

from __future__ import annotations

import fcntl
import ipaddress
import os
import socket
import stat
import struct
from typing import Any
from urllib.parse import urljoin, urlparse
import urllib.request

import requests


if os.environ.get("C2_M5_NETWORK_GUARD_REQUIRED") != "1":
    raise RuntimeError("C2 M5 network guard loaded outside its fixed execution contract")


_ALLOWED_HOST_SUFFIXES = (
    "nature.com",
    "springernature.com",
    "springer.com",
)
_MAX_RESPONSE_BYTES = 256 * 1024 * 1024
_MAX_REDIRECTS = 5
_MAX_TOTAL_REQUESTS = 10_000
_MAX_TOTAL_RESPONSE_BYTES = 8 * 1024 * 1024 * 1024
_BUDGET_PATH = os.environ.get("C2_M5_NETWORK_BUDGET_PATH", "")
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_SESSION_REQUEST = requests.Session.request
_SESSION = requests.Session()
_SESSION.trust_env = False
_SESSION.auth = None
_SESSION.cookies.clear()


def _consume_budget(*, requests_used: int = 0, response_bytes: int = 0) -> None:
    if not _BUDGET_PATH or not os.path.isabs(_BUDGET_PATH):
        raise RuntimeError("network guard has no private cumulative budget")
    flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(_BUDGET_PATH, flags)
    except OSError as exc:
        raise RuntimeError("network guard cumulative budget is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077 != 0
            or metadata.st_size != 16
        ):
            raise RuntimeError("network guard cumulative budget is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = os.read(descriptor, 16)
        if len(payload) != 16:
            raise RuntimeError("network guard cumulative budget is truncated")
        request_count, byte_count = struct.unpack(">QQ", payload)
        request_count += requests_used
        byte_count += response_bytes
        exceeded = (
            request_count > _MAX_TOTAL_REQUESTS
            or byte_count > _MAX_TOTAL_RESPONSE_BYTES
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        written = os.write(
            descriptor,
            struct.pack(">QQ", request_count, byte_count),
        )
        if written != 16:
            raise RuntimeError("network guard cumulative budget write failed")
        os.fsync(descriptor)
        if exceeded:
            raise RuntimeError("network guard cumulative budget was exceeded")
    finally:
        os.close(descriptor)


def _allowed_host(host: str) -> bool:
    normalized = host.rstrip(".").casefold()
    return any(
        normalized == suffix or normalized.endswith(f".{suffix}")
        for suffix in _ALLOWED_HOST_SUFFIXES
    )


def _public_addresses(host: str, port: int) -> list[tuple[Any, ...]]:
    records = _ORIGINAL_GETADDRINFO(host, port, type=socket.SOCK_STREAM)
    if not records:
        raise RuntimeError("network guard DNS resolution returned no addresses")
    for record in records:
        address = ipaddress.ip_address(record[4][0])
        if (
            not address.is_global
            or address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise RuntimeError("network guard rejected a non-public destination")
    return records


def _guarded_getaddrinfo(
    host: str,
    port: int,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list[tuple[Any, ...]]:
    if (
        not isinstance(host, str)
        or not _allowed_host(host)
        or port not in {443, "443", "https"}
    ):
        raise RuntimeError("network guard rejected DNS outside the host allowlist")
    records = _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)
    for record in records:
        address = ipaddress.ip_address(record[4][0])
        if (
            not address.is_global
            or address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise RuntimeError("network guard rejected a non-public destination")
    return records


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or not _allowed_host(parsed.hostname)
        or parsed.port not in {None, 443}
    ):
        raise RuntimeError("network guard rejected a URL outside the HTTPS allowlist")
    _public_addresses(parsed.hostname, parsed.port or 443)


def _bounded_chunks(response: requests.Response, chunk_size: int = 1, **kwargs: Any):
    total = 0
    for chunk in requests.Response.iter_content(
        response,
        chunk_size=chunk_size,
        **kwargs,
    ):
        total += len(chunk)
        _consume_budget(response_bytes=len(chunk))
        if total > _MAX_RESPONSE_BYTES:
            response.close()
            raise RuntimeError("network guard response exceeded its byte cap")
        yield chunk


def _safe_get(url: str, **kwargs: Any) -> requests.Response:
    requested_stream = bool(kwargs.pop("stream", False))
    kwargs.pop("allow_redirects", None)
    if any(kwargs.get(key) is not None for key in ("auth", "cookies", "proxies", "cert")):
        raise RuntimeError("network guard rejected caller-supplied credentials or routing")
    if "verify" in kwargs and kwargs["verify"] is not True:
        raise RuntimeError("network guard requires TLS certificate verification")
    headers = kwargs.get("headers") or {}
    if any(
        str(name).casefold()
        in {"authorization", "cookie", "host", "proxy-authorization"}
        for name in headers
    ):
        raise RuntimeError("network guard rejected a credential-bearing header")
    current = url
    response: requests.Response | None = None
    for _ in range(_MAX_REDIRECTS + 1):
        _validate_url(current)
        _consume_budget(requests_used=1)
        _SESSION.cookies.clear()
        response = _ORIGINAL_SESSION_REQUEST(
            _SESSION,
            "GET",
            current,
            stream=True,
            allow_redirects=False,
            **kwargs,
        )
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > _MAX_RESPONSE_BYTES:
            response.close()
            raise RuntimeError("network guard Content-Length exceeded its byte cap")
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise RuntimeError("network guard received a redirect without Location")
            current = urljoin(current, location)
            continue
        response.iter_content = _bounded_chunks.__get__(response, requests.Response)
        if not requested_stream:
            payload = b"".join(response.iter_content(chunk_size=64 * 1024))
            response._content = payload
            response._content_consumed = True
        return response
    if response is not None:
        response.close()
    raise RuntimeError("network guard redirect limit exceeded")


def _safe_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    if not isinstance(method, str) or method.casefold() != "get":
        raise RuntimeError("network guard permits GET requests only")
    return _safe_get(url, **kwargs)


def _safe_session_request(
    _session: requests.Session,
    method: str,
    url: str,
    **kwargs: Any,
) -> requests.Response:
    return _safe_request(method, url, **kwargs)


def _blocked_urlopen(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("network guard rejected an unguarded urllib request")


requests.get = _safe_get
requests.api.get = _safe_get
requests.request = _safe_request
requests.api.request = _safe_request
requests.Session.request = _safe_session_request
urllib.request.urlopen = _blocked_urlopen
socket.getaddrinfo = _guarded_getaddrinfo
