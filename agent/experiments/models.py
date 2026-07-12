from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


SCHEMA_VERSION = "1.0"
RECORD_FILENAME = "run_record.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_STATUSES = {"running", "completed", "failed"}
_BUDGET_TYPES = {"renders", "wall_clock_seconds"}


class ProvenanceError(ValueError):
    """Raised when a run cannot establish or verify its provenance."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProvenanceError(f"{name} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ProvenanceError(f"{name} must include a timezone")
    return parsed


def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_symlink():
        raise ProvenanceError(f"Symlink artifacts are not accepted: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise ProvenanceError(f"Artifact is neither a file nor directory: {path}")

    digest = hashlib.sha256()
    for child in sorted(resolved.rglob("*")):
        if child.is_symlink():
            raise ProvenanceError(f"Symlink artifacts are not accepted: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def verify_artifacts(record: "RunRecord", run_dir: Path) -> None:
    resolved_root = run_dir.resolve(strict=True)
    for name, relative_text in record.artifact_paths.items():
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProvenanceError(
                f"Artifact {name!r} has a non-portable path: {relative_text}"
            )
        candidate = resolved_root / relative
        if candidate.is_symlink():
            raise ProvenanceError(f"Artifact {name!r} is a symlink")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise ProvenanceError(
                f"Artifact {name!r} is missing or escaped the run directory"
            ) from exc
        actual = sha256_path(resolved)
        expected = record.artifact_hashes.get(name)
        if actual != expected:
            raise ProvenanceError(
                f"Artifact {name!r} failed its SHA-256 integrity check"
            )


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read JSON provenance at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProvenanceError(f"Expected a JSON object at {path}")
    return data


def _strict_dataclass_kwargs(cls: type[Any], data: Mapping[str, Any]) -> Dict[str, Any]:
    known = {item.name for item in fields(cls)}
    extra = set(data) - known
    if extra:
        raise ProvenanceError(
            f"Unknown {cls.__name__} fields: {', '.join(sorted(extra))}"
        )
    return dict(data)


def _validate_json_object(name: str, value: Mapping[str, Any]) -> None:
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"{name} must be JSON serializable: {exc}") from exc


@dataclass(frozen=True)
class ExperimentSpec:
    """One fully expanded, immutable experiment run."""

    run_name: str
    method: str
    schedule: str
    backbone: str
    seed: int
    budget_type: str
    budget_value: float
    dataset_manifest_path: str
    dataset_manifest_hash: str
    git_commit: str
    git_dirty: bool
    provider: str
    artifact_root: str
    repo_root: str
    metric_config: Dict[str, Any]
    metric_config_hash: str
    metric_version: str
    method_config: Dict[str, Any] = field(default_factory=dict)
    provider_options: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_name",
            "method",
            "schedule",
            "backbone",
            "dataset_manifest_path",
            "git_commit",
            "artifact_root",
            "repo_root",
            "metric_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ProvenanceError(f"ExperimentSpec.{name} must be non-empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ProvenanceError(
                f"Unsupported ExperimentSpec schema_version: {self.schema_version}"
            )
        if self.budget_type not in _BUDGET_TYPES:
            raise ProvenanceError(
                f"budget_type must be one of {sorted(_BUDGET_TYPES)}"
            )
        if not math.isfinite(float(self.budget_value)) or self.budget_value <= 0:
            raise ProvenanceError("budget_value must be a positive finite number")
        if self.budget_type == "renders" and not float(self.budget_value).is_integer():
            raise ProvenanceError("A renders budget must be an integer")
        if not _SHA256_RE.fullmatch(self.dataset_manifest_hash):
            raise ProvenanceError("dataset_manifest_hash must be a SHA-256 digest")
        if not _SHA256_RE.fullmatch(self.metric_config_hash):
            raise ProvenanceError("metric_config_hash must be a SHA-256 digest")
        if not _GIT_COMMIT_RE.fullmatch(self.git_commit):
            raise ProvenanceError("git_commit must be a hexadecimal commit id")
        _validate_json_object("metric_config", self.metric_config)
        _validate_json_object("method_config", self.method_config)
        _validate_json_object("provider_options", self.provider_options)
        if sha256_json(self.metric_config) != self.metric_config_hash:
            raise ProvenanceError("metric_config_hash does not match metric_config")

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        return cls(**_strict_dataclass_kwargs(cls, data))


@dataclass
class RunRecord:
    """Tamper-evident record for one attempted ExperimentSpec."""

    experiment_spec: Dict[str, Any]
    spec_hash: str
    run_name: str
    method: str
    backbone: str
    seed: int
    budget_type: str
    budget_value: float
    dataset_manifest_hash: str
    git_commit: str
    started_at: str
    finished_at: Optional[str]
    status: str
    render_count: int
    wall_clock_seconds: float
    artifact_paths: Dict[str, str]
    artifact_hashes: Dict[str, str]
    metric_config: Dict[str, Any]
    metric_config_hash: str
    metric_version: str
    provider: str
    provider_name: Optional[str]
    attempt: int
    test_only: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    candidates: list[Dict[str, Any]] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    best_history: list[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, str]] = None
    record_hash: str = ""
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def start(cls, spec: ExperimentSpec, attempt: int) -> "RunRecord":
        return cls(
            experiment_spec=spec.to_dict(),
            spec_hash=spec.spec_hash,
            run_name=spec.run_name,
            method=spec.method,
            backbone=spec.backbone,
            seed=spec.seed,
            budget_type=spec.budget_type,
            budget_value=spec.budget_value,
            dataset_manifest_hash=spec.dataset_manifest_hash,
            git_commit=spec.git_commit,
            started_at=utc_now(),
            finished_at=None,
            status="running",
            render_count=0,
            wall_clock_seconds=0.0,
            artifact_paths={},
            artifact_hashes={},
            metric_config=dict(spec.metric_config),
            metric_config_hash=spec.metric_config_hash,
            metric_version=spec.metric_version,
            provider=spec.provider,
            provider_name=None,
            attempt=attempt,
            test_only=False,
        )

    def to_dict(self, *, include_record_hash: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_record_hash:
            data.pop("record_hash", None)
        return data

    def seal(self) -> None:
        self.record_hash = sha256_json(self.to_dict(include_record_hash=False))

    def write(self, path: Path) -> None:
        self.seal()
        write_json_atomic(path, self.to_dict())

    def verify_record_hash(self) -> None:
        if not _SHA256_RE.fullmatch(self.record_hash):
            raise ProvenanceError("record_hash is missing or malformed")
        expected = sha256_json(self.to_dict(include_record_hash=False))
        if expected != self.record_hash:
            raise ProvenanceError("run_record.json failed its integrity check")

    def validate_provenance(self, *, require_completed: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProvenanceError(
                f"Unsupported RunRecord schema_version: {self.schema_version}"
            )
        if self.status not in _STATUSES:
            raise ProvenanceError(f"Invalid run status: {self.status}")
        if require_completed and self.status != "completed":
            raise ProvenanceError(f"Run is not completed: {self.run_name}")
        if not self.started_at:
            raise ProvenanceError("started_at is required")
        started = _parse_timestamp(self.started_at, "started_at")
        if self.status != "running" and not self.finished_at:
            raise ProvenanceError("finished_at is required for terminal runs")
        if self.finished_at is not None:
            finished = _parse_timestamp(self.finished_at, "finished_at")
            if finished < started:
                raise ProvenanceError("finished_at cannot precede started_at")
        if self.render_count < 0:
            raise ProvenanceError("render_count cannot be negative")
        if not math.isfinite(self.wall_clock_seconds) or self.wall_clock_seconds < 0:
            raise ProvenanceError("wall_clock_seconds must be finite and non-negative")
        if self.attempt < 1:
            raise ProvenanceError("attempt must be at least one")
        if set(self.artifact_paths) != set(self.artifact_hashes):
            raise ProvenanceError("Each artifact path must have exactly one hash")
        for digest in self.artifact_hashes.values():
            if not _SHA256_RE.fullmatch(digest):
                raise ProvenanceError("Artifact hashes must be SHA-256 digests")

        spec = ExperimentSpec.from_dict(self.experiment_spec)
        if spec.spec_hash != self.spec_hash:
            raise ProvenanceError("spec_hash does not match experiment_spec")
        mirrored = {
            "run_name": self.run_name,
            "method": self.method,
            "backbone": self.backbone,
            "seed": self.seed,
            "budget_type": self.budget_type,
            "budget_value": self.budget_value,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "git_commit": self.git_commit,
            "metric_config": self.metric_config,
            "metric_config_hash": self.metric_config_hash,
            "metric_version": self.metric_version,
            "provider": self.provider,
        }
        for name, actual in mirrored.items():
            if getattr(spec, name) != actual:
                raise ProvenanceError(f"RunRecord.{name} disagrees with ExperimentSpec")
        self.verify_record_hash()

        if self.status == "failed" and not self.error:
            raise ProvenanceError("Failed runs must include an explicit error")
        if self.status != "completed":
            return
        if self.render_count < 1:
            raise ProvenanceError("Completed runs must include at least one render")
        if not self.provider_name:
            raise ProvenanceError("Completed runs must identify the resolved provider")
        if not self.candidates:
            raise ProvenanceError("Completed runs must include candidate provenance")
        candidate_ids = {str(item.get("candidate_id", "")) for item in self.candidates}
        if not self.best_candidate_id or self.best_candidate_id not in candidate_ids:
            raise ProvenanceError("best_candidate_id must refer to a recorded candidate")
        if not self.best_history:
            raise ProvenanceError("Completed runs must include best-so-far history")
        if not self.metrics:
            raise ProvenanceError("Completed runs must include metrics")
        for name, value in self.metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ProvenanceError(f"Metric {name} is not a finite number")
        if not self.artifact_paths:
            raise ProvenanceError("Completed runs must include hashed artifacts")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        return cls(**_strict_dataclass_kwargs(cls, data))

    @classmethod
    def read(cls, path: Path) -> "RunRecord":
        return cls.from_dict(read_json(path))
