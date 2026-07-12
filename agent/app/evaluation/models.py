from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Mapping, Optional


FIGURE_MANIFEST_SCHEMA_VERSION = "1.1.0"
EXPECTATION_SCHEMA_VERSION = "1.1.0"
METRIC_CONFIG_SCHEMA_VERSION = "1.0.0"
EVALUATION_RESULT_SCHEMA_VERSION = "1.1.0"


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        _to_plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class NumericTolerance:
    absolute: float = 1e-8
    relative: float = 1e-6

    def bound(self, expected: float) -> float:
        return self.absolute + self.relative * abs(expected)

    def to_dict(self) -> Dict[str, float]:
        return {"absolute": self.absolute, "relative": self.relative}


@dataclass(frozen=True)
class MetricConfig:
    schema_version: str = METRIC_CONFIG_SCHEMA_VERSION
    metric_version: str = "1.0.0"
    numeric_tolerance: NumericTolerance = field(default_factory=NumericTolerance)
    limit_tolerance: NumericTolerance = field(
        default_factory=lambda: NumericTolerance(absolute=1e-8, relative=1e-8)
    )
    float_precision: int = 12
    case_sensitive_labels: bool = True
    allow_missing_shared_units: bool = False
    unit_aliases: Mapping[str, str] = field(
        default_factory=lambda: {
            "µg": "ug",
            "μg": "ug",
            "microgram": "ug",
            "micrograms": "ug",
            "seconds": "s",
            "second": "s",
            "percent": "%",
        }
    )

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "MetricConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("MetricConfig.from_dict expects a mapping or None")
        from .schema import validate_metric_config_input

        validate_metric_config_input(value)
        numeric = value.get("numeric_tolerance") or {}
        limits = value.get("limit_tolerance") or {}
        unit_aliases = value.get("unit_aliases")
        return cls(
            schema_version=str(value.get("schema_version", METRIC_CONFIG_SCHEMA_VERSION)),
            metric_version=str(value.get("metric_version", "1.0.0")),
            numeric_tolerance=NumericTolerance(
                absolute=float(numeric.get("absolute", 1e-8)),
                relative=float(numeric.get("relative", 1e-6)),
            ),
            limit_tolerance=NumericTolerance(
                absolute=float(limits.get("absolute", 1e-8)),
                relative=float(limits.get("relative", 1e-8)),
            ),
            float_precision=int(value.get("float_precision", 12)),
            case_sensitive_labels=bool(value.get("case_sensitive_labels", True)),
            allow_missing_shared_units=bool(value.get("allow_missing_shared_units", False)),
            unit_aliases=dict(cls().unit_aliases if unit_aliases is None else unit_aliases),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_version": self.metric_version,
            "numeric_tolerance": self.numeric_tolerance.to_dict(),
            "limit_tolerance": self.limit_tolerance.to_dict(),
            "float_precision": self.float_precision,
            "case_sensitive_labels": self.case_sensitive_labels,
            "allow_missing_shared_units": self.allow_missing_shared_units,
            "unit_aliases": dict(sorted(self.unit_aliases.items())),
        }


@dataclass(frozen=True)
class SeriesManifest:
    series_id: str
    kind: str
    axis_slot: Optional[str]
    label: Optional[str]
    x: List[Any]
    y: List[Any]
    value: Any
    color: Optional[str]
    colors: List[str]
    style: Dict[str, Any]
    metadata: Dict[str, Any]
    x_labels: List[Optional[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SeriesManifest":
        return cls(**dict(value))


@dataclass(frozen=True)
class AxisProperties:
    label: str
    unit: Optional[str]
    scale: str
    limits: List[Optional[float]]
    tick_labels: List[str]
    offset_text: str

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AxisProperties":
        return cls(**dict(value))


@dataclass(frozen=True)
class LegendEntry:
    label: str
    color: Optional[str]
    linestyle: Optional[str]
    marker: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LegendEntry":
        return cls(**dict(value))


@dataclass(frozen=True)
class LegendManifest:
    present: bool
    location: Optional[Any]
    entries: List[LegendEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "present": self.present,
            "location": _to_plain(self.location),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LegendManifest":
        payload = dict(value)
        payload["entries"] = [
            LegendEntry.from_dict(entry) for entry in payload.get("entries") or []
        ]
        return cls(**payload)


@dataclass(frozen=True)
class AxisManifest:
    axis_id: str
    panel_id: Optional[str]
    parent_axis_id: Optional[str]
    axis_slot: Optional[str]
    index: int
    source_index: int
    role: str
    bbox: List[float]
    title: str
    x_axis: AxisProperties
    y_axis: AxisProperties
    legend: LegendManifest
    series: List[SeriesManifest]
    annotations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "panel_id": self.panel_id,
            "parent_axis_id": self.parent_axis_id,
            "axis_slot": self.axis_slot,
            "index": self.index,
            "source_index": self.source_index,
            "role": self.role,
            "bbox": list(self.bbox),
            "title": self.title,
            "x_axis": self.x_axis.to_dict(),
            "y_axis": self.y_axis.to_dict(),
            "legend": self.legend.to_dict(),
            "series": [series.to_dict() for series in self.series],
            "annotations": _to_plain(self.annotations),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AxisManifest":
        payload = dict(value)
        payload["x_axis"] = AxisProperties.from_dict(payload["x_axis"])
        payload["y_axis"] = AxisProperties.from_dict(payload["y_axis"])
        payload["legend"] = LegendManifest.from_dict(payload["legend"])
        payload["series"] = [
            SeriesManifest.from_dict(series) for series in payload.get("series") or []
        ]
        return cls(**payload)


@dataclass(frozen=True)
class FigureManifest:
    schema_version: str
    figure: Dict[str, Any]
    axes: List[AxisManifest]
    figure_legends: List[LegendManifest]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "figure": _to_plain(self.figure),
            "axes": [axis.to_dict() for axis in self.axes],
            "figure_legends": [legend.to_dict() for legend in self.figure_legends],
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FigureManifest":
        payload = dict(value)
        payload["axes"] = [
            AxisManifest.from_dict(axis) for axis in payload.get("axes") or []
        ]
        payload["figure_legends"] = [
            LegendManifest.from_dict(legend)
            for legend in payload.get("figure_legends") or []
        ]
        return cls(**payload)


@dataclass(frozen=True)
class Mismatch:
    code: str
    message: str
    panel_id: Optional[str] = None
    series_id: Optional[str] = None
    location: Dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    observed: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "panel_id": self.panel_id,
            "series_id": self.series_id,
            "location": _to_plain(self.location),
            "expected": _to_plain(self.expected),
            "observed": _to_plain(self.observed),
        }


@dataclass(frozen=True)
class CheckResult:
    name: str
    applicable: bool
    numerator: int
    denominator: int
    items: List[Dict[str, Any]] = field(default_factory=list)
    mismatches: List[Mismatch] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> Optional[float]:
        if not self.applicable or self.denominator == 0:
            return None
        return self.numerator / self.denominator

    @property
    def status(self) -> str:
        if not self.applicable or self.denominator == 0:
            return "na"
        return "pass" if self.numerator == self.denominator else "fail"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "applicable": self.applicable,
            "status": self.status,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "ratio": self.ratio,
            "items": _to_plain(self.items),
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
            "details": _to_plain(self.details),
        }


@dataclass(frozen=True)
class MetricResult:
    name: str
    checks: Dict[str, CheckResult]

    @property
    def numerator(self) -> int:
        return sum(
            check.numerator
            for check in self.checks.values()
            if check.applicable and check.denominator > 0
        )

    @property
    def denominator(self) -> int:
        return sum(
            check.denominator
            for check in self.checks.values()
            if check.applicable and check.denominator > 0
        )

    @property
    def applicable(self) -> bool:
        return self.denominator > 0

    @property
    def ratio(self) -> Optional[float]:
        if not self.applicable:
            return None
        return self.numerator / self.denominator

    @property
    def status(self) -> str:
        if not self.applicable:
            return "na"
        return "pass" if self.numerator == self.denominator else "fail"

    @property
    def mismatches(self) -> List[Mismatch]:
        return [
            mismatch
            for name in sorted(self.checks)
            for mismatch in self.checks[name].mismatches
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "applicable": self.applicable,
            "status": self.status,
            "aggregation": "sum_applicable_check_counts",
            "numerator": self.numerator,
            "denominator": self.denominator,
            "ratio": self.ratio,
            "checks": {
                name: self.checks[name].to_dict()
                for name in sorted(self.checks)
            },
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
        }


@dataclass(frozen=True)
class EvaluationResult:
    manifest: FigureManifest
    fidelity: MetricResult
    cohesion: MetricResult
    config: MetricConfig
    schema_version: str = EVALUATION_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric_config": self.config.to_dict(),
            "figure_manifest": self.manifest.to_dict(),
            "fidelity": self.fidelity.to_dict(),
            "cohesion": self.cohesion.to_dict(),
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_dict())
