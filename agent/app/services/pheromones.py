from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MEMORY_SCHEMA_VERSION = 1
_DEFAULT_CAPACITY = {1: 32, 2: 24, 3: 16, 4: 12}
_DEFAULT_TTL_STEPS = {1: 24, 2: 18, 3: 12, 4: 6}
_MISSING = object()


class Scope(str, Enum):
    FIGURE = "figure"
    PANEL_GROUP = "panel_group"
    PANEL = "panel"
    AXIS = "axis"
    ELEMENT = "element"

    @property
    def rank(self) -> int:
        return {
            Scope.FIGURE: 5,
            Scope.PANEL_GROUP: 4,
            Scope.PANEL: 3,
            Scope.AXIS: 2,
            Scope.ELEMENT: 1,
        }[self]


class EvidenceType(str, Enum):
    """Compatibility enum for callers of the original module."""

    constraint = "constraint"
    style = "style"
    geom = "geom"
    layout = "layout"
    ref = "ref"


def _as_scope(value: Scope | str) -> Scope:
    return value if isinstance(value, Scope) else Scope(str(value))


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Memory values must be finite JSON data: {value!r}") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(payload)).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


def _normalized_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return _json_copy(dict(value or {}))


def _normalized_strings(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in (values or ()) if str(value)}))


def _value_equal(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _slot_matches(slot: str, patterns: Iterable[str]) -> bool:
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        if pattern.endswith("*"):
            if slot.startswith(pattern[:-1]):
                return True
        elif slot == pattern:
            return True
    return False


def chart_meta_class(chart_family: str) -> str:
    family = (chart_family or "").strip().lower()
    if family in {"heatmap", "matrix", "hexbin"}:
        return "matrix"
    if family in {"pie", "donut", "stacked_bar", "treemap", "sunburst"}:
        return "compositional"
    if family in {"overlay", "mixed", "dual_axis"}:
        return "overlay"
    return "positional"


@dataclass
class ConstraintRecord:
    scope: Scope | str
    level: int
    slot: str
    value: Any
    hard: bool
    provenance: dict[str, Any]
    support: int = 0
    conflicts: int = 0
    validated_executable: bool = False
    timestamp: float = field(default_factory=time.time)
    id: str = ""
    scope_key: str | None = None
    chart_meta_class: str | None = None
    created_step: int | None = None
    ttl_steps: int | None = None

    def __post_init__(self) -> None:
        self.scope = _as_scope(self.scope)
        if self.level not in {1, 2, 3, 4}:
            raise ValueError(f"level must be in 1..4, got {self.level!r}")
        self.slot = str(self.slot).strip()
        if not self.slot:
            raise ValueError("constraint slot must be non-empty")
        self.value = _json_copy(self.value)
        self.provenance = _normalized_mapping(self.provenance)
        self.scope_key = None if self.scope_key is None else str(self.scope_key)
        self.chart_meta_class = (
            None if self.chart_meta_class is None else str(self.chart_meta_class)
        )
        self.support = max(0, int(self.support))
        self.conflicts = max(0, int(self.conflicts))
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        if self.created_step is not None:
            self.created_step = int(self.created_step)
        if self.ttl_steps is not None:
            self.ttl_steps = max(0, int(self.ttl_steps))
        if not self.id:
            self.id = _stable_id(
                "constraint",
                {
                    "scope": self.scope.value,
                    "scope_key": self.scope_key,
                    "level": self.level,
                    "slot": self.slot,
                    "value": self.value,
                    "hard": self.hard,
                    "provenance": self.provenance,
                    "chart_meta_class": self.chart_meta_class,
                },
            )

    @property
    def grounding_rank(self) -> int:
        grounding = str(self.provenance.get("grounding") or "").lower()
        if self.provenance.get("source_checked") or grounding == "source":
            return 2
        if self.validated_executable or grounding == "executable":
            return 1
        return 0

    @property
    def precedence_key(self) -> tuple[int, int, int, int, int, int, int, str]:
        """Total order; timestamp and insertion order intentionally do not participate."""

        return (
            int(self.hard),
            self.scope.rank,
            5 - self.level,
            int(self.validated_executable),
            self.grounding_rank,
            self.support,
            -self.conflicts,
            self.id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope.value,
            "scope_key": self.scope_key,
            "level": self.level,
            "slot": self.slot,
            "value": _json_copy(self.value),
            "hard": bool(self.hard),
            "provenance": _json_copy(self.provenance),
            "support": self.support,
            "conflicts": self.conflicts,
            "validated_executable": bool(self.validated_executable),
            "timestamp": float(self.timestamp),
            "chart_meta_class": self.chart_meta_class,
            "created_step": self.created_step,
            "ttl_steps": self.ttl_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConstraintRecord":
        return cls(**dict(value))


@dataclass
class SafetyPredicate:
    allowed_slots: tuple[str, ...] = ()
    forbidden_slots: tuple[str, ...] = ()
    required_constraints: dict[str, Any] = field(default_factory=dict)
    require_validated_executable: bool = True

    def __post_init__(self) -> None:
        self.allowed_slots = _normalized_strings(self.allowed_slots)
        self.forbidden_slots = _normalized_strings(self.forbidden_slots)
        self.required_constraints = _normalized_mapping(self.required_constraints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_slots": list(self.allowed_slots),
            "forbidden_slots": list(self.forbidden_slots),
            "required_constraints": _json_copy(self.required_constraints),
            "require_validated_executable": bool(
                self.require_validated_executable
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SafetyPredicate":
        return cls(**dict(value))


@dataclass
class PatchTemplate:
    scope: Scope | str
    level: int
    slot: str
    patch: dict[str, Any]
    chart_meta_class: str
    required_anchors: tuple[str, ...]
    anchor_coverage_threshold: float
    safety: SafetyPredicate
    provenance: dict[str, Any]
    touched_slots: tuple[str, ...] = ()
    projected_values: dict[str, Any] = field(default_factory=dict)
    constraint_ids: tuple[str, ...] = ()
    support: int = 0
    conflicts: int = 0
    validated_executable: bool = False
    timestamp: float = field(default_factory=time.time)
    id: str = ""
    scope_key: str | None = None
    created_step: int | None = None
    ttl_steps: int | None = None

    def __post_init__(self) -> None:
        self.scope = _as_scope(self.scope)
        if self.level not in {1, 2, 3, 4}:
            raise ValueError(f"level must be in 1..4, got {self.level!r}")
        self.slot = str(self.slot).strip()
        if not self.slot:
            raise ValueError("patch slot must be non-empty")
        self.patch = _normalized_mapping(self.patch)
        self.chart_meta_class = str(self.chart_meta_class).strip()
        if not self.chart_meta_class:
            raise ValueError("patch chart_meta_class must be non-empty")
        self.required_anchors = _normalized_strings(self.required_anchors)
        threshold = float(self.anchor_coverage_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("anchor_coverage_threshold must be in [0, 1]")
        self.anchor_coverage_threshold = threshold
        if isinstance(self.safety, Mapping):
            self.safety = SafetyPredicate.from_dict(self.safety)
        if not isinstance(self.safety, SafetyPredicate):
            raise TypeError("safety must be a SafetyPredicate")
        self.provenance = _normalized_mapping(self.provenance)
        self.touched_slots = _normalized_strings(
            self.touched_slots or (self.slot,)
        )
        self.projected_values = _normalized_mapping(self.projected_values)
        self.constraint_ids = _normalized_strings(self.constraint_ids)
        self.scope_key = None if self.scope_key is None else str(self.scope_key)
        self.support = max(0, int(self.support))
        self.conflicts = max(0, int(self.conflicts))
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite")
        if self.created_step is not None:
            self.created_step = int(self.created_step)
        if self.ttl_steps is not None:
            self.ttl_steps = max(0, int(self.ttl_steps))
        if not self.id:
            self.id = _stable_id(
                "patch",
                {
                    "scope": self.scope.value,
                    "scope_key": self.scope_key,
                    "level": self.level,
                    "slot": self.slot,
                    "patch": self.patch,
                    "chart_meta_class": self.chart_meta_class,
                    "required_anchors": self.required_anchors,
                    "anchor_coverage_threshold": self.anchor_coverage_threshold,
                    "safety": self.safety.to_dict(),
                    "provenance": self.provenance,
                    "touched_slots": self.touched_slots,
                    "projected_values": self.projected_values,
                    "constraint_ids": self.constraint_ids,
                },
            )

    @property
    def precedence_key(self) -> tuple[int, int, int, int, int, str]:
        return (
            self.scope.rank,
            5 - self.level,
            int(self.validated_executable),
            self.support,
            -self.conflicts,
            self.id,
        )

    def anchor_coverage(self, available_anchors: Iterable[str]) -> float:
        if not self.required_anchors:
            return 1.0
        available = tuple(str(anchor) for anchor in available_anchors)
        slot_anchors = {self.slot, *self.touched_slots}
        matched = sum(
            anchor in available
            or (
                anchor in slot_anchors
                and _slot_matches(anchor, available)
            )
            for anchor in self.required_anchors
        )
        return matched / len(self.required_anchors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope.value,
            "scope_key": self.scope_key,
            "level": self.level,
            "slot": self.slot,
            "patch": _json_copy(self.patch),
            "chart_meta_class": self.chart_meta_class,
            "required_anchors": list(self.required_anchors),
            "anchor_coverage_threshold": self.anchor_coverage_threshold,
            "safety": self.safety.to_dict(),
            "provenance": _json_copy(self.provenance),
            "touched_slots": list(self.touched_slots),
            "projected_values": _json_copy(self.projected_values),
            "constraint_ids": list(self.constraint_ids),
            "support": self.support,
            "conflicts": self.conflicts,
            "validated_executable": bool(self.validated_executable),
            "timestamp": float(self.timestamp),
            "created_step": self.created_step,
            "ttl_steps": self.ttl_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchTemplate":
        payload = dict(value)
        payload["safety"] = SafetyPredicate.from_dict(payload.get("safety") or {})
        return cls(**payload)


@dataclass(frozen=True)
class PatchReplayContext:
    chart_meta_class: str
    target_level: int
    available_anchors: frozenset[str]
    writable_slots: frozenset[str]
    frozen_values: Mapping[str, Any]
    constraints: Mapping[str, Any]
    panel_id: str | None = None
    panel_group: str | None = None
    scope_key: str | None = None

    @classmethod
    def create(
        cls,
        *,
        chart_meta_class: str,
        target_level: int,
        available_anchors: Iterable[str],
        writable_slots: Iterable[str],
        frozen_values: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
    ) -> "PatchReplayContext":
        return cls(
            chart_meta_class=str(chart_meta_class),
            target_level=int(target_level),
            available_anchors=frozenset(str(v) for v in available_anchors),
            writable_slots=frozenset(str(v) for v in writable_slots),
            frozen_values=_normalized_mapping(frozen_values),
            constraints=_normalized_mapping(constraints),
            panel_id=panel_id,
            panel_group=panel_group,
            scope_key=scope_key,
        )


@dataclass(frozen=True)
class PatchReplayDecision:
    allowed: bool
    template_id: str
    anchor_coverage: float
    reasons: tuple[str, ...] = ()
    patch: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "template_id": self.template_id,
            "anchor_coverage": self.anchor_coverage,
            "reasons": list(self.reasons),
            "patch": _json_copy(self.patch) if self.patch is not None else None,
        }


@dataclass(frozen=True)
class ConstraintResolution:
    selected: Mapping[str, ConstraintRecord]
    conflicts: Mapping[str, tuple[str, ...]]
    blocked: Mapping[str, tuple[str, ...]]

    @property
    def values(self) -> dict[str, Any]:
        return {slot: _json_copy(record.value) for slot, record in self.selected.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": {
                slot: self.selected[slot].to_dict()
                for slot in sorted(self.selected)
            },
            "conflicts": {
                slot: list(self.conflicts[slot]) for slot in sorted(self.conflicts)
            },
            "blocked": {
                slot: list(self.blocked[slot]) for slot in sorted(self.blocked)
            },
        }


@dataclass(frozen=True)
class InvariantDecision:
    allowed: bool
    frozen: Mapping[str, ConstraintRecord]
    violations: tuple[str, ...]

    @property
    def raise_scope(self) -> bool:
        return bool(self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "frozen": {
                slot: self.frozen[slot].to_dict() for slot in sorted(self.frozen)
            },
            "violations": list(self.violations),
            "raise_scope": self.raise_scope,
        }


@dataclass(frozen=True)
class OscillationDecision:
    allowed: bool
    reversal: bool
    reason: str
    rejected_count: int
    cooldown_until: int
    raise_scope: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reversal": self.reversal,
            "reason": self.reason,
            "rejected_count": self.rejected_count,
            "cooldown_until": self.cooldown_until,
            "raise_scope": self.raise_scope,
        }


class OscillationGuard:
    def __init__(
        self,
        *,
        cooldown_steps: int = 2,
        min_improvement: float = 0.01,
        rejections_before_raise: int = 2,
    ) -> None:
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.min_improvement = float(min_improvement)
        self.rejections_before_raise = max(1, int(rejections_before_raise))
        self._states: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(slot: str, scope_key: str | None) -> str:
        return f"{scope_key or '*'}::{slot}"

    def _state(self, slot: str, scope_key: str | None) -> dict[str, Any]:
        return self._states.setdefault(
            self._key(slot, scope_key),
            {
                "history": [],
                "rejected_count": 0,
                "cooldown_until": -1,
            },
        )

    def evaluate(
        self,
        *,
        slot: str,
        proposed_value: Any,
        improvement: float,
        step: int,
        scope_key: str | None = None,
    ) -> OscillationDecision:
        proposed = _json_copy(proposed_value)
        state = self._state(slot, scope_key)
        history = state["history"]
        current = history[-1] if history else _MISSING
        reversal = (
            len(history) >= 2
            and _value_equal(proposed, history[-2])
            and not _value_equal(proposed, history[-1])
        )
        in_cooldown = (
            int(step) < int(state["cooldown_until"])
            and current is not _MISSING
            and not _value_equal(proposed, current)
        )
        below_threshold = reversal and float(improvement) < self.min_improvement

        if in_cooldown or below_threshold:
            state["rejected_count"] += 1
            state["cooldown_until"] = max(
                int(state["cooldown_until"]), int(step) + self.cooldown_steps
            )
            reason = "cooldown" if in_cooldown else "reversal_below_min_improvement"
            rejected = int(state["rejected_count"])
            return OscillationDecision(
                allowed=False,
                reversal=reversal,
                reason=reason,
                rejected_count=rejected,
                cooldown_until=int(state["cooldown_until"]),
                raise_scope=rejected >= self.rejections_before_raise,
            )

        return OscillationDecision(
            allowed=True,
            reversal=reversal,
            reason="accepted",
            rejected_count=int(state["rejected_count"]),
            cooldown_until=int(state["cooldown_until"]),
            raise_scope=False,
        )

    def record_commit(
        self,
        *,
        slot: str,
        value: Any,
        scope_key: str | None = None,
    ) -> None:
        state = self._state(slot, scope_key)
        copied = _json_copy(value)
        history = state["history"]
        if not history or not _value_equal(history[-1], copied):
            history.append(copied)
            del history[:-4]
        state["rejected_count"] = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cooldown_steps": self.cooldown_steps,
            "min_improvement": self.min_improvement,
            "rejections_before_raise": self.rejections_before_raise,
            "states": {key: _json_copy(self._states[key]) for key in sorted(self._states)},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OscillationGuard":
        guard = cls(
            cooldown_steps=int(value.get("cooldown_steps", 2)),
            min_improvement=float(value.get("min_improvement", 0.01)),
            rejections_before_raise=int(value.get("rejections_before_raise", 2)),
        )
        guard._states = _normalized_mapping(value.get("states") or {})
        return guard


@dataclass(frozen=True)
class MemoryContext:
    resolution: ConstraintResolution
    patches: tuple[PatchTemplate, ...]
    patch_decisions: tuple[PatchReplayDecision, ...]

    def to_prompt_dict(self, *, patch_char_limit: int = 2400) -> dict[str, Any]:
        patch_entries = []
        for template in self.patches:
            payload = template.to_dict()
            raw_patch = _canonical_json(payload["patch"])
            if len(raw_patch) > patch_char_limit:
                payload["patch"] = {
                    "truncated": True,
                    "preview": raw_patch[:patch_char_limit],
                }
            patch_entries.append(payload)
        return {
            "constraints": {
                slot: {
                    "value": _json_copy(record.value),
                    "hard": record.hard,
                    "scope": record.scope.value,
                    "level": record.level,
                    "record_id": record.id,
                    "provenance": _json_copy(record.provenance),
                }
                for slot, record in sorted(self.resolution.selected.items())
            },
            "blocked_slots": sorted(self.resolution.blocked),
            "conflicts": {
                slot: list(record_ids)
                for slot, record_ids in sorted(
                    self.resolution.conflicts.items()
                )
            },
            "eligible_patch_templates": patch_entries,
            "patch_rejections": [
                decision.to_dict()
                for decision in self.patch_decisions
                if not decision.allowed
            ],
        }


class PersistentMemory:
    def __init__(
        self,
        *,
        capacity_by_level: Mapping[int, int] | None = None,
        ttl_steps_by_level: Mapping[int, int] | None = None,
        ttl_seconds: float | None = None,
        oscillation_guard: OscillationGuard | None = None,
    ) -> None:
        self.capacity_by_level = {
            int(level): max(0, int(limit))
            for level, limit in (
                dict(capacity_by_level or _DEFAULT_CAPACITY)
            ).items()
        }
        self.ttl_steps_by_level = {
            int(level): max(0, int(ttl))
            for level, ttl in (
                dict(ttl_steps_by_level or _DEFAULT_TTL_STEPS)
            ).items()
        }
        self.ttl_seconds = (
            None if ttl_seconds is None else max(0.0, float(ttl_seconds))
        )
        self.step = 0
        self.constraints: dict[str, ConstraintRecord] = {}
        self.patches: dict[str, PatchTemplate] = {}
        self.trace: list[dict[str, Any]] = []
        self._event_counter = 0
        self._conflict_pairs: set[tuple[str, str]] = set()
        self.oscillation_guard = oscillation_guard or OscillationGuard()

    def _event(self, event: str, **payload: Any) -> None:
        self._event_counter += 1
        self.trace.append(
            {
                "event_id": f"memory-event-{self._event_counter:08d}",
                "event": event,
                "step": self.step,
                "timestamp": time.time(),
                **_json_copy(payload),
            }
        )

    def advance_step(self, amount: int = 1) -> int:
        self.step += max(0, int(amount))
        return self.step

    def clear_records(self, *, reason: str) -> None:
        removed_constraints = len(self.constraints)
        removed_patches = len(self.patches)
        self.constraints.clear()
        self.patches.clear()
        self._conflict_pairs.clear()
        guard = self.oscillation_guard
        self.oscillation_guard = OscillationGuard(
            cooldown_steps=guard.cooldown_steps,
            min_improvement=guard.min_improvement,
            rejections_before_raise=guard.rejections_before_raise,
        )
        self._event(
            "memory_cleared",
            reason=str(reason),
            removed_constraints=removed_constraints,
            removed_patches=removed_patches,
        )

    def _prepare_record(
        self, record: ConstraintRecord | PatchTemplate
    ) -> ConstraintRecord | PatchTemplate:
        if record.created_step is None:
            record.created_step = self.step
        if record.ttl_steps is None and not (
            isinstance(record, ConstraintRecord) and record.hard
        ):
            record.ttl_steps = self.ttl_steps_by_level.get(record.level)
        return record

    def add_constraint(self, record: ConstraintRecord) -> ConstraintRecord:
        self._prepare_record(record)
        existing = self.constraints.get(record.id)
        if existing is not None:
            self._event("constraint_deduplicated", record_id=record.id)
            return existing

        for other in self.constraints.values():
            if other.slot != record.slot or _value_equal(other.value, record.value):
                continue
            pair = tuple(sorted((other.id, record.id)))
            if pair in self._conflict_pairs:
                continue
            self._conflict_pairs.add(pair)
            other.conflicts += 1
            record.conflicts += 1
            self._event(
                "constraint_conflict_observed",
                slot=record.slot,
                record_ids=list(pair),
            )

        self.constraints[record.id] = record
        self._event(
            "constraint_added",
            record_id=record.id,
            slot=record.slot,
            scope=record.scope.value,
            level=record.level,
        )
        return record

    def add_patch(self, record: PatchTemplate) -> PatchTemplate:
        self._prepare_record(record)
        existing = self.patches.get(record.id)
        if existing is not None:
            self._event("patch_deduplicated", record_id=record.id)
            return existing
        self.patches[record.id] = record
        self._event(
            "patch_added",
            record_id=record.id,
            slot=record.slot,
            scope=record.scope.value,
            level=record.level,
        )
        return record

    def append(
        self, record: ConstraintRecord | PatchTemplate | "PheromoneLink"
    ) -> ConstraintRecord | PatchTemplate:
        if isinstance(record, ConstraintRecord):
            return self.add_constraint(record)
        if isinstance(record, PatchTemplate):
            return self.add_patch(record)
        if isinstance(record, PheromoneLink):
            return self._append_legacy(record)
        raise TypeError(f"Unsupported memory record: {type(record).__name__}")

    def _append_legacy(
        self, link: "PheromoneLink"
    ) -> ConstraintRecord | PatchTemplate:
        slot = str(link.patch.get("slot") or link.msg or link.etype.value)
        provenance = {
            "source": "legacy_pheromone_link",
            "message": link.msg,
            "legacy_delta_ignored": True,
        }
        if link.etype == EvidenceType.constraint:
            return self.add_constraint(
                ConstraintRecord(
                    scope=Scope.PANEL,
                    level=link.level,
                    slot=slot,
                    value=link.patch.get("value", link.patch),
                    hard=False,
                    provenance=provenance,
                    timestamp=link.ts,
                )
            )
        return self.add_patch(
            PatchTemplate(
                scope=Scope.PANEL,
                level=link.level,
                slot=slot,
                patch=link.patch,
                chart_meta_class="positional",
                required_anchors=(slot,),
                anchor_coverage_threshold=1.0,
                safety=SafetyPredicate(allowed_slots=(slot,)),
                provenance=provenance,
                timestamp=link.ts,
            )
        )

    def mark_reuse(self, record_id: str, *, successful: bool) -> None:
        record: ConstraintRecord | PatchTemplate | None = self.constraints.get(
            record_id
        ) or self.patches.get(record_id)
        if record is None:
            raise KeyError(f"Unknown memory record: {record_id}")
        if successful:
            record.support += 1
        else:
            record.conflicts += 1
        self._event(
            "reuse_result",
            record_id=record_id,
            successful=bool(successful),
            support=record.support,
            conflicts=record.conflicts,
        )

    def _active(
        self,
        record: ConstraintRecord | PatchTemplate,
        *,
        step: int,
        now: float,
    ) -> bool:
        if isinstance(record, ConstraintRecord) and record.hard:
            return True
        if record.created_step is not None and record.ttl_steps is not None:
            if step - record.created_step > record.ttl_steps:
                return False
        if self.ttl_seconds is not None and now - record.timestamp > self.ttl_seconds:
            return False
        return True

    @staticmethod
    def _scope_applies(
        record: ConstraintRecord | PatchTemplate,
        *,
        panel_id: str | None,
        panel_group: str | None,
        scope_key: str | None,
    ) -> bool:
        if panel_id is None and panel_group is None and scope_key is None:
            return True
        if record.scope == Scope.FIGURE:
            return True
        if record.scope == Scope.PANEL_GROUP:
            return record.scope_key is None or record.scope_key == panel_group
        if record.scope == Scope.PANEL:
            return record.scope_key is None or record.scope_key == panel_id
        return record.scope_key is None or record.scope_key == scope_key

    def _bounded_constraints(
        self, records: Sequence[ConstraintRecord]
    ) -> list[ConstraintRecord]:
        hard = [record for record in records if record.hard]
        soft_by_level: dict[int, list[ConstraintRecord]] = {}
        for record in records:
            if not record.hard:
                soft_by_level.setdefault(record.level, []).append(record)
        bounded = list(hard)
        for level, candidates in soft_by_level.items():
            limit = self.capacity_by_level.get(level, 0)
            bounded.extend(
                sorted(candidates, key=lambda item: item.precedence_key, reverse=True)[
                    :limit
                ]
            )
        return sorted(bounded, key=lambda item: item.precedence_key, reverse=True)

    def _bounded_patches(
        self, records: Sequence[PatchTemplate]
    ) -> list[PatchTemplate]:
        by_level: dict[int, list[PatchTemplate]] = {}
        for record in records:
            by_level.setdefault(record.level, []).append(record)
        bounded: list[PatchTemplate] = []
        for level, candidates in by_level.items():
            limit = self.capacity_by_level.get(level, 0)
            bounded.extend(
                sorted(candidates, key=lambda item: item.precedence_key, reverse=True)[
                    :limit
                ]
            )
        return sorted(bounded, key=lambda item: item.precedence_key, reverse=True)

    def active_constraints(
        self,
        *,
        step: int | None = None,
        now: float | None = None,
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
    ) -> list[ConstraintRecord]:
        query_step = self.step if step is None else int(step)
        query_now = time.time() if now is None else float(now)
        records = [
            record
            for record in self.constraints.values()
            if self._active(record, step=query_step, now=query_now)
            and self._scope_applies(
                record,
                panel_id=panel_id,
                panel_group=panel_group,
                scope_key=scope_key,
            )
        ]
        return self._bounded_constraints(records)

    def active_patches(
        self,
        *,
        step: int | None = None,
        now: float | None = None,
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
    ) -> list[PatchTemplate]:
        query_step = self.step if step is None else int(step)
        query_now = time.time() if now is None else float(now)
        records = [
            record
            for record in self.patches.values()
            if self._active(record, step=query_step, now=query_now)
            and self._scope_applies(
                record,
                panel_id=panel_id,
                panel_group=panel_group,
                scope_key=scope_key,
            )
        ]
        return self._bounded_patches(records)

    def resolve_constraints(
        self,
        *,
        step: int | None = None,
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
    ) -> ConstraintResolution:
        grouped: dict[str, list[ConstraintRecord]] = {}
        for record in self.active_constraints(
            step=step,
            panel_id=panel_id,
            panel_group=panel_group,
            scope_key=scope_key,
        ):
            grouped.setdefault(record.slot, []).append(record)

        selected: dict[str, ConstraintRecord] = {}
        conflicts: dict[str, tuple[str, ...]] = {}
        blocked: dict[str, tuple[str, ...]] = {}
        for slot in sorted(grouped):
            records = grouped[slot]
            values = {_canonical_json(record.value) for record in records}
            if len(values) > 1:
                conflicts[slot] = tuple(sorted(record.id for record in records))
            hard_source = [
                record
                for record in records
                if record.hard and record.grounding_rank == 2
            ]
            if len({_canonical_json(record.value) for record in hard_source}) > 1:
                blocked[slot] = tuple(sorted(record.id for record in hard_source))
            selected[slot] = max(records, key=lambda record: record.precedence_key)
        return ConstraintResolution(
            selected=selected,
            conflicts=conflicts,
            blocked=blocked,
        )

    def constraint_projection(
        self,
        *,
        step: int | None = None,
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
    ) -> ConstraintResolution:
        return self.resolve_constraints(
            step=step,
            panel_id=panel_id,
            panel_group=panel_group,
            scope_key=scope_key,
        )

    def higher_scope_invariants(
        self,
        *,
        target_level: int,
        step: int | None = None,
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
    ) -> dict[str, ConstraintRecord]:
        resolution = self.constraint_projection(
            step=step,
            panel_id=panel_id,
            panel_group=panel_group,
            scope_key=scope_key,
        )
        return {
            slot: record
            for slot, record in resolution.selected.items()
            if record.level < int(target_level)
        }

    def check_invariant_freeze(
        self,
        *,
        target_level: int,
        touched_slots: Iterable[str],
        proposed_values: Mapping[str, Any] | None = None,
        step: int | None = None,
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
    ) -> InvariantDecision:
        frozen = self.higher_scope_invariants(
            target_level=target_level,
            step=step,
            panel_id=panel_id,
            panel_group=panel_group,
            scope_key=scope_key,
        )
        proposals = dict(proposed_values or {})
        violations: list[str] = []
        for slot in sorted({str(value) for value in touched_slots}):
            record = frozen.get(slot)
            if record is None:
                continue
            proposed = proposals.get(slot, _MISSING)
            if proposed is _MISSING or not _value_equal(proposed, record.value):
                violations.append(slot)
        return InvariantDecision(
            allowed=not violations,
            frozen=frozen,
            violations=tuple(violations),
        )

    def evaluate_patch(
        self, template: PatchTemplate, context: PatchReplayContext
    ) -> PatchReplayDecision:
        reasons: list[str] = []
        coverage = template.anchor_coverage(context.available_anchors)
        if template.chart_meta_class != context.chart_meta_class:
            reasons.append("chart_meta_class_mismatch")
        if template.level != context.target_level:
            reasons.append("level_mismatch")
        if coverage < template.anchor_coverage_threshold:
            reasons.append("insufficient_anchor_coverage")
        if template.safety.require_validated_executable and not (
            template.validated_executable
        ):
            reasons.append("not_executable_validated")
        if not all(
            _slot_matches(slot, context.writable_slots)
            for slot in template.touched_slots
        ):
            reasons.append("write_outside_target_scope")
        if template.safety.allowed_slots and not set(
            template.touched_slots
        ).issubset(template.safety.allowed_slots):
            reasons.append("safety_allowed_slots_violation")
        if set(template.touched_slots) & set(template.safety.forbidden_slots):
            reasons.append("safety_forbidden_slot")
        for slot, required in template.safety.required_constraints.items():
            if slot not in context.constraints or not _value_equal(
                context.constraints[slot], required
            ):
                reasons.append(f"required_constraint_mismatch:{slot}")
        for slot in sorted(set(template.touched_slots) & set(context.frozen_values)):
            projected = template.projected_values.get(slot, _MISSING)
            if projected is _MISSING or not _value_equal(
                projected, context.frozen_values[slot]
            ):
                reasons.append(f"frozen_invariant:{slot}")
        allowed = not reasons
        return PatchReplayDecision(
            allowed=allowed,
            template_id=template.id,
            anchor_coverage=coverage,
            reasons=tuple(reasons),
            patch=copy.deepcopy(template.patch) if allowed else None,
        )

    def reuse_context(
        self,
        *,
        chart_meta_class: str,
        target_level: int,
        available_anchors: Iterable[str],
        writable_slots: Iterable[str],
        panel_id: str | None = None,
        panel_group: str | None = None,
        scope_key: str | None = None,
        step: int | None = None,
        include_constraints: bool = True,
        include_patches: bool = True,
    ) -> MemoryContext:
        if include_constraints:
            resolution = self.constraint_projection(
                step=step,
                panel_id=panel_id,
                panel_group=panel_group,
                scope_key=scope_key,
            )
        else:
            resolution = ConstraintResolution(selected={}, conflicts={}, blocked={})
        frozen = {
            slot: record.value
            for slot, record in resolution.selected.items()
            if record.level < int(target_level)
        }
        context = PatchReplayContext.create(
            chart_meta_class=chart_meta_class,
            target_level=target_level,
            available_anchors=available_anchors,
            writable_slots=writable_slots,
            frozen_values=frozen,
            constraints=resolution.values,
            panel_id=panel_id,
            panel_group=panel_group,
            scope_key=scope_key,
        )
        decisions = (
            tuple(
                self.evaluate_patch(template, context)
                for template in self.active_patches(
                    step=step,
                    panel_id=panel_id,
                    panel_group=panel_group,
                    scope_key=scope_key,
                )
            )
            if include_patches
            else ()
        )
        allowed_ids = {decision.template_id for decision in decisions if decision.allowed}
        templates = (
            tuple(
                template
                for template in self.active_patches(
                    step=step,
                    panel_id=panel_id,
                    panel_group=panel_group,
                    scope_key=scope_key,
                )
                if template.id in allowed_ids
            )
            if include_patches
            else ()
        )
        return MemoryContext(
            resolution=resolution,
            patches=templates,
            patch_decisions=decisions,
        )

    def check_oscillation(
        self,
        *,
        slot: str,
        proposed_value: Any,
        improvement: float,
        scope_key: str | None = None,
        step: int | None = None,
    ) -> OscillationDecision:
        decision = self.oscillation_guard.evaluate(
            slot=slot,
            proposed_value=proposed_value,
            improvement=improvement,
            step=self.step if step is None else int(step),
            scope_key=scope_key,
        )
        self._event(
            "oscillation_check",
            slot=slot,
            scope_key=scope_key,
            improvement=float(improvement),
            decision=decision.to_dict(),
        )
        return decision

    def record_slot_commit(
        self, *, slot: str, value: Any, scope_key: str | None = None
    ) -> None:
        self.oscillation_guard.record_commit(
            slot=slot, value=value, scope_key=scope_key
        )

    def snapshot(self, *, step: int | None = None) -> dict[str, Any]:
        query_step = self.step if step is None else int(step)
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "config": {
                "capacity_by_level": {
                    str(level): self.capacity_by_level[level]
                    for level in sorted(self.capacity_by_level)
                },
                "ttl_steps_by_level": {
                    str(level): self.ttl_steps_by_level[level]
                    for level in sorted(self.ttl_steps_by_level)
                },
                "ttl_seconds": self.ttl_seconds,
            },
            "step": self.step,
            "constraints": [
                self.constraints[record_id].to_dict()
                for record_id in sorted(self.constraints)
            ],
            "patch_templates": [
                self.patches[record_id].to_dict()
                for record_id in sorted(self.patches)
            ],
            "active_cache": {
                "step": query_step,
                "constraint_ids": [
                    record.id for record in self.active_constraints(step=query_step)
                ],
                "patch_ids": [
                    record.id for record in self.active_patches(step=query_step)
                ],
            },
            "oscillation_guard": self.oscillation_guard.to_dict(),
            "trace": _json_copy(self.trace),
        }

    def dumps(self, *, indent: int = 2, step: int | None = None) -> str:
        return (
            json.dumps(
                self.snapshot(step=step),
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
                allow_nan=False,
            )
            + "\n"
        )

    def write_snapshot(self, path: str | Path, *, step: int | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.dumps(step=step), encoding="utf-8")
        return output

    def write_trace(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.trace,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    def compatibility_records(self) -> list[dict[str, Any]]:
        """List-shaped export for historical ``pheromones.json`` consumers."""

        records: list[dict[str, Any]] = []
        for record_id in sorted(self.constraints):
            records.append(
                {
                    "record_type": "constraint",
                    **self.constraints[record_id].to_dict(),
                }
            )
        for record_id in sorted(self.patches):
            records.append(
                {
                    "record_type": "patch_template",
                    **self.patches[record_id].to_dict(),
                }
            )
        return records

    def write_compatibility(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.compatibility_records(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PersistentMemory":
        if int(value.get("schema_version", 0)) != MEMORY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported memory schema version: {value.get('schema_version')!r}"
            )
        config = dict(value.get("config") or {})
        memory = cls(
            capacity_by_level={
                int(level): int(limit)
                for level, limit in dict(
                    config.get("capacity_by_level") or _DEFAULT_CAPACITY
                ).items()
            },
            ttl_steps_by_level={
                int(level): int(ttl)
                for level, ttl in dict(
                    config.get("ttl_steps_by_level") or _DEFAULT_TTL_STEPS
                ).items()
            },
            ttl_seconds=config.get("ttl_seconds"),
            oscillation_guard=OscillationGuard.from_dict(
                value.get("oscillation_guard") or {}
            ),
        )
        memory.step = int(value.get("step", 0))
        memory.constraints = {
            record.id: record
            for record in (
                ConstraintRecord.from_dict(item)
                for item in value.get("constraints") or []
            )
        }
        memory.patches = {
            record.id: record
            for record in (
                PatchTemplate.from_dict(item)
                for item in value.get("patch_templates") or []
            )
        }
        memory.trace = _json_copy(value.get("trace") or [])
        memory._event_counter = len(memory.trace)
        for left in memory.constraints.values():
            for right in memory.constraints.values():
                if left.id >= right.id:
                    continue
                if left.slot == right.slot and not _value_equal(
                    left.value, right.value
                ):
                    memory._conflict_pairs.add((left.id, right.id))
        return memory

    @classmethod
    def loads(cls, value: str) -> "PersistentMemory":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Memory JSON must contain an object")
        return cls.from_dict(parsed)

    @classmethod
    def load(cls, path: str | Path) -> "PersistentMemory":
        return cls.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class PheromoneLink:
    """Deprecated input adapter retained for old imports."""

    level: int
    etype: EvidenceType
    delta: dict[str, float]
    patch: dict[str, Any]
    msg: str = ""
    ts: float = field(default_factory=time.time)


class PheroStore(PersistentMemory):
    """Compatibility name for the persistent memory implementation."""

    @property
    def links(self) -> list[ConstraintRecord | PatchTemplate]:
        return [
            *[self.constraints[key] for key in sorted(self.constraints)],
            *[self.patches[key] for key in sorted(self.patches)],
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.constraints) + len(self.patches),
            "constraints": len(self.constraints),
            "patch_templates": len(self.patches),
            "active_constraints": len(self.active_constraints()),
            "active_patch_templates": len(self.active_patches()),
        }

    def to_json(self) -> list[dict[str, Any]]:
        return self.compatibility_records()

    def tail(self, n: int = 3) -> list[ConstraintRecord | PatchTemplate]:
        if n <= 0:
            return []
        return self.links[-n:]


def constraint_projection(
    records: Iterable[ConstraintRecord | PatchTemplate],
) -> tuple[ConstraintRecord, ...]:
    """Return only typed constraint records; patch templates never masquerade as facts."""

    return tuple(
        sorted(
            (record for record in records if isinstance(record, ConstraintRecord)),
            key=lambda record: record.id,
        )
    )


__all__ = [
    "ConstraintRecord",
    "ConstraintResolution",
    "EvidenceType",
    "InvariantDecision",
    "MemoryContext",
    "OscillationDecision",
    "OscillationGuard",
    "PatchReplayContext",
    "PatchReplayDecision",
    "PatchTemplate",
    "PersistentMemory",
    "PheroStore",
    "PheromoneLink",
    "SafetyPredicate",
    "Scope",
    "chart_meta_class",
    "constraint_projection",
]
