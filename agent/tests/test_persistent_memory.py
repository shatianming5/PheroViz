from __future__ import annotations

import json
from pathlib import Path

from app.services.pheromones import (
    ConstraintRecord,
    OscillationGuard,
    PatchReplayContext,
    PatchTemplate,
    PersistentMemory,
    SafetyPredicate,
    Scope,
)


def constraint(
    *,
    record_id: str,
    value: object,
    slot: str = "theme.palette_global",
    scope: Scope = Scope.FIGURE,
    level: int = 1,
    hard: bool = False,
    ttl_steps: int | None = None,
) -> ConstraintRecord:
    return ConstraintRecord(
        id=record_id,
        scope=scope,
        level=level,
        slot=slot,
        value=value,
        hard=hard,
        provenance={"source": record_id, "grounding": "executable"},
        validated_executable=True,
        timestamp=10.0,
        ttl_steps=ttl_steps,
    )


def test_conflict_resolution_is_insertion_order_independent() -> None:
    records = [
        constraint(record_id="constraint-a", value="viridis"),
        constraint(record_id="constraint-b", value="cividis"),
    ]
    forward = PersistentMemory()
    reverse = PersistentMemory()
    for record in records:
        forward.add_constraint(ConstraintRecord.from_dict(record.to_dict()))
    for record in reversed(records):
        reverse.add_constraint(ConstraintRecord.from_dict(record.to_dict()))

    forward_resolution = forward.resolve_constraints()
    reverse_resolution = reverse.resolve_constraints()
    assert forward_resolution.values == reverse_resolution.values
    assert (
        forward_resolution.selected["theme.palette_global"].id
        == reverse_resolution.selected["theme.palette_global"].id
        == "constraint-b"
    )


def test_higher_scope_invariant_freeze() -> None:
    memory = PersistentMemory()
    memory.add_constraint(
        constraint(record_id="global-font", slot="theme.font", value="Arial")
    )

    denied = memory.check_invariant_freeze(
        target_level=4,
        touched_slots=["theme.font", "axes.title"],
        proposed_values={"theme.font": "Helvetica"},
    )
    assert not denied.allowed
    assert denied.raise_scope
    assert denied.violations == ("theme.font",)

    unchanged = memory.check_invariant_freeze(
        target_level=4,
        touched_slots=["theme.font"],
        proposed_values={"theme.font": "Arial"},
    )
    assert unchanged.allowed


def test_safe_and_unsafe_patch_replay_keep_constraint_projection() -> None:
    memory = PersistentMemory()
    memory.add_constraint(
        constraint(record_id="palette", value="viridis")
    )
    template = PatchTemplate(
        id="patch-line",
        scope=Scope.FIGURE,
        level=3,
        slot="marks.line",
        patch={"slots": {"marks.line": "return ax.plot(df['x'], df['y'])"}},
        chart_meta_class="positional",
        required_anchors=("marks.line", "x", "y"),
        anchor_coverage_threshold=1.0,
        safety=SafetyPredicate(
            allowed_slots=("marks.line",),
            required_constraints={"theme.palette_global": "viridis"},
        ),
        provenance={"source": "panel-a"},
        touched_slots=("marks.line",),
        validated_executable=True,
        timestamp=10.0,
    )
    memory.add_patch(template)

    safe = memory.reuse_context(
        chart_meta_class="positional",
        target_level=3,
        available_anchors=("marks.line", "x", "y"),
        writable_slots=("marks.line",),
        panel_id="panel-b",
    )
    assert [patch.id for patch in safe.patches] == ["patch-line"]
    assert safe.resolution.values["theme.palette_global"] == "viridis"

    unsafe = memory.reuse_context(
        chart_meta_class="matrix",
        target_level=3,
        available_anchors=("marks.line", "x"),
        writable_slots=("marks.line",),
        panel_id="panel-b",
    )
    assert not unsafe.patches
    assert unsafe.resolution.values["theme.palette_global"] == "viridis"
    reasons = {
        reason
        for decision in unsafe.patch_decisions
        for reason in decision.reasons
    }
    assert "chart_meta_class_mismatch" in reasons
    assert "insufficient_anchor_coverage" in reasons

    frozen_context = PatchReplayContext.create(
        chart_meta_class="positional",
        target_level=3,
        available_anchors=("marks.line", "x", "y"),
        writable_slots=("marks.line",),
        frozen_values={"marks.line": "different"},
        constraints={"theme.palette_global": "viridis"},
    )
    assert not memory.evaluate_patch(template, frozen_context).allowed


def test_patch_replay_matches_concrete_slots_against_stage_prefixes() -> None:
    memory = PersistentMemory()
    template = PatchTemplate(
        id="line-template",
        scope=Scope.PANEL,
        level=3,
        slot="marks.line.main",
        patch={"slots": {"marks.line.main": "return []"}},
        chart_meta_class="positional",
        required_anchors=("marks.line.main", "x"),
        anchor_coverage_threshold=1.0,
        safety=SafetyPredicate(allowed_slots=("marks.line.main",)),
        provenance={"source": "test"},
        touched_slots=("marks.line.main",),
        validated_executable=True,
    )
    memory.add_patch(template)

    context = memory.reuse_context(
        chart_meta_class="positional",
        target_level=3,
        available_anchors=("marks.*", "scales.*", "x"),
        writable_slots=("marks.*", "scales.*", "colorbar.apply"),
        panel_id="panel-a",
        include_constraints=False,
        include_patches=True,
    )

    assert [item.id for item in context.patches] == ["line-template"]
    assert context.patch_decisions[0].allowed is True


def test_stage_prefix_does_not_satisfy_missing_dataframe_anchor() -> None:
    memory = PersistentMemory()
    template = PatchTemplate(
        id="column-sensitive-template",
        scope=Scope.PANEL,
        level=3,
        slot="marks.line.main",
        patch={"slots": {"marks.line.main": "return []"}},
        chart_meta_class="positional",
        required_anchors=("marks.line.main", "marks.value"),
        anchor_coverage_threshold=1.0,
        safety=SafetyPredicate(allowed_slots=("marks.line.main",)),
        provenance={"source": "test"},
        touched_slots=("marks.line.main",),
        validated_executable=True,
    )
    memory.add_patch(template)

    context = memory.reuse_context(
        chart_meta_class="positional",
        target_level=3,
        available_anchors=("marks.*", "scales.*"),
        writable_slots=("marks.*", "scales.*", "colorbar.apply"),
        panel_id="panel-a",
        include_constraints=False,
        include_patches=True,
    )

    assert context.patches == ()
    assert context.patch_decisions[0].anchor_coverage == 0.5
    assert context.patch_decisions[0].reasons == (
        "insufficient_anchor_coverage",
    )


def test_prompt_context_exposes_conflicting_record_ids() -> None:
    memory = PersistentMemory()
    memory.add_constraint(
        constraint(record_id="palette-a", value="viridis")
    )
    memory.add_constraint(
        constraint(record_id="palette-b", value="cividis")
    )

    prompt = memory.reuse_context(
        chart_meta_class="positional",
        target_level=1,
        available_anchors=("spec.*",),
        writable_slots=("spec.*",),
    ).to_prompt_dict()

    assert prompt["conflicts"]["theme.palette_global"] == [
        "palette-a",
        "palette-b",
    ]


def test_ttl_and_capacity_bound_active_views() -> None:
    memory = PersistentMemory(
        capacity_by_level={1: 1, 2: 1, 3: 1, 4: 1},
        ttl_steps_by_level={1: 1, 2: 1, 3: 1, 4: 1},
    )
    memory.add_constraint(
        constraint(record_id="soft-a", value="a", ttl_steps=1)
    )
    memory.add_constraint(
        constraint(
            record_id="soft-b",
            slot="theme.font",
            value="Arial",
            ttl_steps=1,
        )
    )
    hard = constraint(
        record_id="hard-source",
        slot="encoding.x",
        value="category",
        scope=Scope.PANEL,
        level=2,
        hard=True,
        ttl_steps=0,
    )
    memory.add_constraint(hard)
    assert len([record for record in memory.active_constraints(step=0) if not record.hard]) == 1
    assert hard in memory.active_constraints(step=100)
    assert not [
        record
        for record in memory.active_constraints(step=2)
        if not record.hard
    ]


def test_oscillation_guard_cooldown_and_raise_scope() -> None:
    guard = OscillationGuard(
        cooldown_steps=2,
        min_improvement=0.05,
        rejections_before_raise=2,
    )
    guard.record_commit(slot="theme.font", value="Arial")
    guard.record_commit(slot="theme.font", value="Helvetica")

    first = guard.evaluate(
        slot="theme.font",
        proposed_value="Arial",
        improvement=0.01,
        step=2,
    )
    second = guard.evaluate(
        slot="theme.font",
        proposed_value="Arial",
        improvement=0.01,
        step=3,
    )
    assert not first.allowed and first.reversal and not first.raise_scope
    assert not second.allowed and second.raise_scope

    permissive = OscillationGuard(min_improvement=0.05)
    permissive.record_commit(slot="theme.font", value="Arial")
    permissive.record_commit(slot="theme.font", value="Helvetica")
    accepted = permissive.evaluate(
        slot="theme.font",
        proposed_value="Arial",
        improvement=0.10,
        step=2,
    )
    assert accepted.allowed and accepted.reversal


def test_stable_artifact_serialization_round_trip(tmp_path: Path) -> None:
    memory = PersistentMemory()
    record = constraint(record_id="palette", value="viridis")
    memory.add_constraint(record)
    memory.mark_reuse(record.id, successful=True)
    template = PatchTemplate(
        id="patch",
        scope=Scope.FIGURE,
        level=4,
        slot="axes.title",
        patch={"slots": {"axes.title": "return None"}},
        chart_meta_class="positional",
        required_anchors=("axes.title",),
        anchor_coverage_threshold=1.0,
        safety=SafetyPredicate(allowed_slots=("axes.title",)),
        provenance={"source": "test"},
        validated_executable=True,
        timestamp=10.0,
    )
    memory.add_patch(template)
    path = memory.write_snapshot(tmp_path / "memory.json")
    compatibility_path = memory.write_compatibility(tmp_path / "pheromones.json")
    restored = PersistentMemory.load(path)

    assert restored.dumps() == memory.dumps()
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert restored.resolve_constraints().values == {
        "theme.palette_global": "viridis"
    }
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    assert {record["record_type"] for record in compatibility} == {
        "constraint",
        "patch_template",
    }


def test_clear_records_resets_active_memory_and_oscillation_state() -> None:
    memory = PersistentMemory()
    memory.add_constraint(constraint(record_id="palette", value="viridis"))
    memory.oscillation_guard.record_commit(slot="theme.font", value="Arial")

    memory.clear_records(reason="next_ephemeral_round")

    assert memory.constraints == {}
    assert memory.patches == {}
    assert memory.oscillation_guard.to_dict()["states"] == {}
    assert memory.trace[-1]["event"] == "memory_cleared"
