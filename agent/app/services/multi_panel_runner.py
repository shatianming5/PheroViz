from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image

from app.evaluation import (
    combine_figure_manifests,
    evaluate_cohesion,
    validate_expectation,
)
from app.services.model_client import ModelClient
from app.services.pheromones import PersistentMemory
from app.services.single_chain_runner import MEMORY_MODES, iter_chain


PanelProgressCallback = Callable[[str, dict[str, Any]], None]
_SAFE_PANEL_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _load_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(manifest, Mapping):
        return dict(manifest), (
            Path(base_dir).resolve() if base_dir is not None else Path.cwd()
        )
    path = Path(manifest).resolve()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("multi-panel manifest must contain a JSON object")
    return parsed, path.parent


def _validated_panels(
    manifest: Mapping[str, Any], *, base_dir: Path
) -> list[dict[str, Any]]:
    raw_panels = manifest.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise ValueError("multi-panel manifest requires a non-empty 'panels' list")
    panels: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_panel in enumerate(raw_panels):
        if not isinstance(raw_panel, Mapping):
            raise ValueError(f"panel[{index}] must be an object")
        panel = dict(raw_panel)
        panel_id = str(panel.get("id") or "").strip()
        if not panel_id or not _SAFE_PANEL_ID.fullmatch(panel_id):
            raise ValueError(
                f"panel[{index}].id must match {_SAFE_PANEL_ID.pattern!r}"
            )
        if panel_id in seen:
            raise ValueError(f"duplicate panel id: {panel_id}")
        seen.add(panel_id)
        data_value = panel.get("data_path") or panel.get("excel_path")
        if not data_value:
            raise ValueError(f"panel[{index}] requires data_path")
        data_path = Path(str(data_value))
        if not data_path.is_absolute():
            data_path = (base_dir / data_path).resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"panel data does not exist: {data_path}")
        chart_family = str(panel.get("chart_family") or "").strip()
        if not chart_family:
            raise ValueError(f"panel[{index}] requires chart_family")
        user_goal = str(panel.get("user_goal") or panel.get("intent_text") or "").strip()
        if not user_goal:
            raise ValueError(f"panel[{index}] requires user_goal")
        intent = panel.get("intent")
        if intent is not None and not isinstance(intent, Mapping):
            raise ValueError(f"panel[{index}].intent must be an object")
        panels.append(
            {
                "id": panel_id,
                "data_path": str(data_path),
                "user_goal": user_goal,
                "chart_family": chart_family,
                "sheet": panel.get("sheet"),
                "intent": dict(intent or {}),
                "evaluation_expectation": panel.get("evaluation_expectation"),
            }
        )
    return panels


def _compose_panel_images(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int,
) -> Path:
    if not image_paths:
        raise ValueError("Cannot compose an empty panel image list")
    if columns < 1:
        raise ValueError("layout columns must be positive")
    images = [Image.open(path).convert("RGBA") for path in image_paths]
    try:
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        rows = math.ceil(len(images) / columns)
        canvas = Image.new(
            "RGBA",
            (columns * cell_width, rows * cell_height),
            (255, 255, 255, 255),
        )
        for index, image in enumerate(images):
            row, column = divmod(index, columns)
            left = column * cell_width + (cell_width - image.width) // 2
            top = row * cell_height + (cell_height - image.height) // 2
            canvas.alpha_composite(image, (left, top))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.convert("RGB").save(output_path)
        return output_path
    finally:
        for image in images:
            image.close()


def run_multi_panel(
    manifest: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    rounds: int | None = None,
    memory: PersistentMemory | None = None,
    model_client: ModelClient | None = None,
    progress_callback: PanelProgressCallback | None = None,
    initial_generation: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    memory_mode: str | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest_data, manifest_base_dir = _load_manifest(
        manifest,
        base_dir=base_dir,
    )
    panels = _validated_panels(
        manifest_data,
        base_dir=manifest_base_dir,
    )
    full_expectation = manifest_data.get("evaluation_expectation")
    metric_config = manifest_data.get("metric_config")
    expectations_by_panel: dict[str, dict[str, Any]] = {}
    if full_expectation is not None:
        if not isinstance(full_expectation, Mapping):
            raise ValueError("evaluation_expectation must be an object")
        full_expectation = dict(full_expectation)
        validate_expectation(full_expectation)
        expectations_by_panel = {
            str(panel["panel_id"]): dict(panel)
            for panel in full_expectation["panels"]
        }
        missing = [
            panel["id"]
            for panel in panels
            if panel["id"] not in expectations_by_panel
        ]
        if missing:
            raise ValueError(
                "evaluation_expectation is missing panels: " + ", ".join(missing)
            )
    configured_rounds = int(
        rounds if rounds is not None else manifest_data.get("rounds", 1)
    )
    if configured_rounds < 1:
        raise ValueError("rounds must be at least 1")
    configured_initial_generation = str(
        initial_generation
        if initial_generation is not None
        else manifest_data.get("initial_generation", "defaults")
    ).strip().lower()
    if configured_initial_generation == "default":
        configured_initial_generation = "defaults"
    if configured_initial_generation not in {
        "defaults",
        "model_spec",
        "model",
    }:
        raise ValueError(
            "initial_generation must be 'defaults', 'model_spec', or 'model'"
        )
    configured_memory_mode = str(
        memory_mode
        if memory_mode is not None
        else manifest_data.get("memory_mode", "full")
    ).strip().lower()
    if configured_memory_mode not in MEMORY_MODES:
        raise ValueError(
            f"memory_mode must be one of {', '.join(MEMORY_MODES)}"
        )
    configured_seed = (
        seed if seed is not None else manifest_data.get("seed")
    )
    configured_temperature = (
        temperature
        if temperature is not None
        else manifest_data.get("temperature")
    )
    if configured_seed is not None:
        configured_seed = int(configured_seed)
    if configured_temperature is not None:
        configured_temperature = float(configured_temperature)
    panel_group = str(
        manifest_data.get("panel_group")
        or manifest_data.get("figure_id")
        or "multi-panel-figure"
    )
    if output_dir is None:
        stamp = f"{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        active_output_dir = Path("runs") / f"multi_{stamp}"
    else:
        active_output_dir = Path(output_dir)
    active_output_dir.mkdir(parents=True, exist_ok=True)

    shared_memory = memory or PersistentMemory()
    shared_untyped_memory: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {
        panel["id"]: [] for panel in panels
    }
    render_counts = {panel["id"]: 0 for panel in panels}
    generators = {}

    for panel in panels:
        panel_id = panel["id"]

        def panel_progress(
            event: str,
            payload: dict[str, Any],
            *,
            current_panel_id: str = panel_id,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    "panel_event",
                    {
                        "panel_id": current_panel_id,
                        "event": event,
                        "payload": payload,
                    },
                )

        generators[panel_id] = iter_chain(
            panel["data_path"],
            panel["user_goal"],
            panel["chart_family"],
            rounds=configured_rounds,
            sheet=panel["sheet"],
            intent=panel["intent"],
            progress_callback=panel_progress,
            memory=shared_memory,
            model_client=model_client,
            panel_id=panel_id,
            panel_group=panel_group,
            run_dir=active_output_dir / "panels" / panel_id,
            initial_generation=configured_initial_generation,
            seed=configured_seed,
            temperature=configured_temperature,
            memory_mode=configured_memory_mode,
            untyped_memory=shared_untyped_memory,
            manage_ephemeral_reset=False,
            evaluation_expectation=(
                {
                    "schema_version": full_expectation["schema_version"],
                    "panels": [expectations_by_panel[panel_id]],
                    "panel_groups": [],
                }
                if full_expectation is not None
                else panel.get("evaluation_expectation")
            ),
            metric_config=metric_config,
        )

    panel_order = [panel["id"] for panel in panels]
    scheduler_tick = 0
    for global_round in range(1, configured_rounds + 1):
        if configured_memory_mode == "ephemeral" and global_round > 1:
            shared_memory.clear_records(
                reason=f"multi_panel_global_round_{global_round}"
            )
        for panel_id in panel_order:
            try:
                result = next(generators[panel_id])
            except StopIteration as exc:
                raise RuntimeError(
                    f"Panel {panel_id!r} stopped before round {global_round}"
                ) from exc
            scheduler_tick += 1
            render_counts[panel_id] += 1
            histories[panel_id].append(result)
            event = {
                "tick": scheduler_tick,
                "panel_id": panel_id,
                "round": result["round"],
                "artifact_path": result["artifact_path"],
                "render_count": render_counts[panel_id],
            }
            schedule.append(event)
            if progress_callback is not None:
                progress_callback("round_robin_commit", dict(event))

    memory_snapshot_path = shared_memory.write_snapshot(
        active_output_dir / "shared_memory_snapshot.json"
    )
    memory_trace_path = shared_memory.write_trace(
        active_output_dir / "shared_memory_trace.json"
    )
    compatibility_path = shared_memory.write_compatibility(
        active_output_dir / "pheromones.json"
    )
    untyped_memory_path: Path | None = None
    if configured_memory_mode.strip().lower() == "untyped":
        untyped_memory_path = active_output_dir / "untyped_memory.json"
        untyped_memory_path.write_text(
            json.dumps(
                shared_untyped_memory,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    schedule_path = active_output_dir / "schedule_trace.json"
    schedule_path.write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    panel_results = {}
    final_image_paths: list[Path] = []
    panel_programmatic: dict[str, dict[str, Any]] = {}
    for panel in panels:
        panel_id = panel["id"]
        history = histories[panel_id]
        panel_results[panel_id] = {
            "render_count": render_counts[panel_id],
            "artifacts": [result["artifact_path"] for result in history],
            "result": history[-1] if history else None,
            "run_dir": str(active_output_dir / "panels" / panel_id),
        }
        if history:
            final_image = history[-1].get("png_path")
            if final_image:
                final_image_paths.append(Path(str(final_image)))
            evaluation = history[-1].get("programmatic_evaluation")
            if isinstance(evaluation, dict):
                panel_programmatic[panel_id] = evaluation

    layout = manifest_data.get("layout") or {}
    columns = int(layout.get("columns") or math.ceil(math.sqrt(len(panels))))
    combined_image_path = _compose_panel_images(
        final_image_paths,
        active_output_dir / "combined_figure.png",
        columns=columns,
    )

    programmatic_path: Path | None = None
    programmatic_summary: dict[str, Any] | None = None
    if full_expectation is not None:
        if set(panel_programmatic) != set(panel_order):
            missing = sorted(set(panel_order) - set(panel_programmatic))
            raise RuntimeError(
                "Programmatic evaluation missing panel results: "
                + ", ".join(missing)
            )
        combined_manifest = combine_figure_manifests(
            {
                panel_id: panel_programmatic[panel_id]["figure_manifest"]
                for panel_id in panel_order
            }
        )
        cohesion = evaluate_cohesion(
            combined_manifest,
            full_expectation,
            metric_config,
        )
        programmatic_summary = {
            "metric_config": panel_programmatic[panel_order[0]][
                "metric_config"
            ],
            "figure_manifest": combined_manifest.to_dict(),
            "panel_fidelity": {
                panel_id: panel_programmatic[panel_id]["fidelity"]
                for panel_id in panel_order
            },
            "cohesion": cohesion.to_dict(),
        }
        programmatic_path = active_output_dir / "programmatic_evaluation.json"
        programmatic_path.write_text(
            json.dumps(
                programmatic_summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    result = {
        "panel_group": panel_group,
        "rounds": configured_rounds,
        "initial_generation": configured_initial_generation,
        "seed": configured_seed,
        "temperature": configured_temperature,
        "memory_mode": configured_memory_mode,
        "panel_order": panel_order,
        "panels": panel_results,
        "render_counts": render_counts,
        "schedule_trace_path": str(schedule_path),
        "shared_memory_snapshot_path": str(memory_snapshot_path),
        "shared_memory_trace_path": str(memory_trace_path),
        "pheromones_path": str(compatibility_path),
        "untyped_memory_path": (
            str(untyped_memory_path) if untyped_memory_path is not None else None
        ),
        "combined_figure_path": str(combined_image_path),
        "programmatic_evaluation": programmatic_summary,
        "programmatic_evaluation_path": (
            str(programmatic_path) if programmatic_path is not None else None
        ),
        "output_dir": str(active_output_dir),
    }
    result_path = active_output_dir / "multi_panel_result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PheroViz panels with deterministic round-robin scheduling."
    )
    parser.add_argument("manifest", help="JSON manifest containing a panels list")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument(
        "--initial-generation",
        choices=("defaults", "model_spec", "model"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--memory-mode",
        choices=("none", "ephemeral", "untyped", "constraints", "patches", "full"),
        default=None,
    )
    args = parser.parse_args()
    result = run_multi_panel(
        args.manifest,
        output_dir=args.output_dir,
        rounds=args.rounds,
        initial_generation=args.initial_generation,
        seed=args.seed,
        temperature=args.temperature,
        memory_mode=args.memory_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
