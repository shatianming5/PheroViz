from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .models import MetricConfig


SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


@lru_cache(maxsize=None)
def load_schema(name: str) -> Dict[str, Any]:
    path = SCHEMA_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against_schema(instance: Any, schema_name: str) -> None:
    validator = Draft202012Validator(load_schema(schema_name))
    validator.validate(instance)


def validate_expectation(expectation: Any) -> None:
    validate_against_schema(expectation, "expectation.schema.json")
    panel_ids: set[str] = set()
    series_by_panel: Dict[str, set[str]] = {}
    for panel in expectation["panels"]:
        panel_id = panel["panel_id"]
        if panel_id in panel_ids:
            raise ValidationError(f"Duplicate panel_id '{panel_id}'")
        panel_ids.add(panel_id)
        series_ids: set[str] = set()
        for series in panel["series"]:
            series_id = series["series_id"]
            if series_id in series_ids:
                raise ValidationError(
                    f"Duplicate series_id '{series_id}' in panel '{panel_id}'"
                )
            series_ids.add(series_id)
        series_by_panel[panel_id] = series_ids

    group_ids: set[str] = set()
    for group in expectation.get("panel_groups") or []:
        group_id = group["group_id"]
        if group_id in group_ids:
            raise ValidationError(f"Duplicate group_id '{group_id}'")
        group_ids.add(group_id)
        for panel_id in group["panels"]:
            if panel_id not in panel_ids:
                raise ValidationError(
                    f"Panel group '{group_id}' references unknown panel_id '{panel_id}'"
                )
        for series_id in group.get("series") or []:
            missing = [
                panel_id
                for panel_id in group["panels"]
                if series_id not in series_by_panel[panel_id]
            ]
            if missing:
                raise ValidationError(
                    f"Panel group '{group_id}' references series_id '{series_id}' "
                    f"missing from panels {missing}"
                )


def validate_metric_config_input(config: Any) -> None:
    validate_against_schema(config, "metric_config_input.schema.json")


def validate_metric_config(config: Any) -> None:
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    validate_against_schema(config, "metric_config.schema.json")


def coerce_metric_config(config: Any = None) -> MetricConfig:
    if config is None:
        resolved = MetricConfig()
    elif isinstance(config, MetricConfig):
        resolved = config
    elif isinstance(config, Mapping):
        validate_metric_config_input(config)
        resolved = MetricConfig.from_dict(config)
    else:
        raise TypeError("config must be a MetricConfig, mapping, or None")
    validate_metric_config(resolved)
    return resolved


def validate_figure_manifest(manifest: Any) -> None:
    if hasattr(manifest, "to_dict"):
        manifest = manifest.to_dict()
    validate_against_schema(manifest, "figure_manifest.schema.json")


def validate_evaluation_result(result: Any) -> None:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    validate_metric_config(result["metric_config"])
    validate_figure_manifest(result["figure_manifest"])
    validate_against_schema(result, "evaluation_result.schema.json")
