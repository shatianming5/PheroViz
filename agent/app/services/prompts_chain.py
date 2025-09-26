from __future__ import annotations

SYSTEM_COMMON = (
    "?????????????????? JSON ?????????"
    "????? JSON,????????????,?????????? JSON?"
)

USER_L1 = """??:
{{profile_json}}

??:{{user_goal}}
??????:{{chart_family}}

{{feedback}}

??? ChartPlan:
{{schema}}
"""

USER_L2 = """???? ChartPlan,???????????:
ChartPlan:
{{chart_plan_json}}

??? Orchestration:
{{schema}}
"""

USER_L3 = """?? ChartPlan + Orchestration,????/??/?????:
ChartPlan:
{{chart_plan_json}}
Orchestration:
{{orchestration_json}}

??? Calibration:
{{schema}}
"""

USER_L4 = """??????,??????????:
ChartPlan:
{{chart_plan_json}}
Orchestration:
{{orchestration_json}}
Calibration:
{{calibration_json}}

??? Refinement:
{{schema}}
"""

SCHEMA_L1 = """{
  "table": "string",
  "chart_family": "line|bar|stacked_bar|area|pie|scatter|hist|box|heatmap",
  "dimensions": ["string"],
  "measures": ["string"],
  "time_granularity": "auto|day|week|month|quarter|year",
  "constraints": {
    "time_range": ["YYYY-MM-DD","YYYY-MM-DD"],
    "filters": [{"column":"...", "op":"==|!=|>|<|>=|<=|in|not in|between|contains", "value": "..."}],
    "topk": {"by":"...", "k":5, "per":"..."}
  }
}"""

SCHEMA_L2 = """{
  "transforms": [
    {"op":"filter","expr":{"column":"...","op":"...","value":"..."}},
    {"op":"derive","as":"...","expr":"to_period(??,'M')"},
    {"op":"topk","group_by":["..."],"order_by":{"col":"...","desc":true},"k":5},
    {"op":"aggregate","group_by":["..."],"measures":[{"col":"...","agg":"sum","as":"..."}]},
    {"op":"sort","by":[{"col":"...","asc":true}]}
  ],
  "encoding_plan":{"x":"...","y":"...","series":"...|null"}
}"""

SCHEMA_L3 = """{
  "scales":{"x":{"type":"temporal","timeUnit":"yearmonth"},"y":{"type":"linear","nice":true}},
  "bins": 20,
  "stack": null,
  "order":{"x":"ascending"},
  "units":{"y":"??"},
  "legend": true
}"""

SCHEMA_L4 = """{
  "title":"...",
  "xlabel":"...",
  "ylabel":"...",
  "labels":{"show":false},
  "tick_density":"auto|sparse|dense",
  "theme":{"fontSize":10,"padding":4}
}"""

FEEDBACK_TEMPLATE = """
??????,????????????:
{{feedback}}
"""
