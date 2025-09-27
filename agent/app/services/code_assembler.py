from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict

from jinja2 import Template

from .slot_registry import slot_to_jinja_var

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "runtime" / "scaffold_elements_pro.py.j2"

_FORBIDDEN_CALLS = {"eval", "exec", "__import__", "open"}
_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)


def _safety_ast_check(py_code: str) -> None:
    tree = ast.parse(py_code)

    def _inspect(node: ast.AST, allow_imports: bool = False) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FORBIDDEN_NODES):
                if allow_imports:
                    continue
                raise ValueError("Forbidden import used in assembled scaffold.")
            if isinstance(child, ast.Call):
                func_name = ""
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    func_name = child.func.attr
                if func_name in _FORBIDDEN_CALLS:
                    raise ValueError(f"Forbidden call `{func_name}` used in scaffold.")
            _inspect(child, False)

    for top in tree.body:
        if isinstance(top, ast.Call):
            func_name = ""
            if isinstance(top.func, ast.Name):
                func_name = top.func.id
            elif isinstance(top.func, ast.Attribute):
                func_name = top.func.attr
            if func_name in _FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden call `{func_name}` used in scaffold.")
        allow_imports = isinstance(top, _FORBIDDEN_NODES)
        _inspect(top, allow_imports=allow_imports)


def assemble_with_slots(slot_code_map: Dict[str, str]) -> str:
    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
    template = Template(template_text)
    context: Dict[str, str] = {}
    for key, value in (slot_code_map or {}).items():
        jinja_var = slot_to_jinja_var(key)
        context[jinja_var] = value.strip()
    rendered = template.render(**context)
    _safety_ast_check(rendered)
    return rendered
