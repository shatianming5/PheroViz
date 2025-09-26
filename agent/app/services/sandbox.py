from __future__ import annotations

import ast
import builtins
import math
from io import BytesIO
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Safe backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SandboxError(RuntimeError):
    """Base sandbox execution error."""


class UnsafeCodeError(SandboxError):
    """Raised when code fails safety checks."""


ALLOWED_IMPORTS = {"math", "numpy", "pandas", "matplotlib", "matplotlib.pyplot", "io"}
DISALLOWED_CALLS = {"__import__", "eval", "exec", "open", "compile", "globals", "locals", "input"}
DISALLOWED_ATTRIBUTES_PREFIX = "__"


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = (alias.name or "").split(".")[0]
                if module not in ALLOWED_IMPORTS:
                    raise UnsafeCodeError(f"Import of module '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(f"Import of module '{node.module}' is not allowed.")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DISALLOWED_CALLS:
                raise UnsafeCodeError(f"Call to '{func.id}' is not permitted in sandboxed code.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith(DISALLOWED_ATTRIBUTES_PREFIX):
                raise UnsafeCodeError("Access to dunder attributes is not permitted.")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
            raise UnsafeCodeError("Defining classes or async functions is not supported in sandbox.")


def _safe_builtins() -> Dict[str, Any]:
    allowed = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "sum",
        "zip",
    }
    return {name: getattr(builtins, name) for name in allowed}


def run_render_chart(code: str, dataframe: pd.DataFrame) -> bytes:
    """Execute generated code safely and return PNG bytes."""

    tree = ast.parse(code, mode="exec")
    _validate_ast(tree)

    safe_globals: Dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "BytesIO": BytesIO,
        "math": math,
    }

    exec(compile(tree, "<chart_code>", "exec"), safe_globals, safe_globals)

    if "generate_chart" not in safe_globals or not callable(safe_globals["generate_chart"]):
        raise SandboxError("Generated code must define a callable 'generate_chart(dataset)'.")

    try:
        result = safe_globals["generate_chart"](dataframe)
    finally:
        plt.close("all")

    if not isinstance(result, (bytes, bytearray)):
        raise SandboxError("generate_chart must return image bytes.")

    return bytes(result)
