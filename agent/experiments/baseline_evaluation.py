from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import os
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


_ALLOWED_IMPORT_MODULES = {
    "datetime",
    "duckdb",
    "math",
    "matplotlib",
    "matplotlib.cm",
    "matplotlib.colors",
    "matplotlib.dates",
    "matplotlib.lines",
    "matplotlib.patches",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "numpy",
    "os",
    "os.path",
    "pandas",
    "seaborn",
    "statistics",
}
_ALLOWED_MODULE_GRAPH = _ALLOWED_IMPORT_MODULES | {
    "numpy.linalg",
    "numpy.ma",
    "numpy.random",
    "pandas.api",
    "pandas.api.types",
    "pandas.plotting",
}
_ALLOWED_IMPORTS = {
    module.partition(".")[0] for module in _ALLOWED_IMPORT_MODULES
}
_ALLOWED_BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "format",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_FORBIDDEN_ATTRIBUTES = {
    "apply",
    "chmod",
    "chown",
    "cdll",
    "check_call",
    "check_output",
    "communicate",
    "compile",
    "ctypes",
    "dump",
    "dumps",
    "datasource",
    "eval",
    "exec",
    "excelfile",
    "fdopen",
    "fork",
    "fromfile",
    "genfromtxt",
    "get_dataset_names",
    "get_sample_data",
    "getattr",
    "glob",
    "iglob",
    "hdfstore",
    "imread",
    "imsave",
    "importlib",
    "kill",
    "load",
    "load_dataset",
    "load_library",
    "loads",
    "loadtxt",
    "makedirs",
    "memmap",
    "mkdir",
    "open",
    "open_memmap",
    "os",
    "pathlib",
    "pydll",
    "popen",
    "putenv",
    "query",
    "rc_context",
    "rc_file",
    "rc_params_from_file",
    "read_feather",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_stata",
    "read_xml",
    "recv",
    "remove",
    "removedirs",
    "rename",
    "renames",
    "request",
    "requests",
    "run",
    "rmdir",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "scandir",
    "send",
    "setattr",
    "shutil",
    "socket",
    "spawn",
    "subprocess",
    "switch_backend",
    "symlink",
    "sys",
    "system",
    "tempfile",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_gbq",
    "to_hdf",
    "to_html",
    "to_json",
    "to_latex",
    "to_markdown",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
    "unlink",
    "urllib",
    "urlopen",
    "urlretrieve",
    "walk",
    "windll",
    "write",
    "write_bytes",
    "write_text",
    "writelines",
}
_ALLOWED_OS_ATTRIBUTES = {
    "listdir",
    "path",
}
_ALLOWED_OS_PATH_ATTRIBUTES = {
    "basename",
    "dirname",
    "join",
    "split",
    "splitext",
}
_PATH_SUFFIXES = {
    ".csv",
    ".json",
    ".png",
    ".svg",
    ".tsv",
    ".xls",
    ".xlsm",
    ".xlsx",
}
_REJECTED_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


class BaselineEvaluationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        artifacts: Optional[Mapping[str, Path]] = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.artifacts = dict(artifacts or {})
        self.timed_out = timed_out


class StaticCodeError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineEvaluationOutcome:
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    metadata: Dict[str, Any]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contained(path: Path, root: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise BaselineEvaluationError(
            f"Programmatic-evaluation path cannot be a symlink: {path}"
        )
    try:
        resolved = expanded.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise BaselineEvaluationError(
            f"Programmatic-evaluation path escaped its work directory: {path}"
        ) from exc
    return resolved


def _attribute_parts(node: ast.AST) -> list[str]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return list(reversed(parts))


class _StaticValidator(ast.NodeVisitor):
    def __init__(self, *, allowed_root: Path, execution_cwd: Path) -> None:
        self.allowed_root = allowed_root.resolve()
        self.execution_cwd = execution_cwd.resolve()
        self.imported_callables: set[str] = set()

    def _reject(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        raise StaticCodeError(f"line {line}: {message}")

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, _REJECTED_NODES):
            self._reject(
                node,
                f"unsupported statement {type(node).__name__}",
            )
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.partition(".")[0]
            if (
                root not in _ALLOWED_IMPORTS
                or alias.name not in _ALLOWED_IMPORT_MODULES
            ):
                self._reject(node, f"import {alias.name!r} is not allowed")
            if alias.asname and alias.asname.startswith("__"):
                self._reject(node, "dunder import aliases are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._reject(node, "relative imports are not allowed")
        module = node.module or ""
        root = module.partition(".")[0]
        if (
            root not in _ALLOWED_IMPORTS
            or module not in _ALLOWED_IMPORT_MODULES
        ):
            self._reject(node, f"import from {module!r} is not allowed")
        for alias in node.names:
            local_name = alias.asname or alias.name
            if alias.name == "*" or alias.name.startswith("_"):
                self._reject(node, "wildcard/private imports are not allowed")
            if alias.name in _FORBIDDEN_ATTRIBUTES:
                self._reject(node, f"imported callable {alias.name!r} is unsafe")
            self.imported_callables.add(local_name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("_") or node.decorator_list:
            self._reject(
                node,
                "private/decorated helper functions are not allowed",
            )
        self.imported_callables.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self._reject(node, "dunder names are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts = _attribute_parts(node)
        if node.attr.startswith("_"):
            self._reject(node, "private/dunder attributes are not allowed")
        attribute = node.attr.casefold()
        if (
            attribute in _FORBIDDEN_ATTRIBUTES
            or (
                attribute.startswith("read_")
                and attribute not in {"read_csv", "read_excel", "read_table"}
            )
            or attribute.startswith("open_")
            or attribute.startswith("spawn")
            or attribute.startswith("to_")
            or attribute.startswith("write_")
            or attribute.startswith("print_")
        ):
            qualified = ".".join(parts) if parts else node.attr
            self._reject(node, f"attribute {qualified!r} is not allowed")
        if attribute == "connect" and (not parts or parts[0] != "duckdb"):
            self._reject(node, "only duckdb.connect is allowed")
        if parts and parts[0] == "os":
            if len(parts) == 2 and parts[1] not in _ALLOWED_OS_ATTRIBUTES:
                self._reject(node, f"os.{parts[1]} is not allowed")
            if (
                len(parts) >= 3
                and parts[1] == "path"
                and parts[2] not in _ALLOWED_OS_PATH_ATTRIBUTES
            ):
                self._reject(node, f"os.path.{parts[2]} is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if (
                name not in _ALLOWED_BUILTIN_CALLS
                and name not in self.imported_callables
            ):
                self._reject(node, f"call to {name!r} is not allowed")
        elif not isinstance(node.func, ast.Attribute):
            self._reject(node, "dynamic call targets are not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        value = node.value
        lowered = value.strip().lower()
        if lowered.startswith(("http://", "https://", "ftp://", "file://")):
            self._reject(node, "network/file URLs are not allowed")
        if "\x00" in value:
            self._reject(node, "NUL bytes are not allowed")
        path = Path(value)
        looks_like_path = (
            path.is_absolute()
            or bool(re.match(r"^[A-Za-z]:[\\/]", value))
            or ".." in path.parts
            or path.suffix.lower() in _PATH_SUFFIXES
        )
        if not looks_like_path:
            return
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (self.execution_cwd / path).resolve()
        )
        try:
            resolved.relative_to(self.allowed_root)
        except ValueError:
            self._reject(node, f"path literal escapes work directory: {value!r}")


def validate_generated_code(
    source: str,
    *,
    allowed_root: Path,
    execution_cwd: Path,
) -> None:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise StaticCodeError(f"generated code is not valid Python: {exc}") from exc
    _StaticValidator(
        allowed_root=allowed_root,
        execution_cwd=execution_cwd,
    ).visit(tree)


def run_programmatic_evaluation(
    *,
    python_executable: str,
    work_dir: Path,
    code_path: Path,
    source_path: Path,
    sheet: str | int | None,
    expectation: Mapping[str, Any],
    metric_config: Optional[Mapping[str, Any]],
    environment: Mapping[str, str],
    timeout_seconds: float,
    alias_path: Optional[Path] = None,
) -> BaselineEvaluationOutcome:
    root = work_dir.resolve()
    code = _contained(code_path, root)
    execution_cwd = code.parent
    source_input = source_path.expanduser()
    if source_input.is_symlink():
        raise BaselineEvaluationError(
            f"Programmatic-evaluation source cannot be a symlink: {source_input}"
        )
    try:
        source = source_input.resolve(strict=True)
    except OSError as exc:
        raise BaselineEvaluationError(
            f"Programmatic-evaluation source does not exist: {source_input}"
        ) from exc
    if not source.is_file():
        raise BaselineEvaluationError(
            f"Programmatic-evaluation source is not a regular file: {source}"
        )
    aliases = _contained(alias_path, root) if alias_path is not None else None

    input_path = root / "programmatic_evaluation_input.json"
    output_path = root / "programmatic_evaluation.json"
    render_path = root / "programmatic_evaluation_render.png"
    validation_path = root / "programmatic_evaluation_validation.json"
    stdout_path = root / "programmatic_evaluation.stdout.log"
    stderr_path = root / "programmatic_evaluation.stderr.log"
    subprocess_path = root / "programmatic_evaluation_subprocess.json"
    for owned_path in (
        input_path,
        output_path,
        render_path,
        validation_path,
        stdout_path,
        stderr_path,
        subprocess_path,
    ):
        if owned_path.exists():
            raise BaselineEvaluationError(
                "Programmatic-evaluation artifact already exists: "
                f"{owned_path.name}"
            )
    owned_artifacts: Dict[str, Path] = {
        "programmatic_evaluation_input": input_path,
        "programmatic_evaluation_validation": validation_path,
        "programmatic_evaluation_stdout": stdout_path,
        "programmatic_evaluation_stderr": stderr_path,
        "programmatic_evaluation_subprocess": subprocess_path,
    }

    code_bytes = code.read_bytes()
    code_sha256 = _sha256_bytes(code_bytes)
    try:
        code_text = code_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        _write_json(
            validation_path,
            {
                "schema_version": "1.0",
                "status": "rejected",
                "code_sha256": code_sha256,
                "error": "generated code is not UTF-8",
            },
        )
        raise BaselineEvaluationError(
            "Generated baseline code is not UTF-8",
            artifacts={"programmatic_evaluation_validation": validation_path},
        ) from exc

    try:
        validate_generated_code(
            code_text,
            allowed_root=root,
            execution_cwd=execution_cwd,
        )
    except StaticCodeError as exc:
        _write_json(
            validation_path,
            {
                "schema_version": "1.0",
                "status": "rejected",
                "code_sha256": code_sha256,
                "error": str(exc),
            },
        )
        raise BaselineEvaluationError(
            f"Generated baseline code rejected by static policy: {exc}",
            artifacts={"programmatic_evaluation_validation": validation_path},
        ) from exc

    _write_json(
        validation_path,
        {
            "schema_version": "1.0",
            "status": "accepted",
            "code_sha256": code_sha256,
        },
    )
    _write_json(
        input_path,
        {
            "schema_version": "1.0",
            "agent_root": str(Path(__file__).resolve().parents[1]),
            "allowed_root": str(root),
            "execution_cwd": str(execution_cwd),
            "code_path": str(code),
            "code_sha256": code_sha256,
            "source_path": str(source),
            "sheet": sheet,
            "expectation": dict(expectation),
            "metric_config": (
                dict(metric_config) if metric_config is not None else None
            ),
            "alias_path": str(aliases) if aliases is not None else None,
            "output_path": str(output_path),
            "render_path": str(render_path),
        },
    )

    timed_out = timeout_seconds <= 0
    stdout = ""
    stderr = ""
    exit_code = -1
    if timed_out:
        stderr = "Programmatic-evaluation deadline expired before launch"
    else:
        try:
            completed = subprocess.run(
                [
                    python_executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(input_path),
                ],
                cwd=execution_cwd,
                env=dict(environment),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            stderr = (
                f"{stderr}\nProgrammatic evaluation timed out after "
                f"{timeout_seconds}s"
            )

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    _write_json(
        subprocess_path,
        {
            "schema_version": "1.0",
            "python_executable": python_executable,
            "code_sha256": code_sha256,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "timeout_seconds": timeout_seconds,
            "stdout_summary": stdout[:2000],
            "stderr_summary": stderr[:2000],
        },
    )
    if exit_code != 0 or not output_path.is_file() or not render_path.is_file():
        if output_path.is_file():
            owned_artifacts["programmatic_evaluation"] = output_path
        if render_path.is_file():
            owned_artifacts["programmatic_evaluation_render"] = render_path
        detail = ""
        stderr_lines = [
            line.strip() for line in stderr.splitlines() if line.strip()
        ]
        if stderr_lines:
            detail = f": {stderr_lines[-1][:500]}"
        raise BaselineEvaluationError(
            "Programmatic evaluation failed"
            + (f" with exit code {exit_code}" if exit_code != 0 else "")
            + detail,
            artifacts=owned_artifacts,
            timed_out=timed_out,
        )

    owned_artifacts["programmatic_evaluation"] = output_path
    owned_artifacts["programmatic_evaluation_render"] = render_path
    render_bytes = render_path.read_bytes()
    if not render_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise BaselineEvaluationError(
            "Programmatic evaluation render is not a PNG",
            artifacts=owned_artifacts,
        )
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineEvaluationError(
            f"Programmatic evaluation produced invalid JSON: {exc}",
            artifacts=owned_artifacts,
        ) from exc
    if not isinstance(payload, Mapping):
        raise BaselineEvaluationError(
            "Programmatic evaluation result must be an object",
            artifacts=owned_artifacts,
        )

    metrics: Dict[str, float] = {}
    for result_name, metric_name in (
        ("fidelity", "data_fidelity"),
        ("cohesion", "series_cohesion"),
    ):
        result = payload.get(result_name)
        if not isinstance(result, Mapping) or result.get("applicable") is not True:
            continue
        ratio = result.get("ratio")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not 0.0 <= float(ratio) <= 1.0
        ):
            raise BaselineEvaluationError(
                f"Programmatic {result_name} has no valid applicable ratio",
                artifacts=owned_artifacts,
            )
        metrics[metric_name] = float(ratio)

    metadata = {
        "code_sha256": code_sha256,
        "fidelity_applicable": "data_fidelity" in metrics,
        "cohesion_applicable": "series_cohesion" in metrics,
        "alias_reversal": aliases is not None,
        "render_sha256": _sha256_bytes(render_bytes),
    }
    return BaselineEvaluationOutcome(
        metrics=metrics,
        artifacts={
            label: str(path) for label, path in owned_artifacts.items()
        },
        metadata=metadata,
    )


def _load_source(path: Path, sheet: str | int | None):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    raise RuntimeError(f"Unsupported evaluation source format: {suffix}")


def _reverse_aliases(fig: Any, alias_path: Optional[Path]) -> int:
    if alias_path is None:
        return 0
    value = json.loads(alias_path.read_text(encoding="utf-8"))
    tables = value.get("tables") if isinstance(value, Mapping) else None
    if not isinstance(tables, list):
        raise RuntimeError("nvAgent alias artifact has no tables array")
    reverse: Dict[str, str] = {}
    for table in tables:
        if not isinstance(table, Mapping):
            raise RuntimeError("nvAgent alias table entry is invalid")
        candidates: Dict[str, str] = {}
        source_name = table.get("source_name")
        table_alias = table.get("table_alias")
        if isinstance(source_name, str) and isinstance(table_alias, str):
            candidates[table_alias] = Path(source_name).stem
        column_aliases = table.get("column_aliases")
        if not isinstance(column_aliases, Mapping):
            raise RuntimeError("nvAgent alias table has no column_aliases")
        for original, alias in column_aliases.items():
            if not isinstance(original, str) or not isinstance(alias, str):
                raise RuntimeError("nvAgent column alias must be a string")
            candidates[alias] = original
        for alias, original in candidates.items():
            previous = reverse.get(alias)
            if previous is not None and previous != original:
                raise RuntimeError(
                    f"nvAgent alias {alias!r} maps to multiple original labels"
                )
            reverse[alias] = original

    replacements = 0
    for artist in fig.findobj():
        get_text = getattr(artist, "get_text", None)
        set_text = getattr(artist, "set_text", None)
        if callable(get_text) and callable(set_text):
            text = get_text()
            if isinstance(text, str) and text in reverse:
                set_text(reverse[text])
                replacements += 1
        get_label = getattr(artist, "get_label", None)
        set_label = getattr(artist, "set_label", None)
        if callable(get_label) and callable(set_label):
            label = get_label()
            if isinstance(label, str) and label in reverse:
                set_label(reverse[label])
                replacements += 1
    return replacements


class _SafeOSPath:
    basename = staticmethod(os.path.basename)
    dirname = staticmethod(os.path.dirname)
    join = staticmethod(os.path.join)
    split = staticmethod(os.path.split)
    splitext = staticmethod(os.path.splitext)


class _SafeOS:
    path = _SafeOSPath()

    def __init__(self, *, allowed_root: Path, execution_cwd: Path) -> None:
        self.allowed_root = allowed_root
        self.execution_cwd = execution_cwd

    def listdir(self, path: Any = ".") -> list[str]:
        resolved = _guard_read_path(
            path,
            allowed_root=self.allowed_root,
            execution_cwd=self.execution_cwd,
            require_file=False,
        )
        return os.listdir(resolved)


def _guard_read_path(
    value: Any,
    *,
    allowed_root: Path,
    execution_cwd: Path,
    require_file: bool = True,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise RuntimeError("Generated code may only read filesystem paths")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = execution_cwd / path
    if path.is_symlink():
        raise RuntimeError("Generated code cannot read symlinks")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Generated code attempted to read outside its work directory: "
            f"{resolved}"
        ) from exc
    if require_file and not resolved.is_file():
        raise RuntimeError(f"Generated code read target is not a file: {resolved}")
    if not require_file and not resolved.is_dir():
        raise RuntimeError(
            f"Generated code list target is not a directory: {resolved}"
        )
    return resolved


def _validate_duckdb_sql(
    query: Any,
    *,
    allowed_root: Path,
    execution_cwd: Path,
) -> str:
    if not isinstance(query, str):
        raise RuntimeError("DuckDB queries must be static strings")
    normalized = re.sub(r"\s+", " ", query.strip())
    without_trailing = normalized[:-1] if normalized.endswith(";") else normalized
    if ";" in without_trailing:
        raise RuntimeError("Multiple DuckDB statements are not allowed")
    upper = without_trailing.upper()
    forbidden = (
        "ATTACH",
        "CALL",
        "COPY",
        "DELETE",
        "DETACH",
        "DROP",
        "EXPORT",
        "IMPORT",
        "INSERT",
        "INSTALL",
        "LOAD",
        "PRAGMA",
        "READ_JSON",
        "READ_PARQUET",
        "SECRET",
        "SHELL",
        "UPDATE",
    )
    if any(re.search(rf"\b{token}\b", upper) for token in forbidden):
        raise RuntimeError("DuckDB query contains a forbidden operation")
    if re.search(
        r"\b(?:GLOB|HTTP_GET|PARQUET_SCAN|POSTGRES_SCAN|SQLITE_SCAN)\s*\(",
        upper,
    ):
        raise RuntimeError("DuckDB query contains a forbidden table function")
    is_select = upper.startswith(("SELECT ", "WITH "))
    is_csv_view = bool(
        re.match(
            r'^CREATE VIEW "?[A-Za-z0-9_]+"?\s+AS SELECT \* FROM '
            r"READ_CSV_AUTO\(.+\)$",
            without_trailing,
            flags=re.IGNORECASE,
        )
    )
    if not is_select and not is_csv_view:
        raise RuntimeError("Only read-only SELECT and CSV views are allowed")
    paths = re.findall(
        r"read_csv_auto\(\s*'([^']+)'\s*\)",
        query,
        flags=re.IGNORECASE,
    )
    if is_csv_view and len(paths) != 1:
        raise RuntimeError("CSV view must contain exactly one literal path")
    if re.search(r"\bREAD_[A-Z0-9_]+\s*\(", upper) and not paths:
        raise RuntimeError("Only read_csv_auto is allowed in DuckDB queries")
    for path in paths:
        _guard_read_path(
            path,
            allowed_root=allowed_root,
            execution_cwd=execution_cwd,
        )
    return query


def _worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["allowed_root"]).resolve(strict=True)
    execution_cwd = Path(config["execution_cwd"]).resolve(strict=True)
    code_path = _contained(Path(config["code_path"]), root)
    output_path = Path(config["output_path"]).resolve()
    output_path.relative_to(root)
    render_path = Path(config["render_path"]).resolve()
    render_path.relative_to(root)
    source_path = Path(config["source_path"]).resolve(strict=True)
    alias_path = (
        _contained(Path(config["alias_path"]), root)
        if config.get("alias_path")
        else None
    )

    code_bytes = code_path.read_bytes()
    if _sha256_bytes(code_bytes) != config["code_sha256"]:
        raise RuntimeError("Generated code changed after static validation")
    code = code_bytes.decode("utf-8")
    validate_generated_code(
        code,
        allowed_root=root,
        execution_cwd=execution_cwd,
    )
    expectation = config.get("expectation")
    panels = expectation.get("panels") if isinstance(expectation, Mapping) else None
    if not isinstance(panels, list) or len(panels) != 1:
        raise RuntimeError(
            "External baseline programmatic evaluation requires exactly one panel"
        )
    source_df = _load_source(source_path, config.get("sheet"))

    agent_root = Path(config["agent_root"]).resolve(strict=True)
    sys.path.insert(0, str(agent_root))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure
    real_figure_savefig = Figure.savefig

    original_read_csv = pd.read_csv
    original_read_excel = pd.read_excel
    original_read_table = pd.read_table

    def guarded_read_csv(path: Any, *args: Any, **kwargs: Any):
        return original_read_csv(
            _guard_read_path(
                path,
                allowed_root=root,
                execution_cwd=execution_cwd,
            ),
            *args,
            **kwargs,
        )

    def guarded_read_excel(path: Any, *args: Any, **kwargs: Any):
        return original_read_excel(
            _guard_read_path(
                path,
                allowed_root=root,
                execution_cwd=execution_cwd,
            ),
            *args,
            **kwargs,
        )

    def guarded_read_table(path: Any, *args: Any, **kwargs: Any):
        return original_read_table(
            _guard_read_path(
                path,
                allowed_root=root,
                execution_cwd=execution_cwd,
            ),
            *args,
            **kwargs,
        )

    pd.read_csv = guarded_read_csv
    pd.read_excel = guarded_read_excel
    pd.read_table = guarded_read_table

    real_import = builtins.__import__
    safe_os = _SafeOS(
        allowed_root=root,
        execution_cwd=execution_cwd,
    )
    import numpy as np

    class SafeRcParams:
        _blocked = {
            "backend",
            "backend_fallback",
            "text.usetex",
        }

        def __init__(self, values: Any) -> None:
            self._values = values

        def __getitem__(self, key: str) -> Any:
            return self._values[key]

        def __setitem__(self, key: str, value: Any) -> None:
            normalized = str(key).casefold()
            if (
                normalized in self._blocked
                or normalized.startswith("animation.")
                or normalized.startswith("pgf.")
            ):
                raise RuntimeError(
                    f"Generated code cannot change rcParams[{key!r}]"
                )
            self._values[key] = value

        def get(self, key: str, default: Any = None) -> Any:
            return self._values.get(key, default)

        def update(
            self,
            values: Optional[Mapping[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            pending = dict(values or {})
            pending.update(kwargs)
            for key, value in pending.items():
                self[key] = value

    safe_rc_params = SafeRcParams(matplotlib.rcParams)

    class RestrictedModuleProxy:
        def __init__(
            self,
            module: Any,
            *,
            denied: Sequence[str],
            wrap_module: Any,
        ) -> None:
            self._module = module
            self._denied = set(denied)
            self._wrap_module = wrap_module

        def __getattr__(self, attribute: str) -> Any:
            normalized = attribute.casefold()
            if (
                attribute.startswith("_")
                or normalized in _FORBIDDEN_ATTRIBUTES
                or attribute in self._denied
            ):
                raise AttributeError(
                    f"{self._module.__name__}.{attribute} is not available "
                    "to generated code"
                )
            if self._module.__name__ == "matplotlib" and attribute == "use":
                def use_agg_only(
                    backend: Any,
                    *,
                    force: bool = True,
                ) -> None:
                    if str(backend).casefold() != "agg":
                        raise RuntimeError(
                            "Generated code may only select the Agg backend"
                        )
                    self._module.use("Agg", force=bool(force))

                return use_agg_only
            if (
                self._module.__name__ in {"matplotlib", "matplotlib.pyplot"}
                and attribute == "rcParams"
            ):
                return safe_rc_params
            value = getattr(self._module, attribute)
            if isinstance(value, types.ModuleType):
                return self._wrap_module(value)
            return value

    module_proxies: Dict[str, RestrictedModuleProxy] = {}

    def restricted_module(module: Any) -> RestrictedModuleProxy:
        module_name = str(getattr(module, "__name__", ""))
        if module_name not in _ALLOWED_MODULE_GRAPH:
            raise AttributeError(
                f"module graph access to {module_name!r} is not allowed"
            )
        cached = module_proxies.get(module_name)
        if cached is not None:
            return cached
        denied: Sequence[str] = ()
        if module_name == "pandas":
            denied = ("compat", "io", "testing")
        elif module_name == "numpy":
            denied = ("ctypeslib", "distutils", "f2py", "lib", "testing")
        proxy = RestrictedModuleProxy(
            module,
            denied=denied,
            wrap_module=restricted_module,
        )
        module_proxies[module_name] = proxy
        return proxy

    duckdb_proxy = None

    def safe_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ):
        nonlocal duckdb_proxy
        if level:
            raise ImportError("Relative imports are not allowed")
        module_root = name.partition(".")[0]
        if module_root not in _ALLOWED_IMPORTS:
            raise ImportError(f"Import {name!r} is not allowed")
        if module_root == "os":
            return safe_os
        module = real_import(name, globals, locals, fromlist, level)
        if module_root != "duckdb":
            return restricted_module(module)
        if duckdb_proxy is not None:
            return duckdb_proxy
        real_connect = module.connect

        class GuardedConnection:
            def __init__(self, connection: Any) -> None:
                self._connection = connection

            def execute(
                self,
                query: Any,
                parameters: Any = None,
            ) -> "GuardedConnection":
                safe_query = _validate_duckdb_sql(
                    query,
                    allowed_root=root,
                    execution_cwd=execution_cwd,
                )
                if parameters is not None:
                    raise RuntimeError(
                        "Parameterized DuckDB queries are not supported by "
                        "the static evaluator"
                    )
                self._connection.execute(safe_query)
                return self

            def fetchdf(self):
                return self._connection.fetchdf()

            def fetchall(self):
                return self._connection.fetchall()

            def fetchone(self):
                return self._connection.fetchone()

            def close(self) -> None:
                self._connection.close()

        class DuckDBProxy:
            def __getattr__(self, attribute: str) -> Any:
                if attribute != "connect":
                    raise AttributeError(
                        f"duckdb.{attribute} is not available to generated code"
                    )
                return self.connect

            @staticmethod
            def connect(database: Any = ":memory:", *args: Any, **kwargs: Any):
                if database != ":memory:" or args or kwargs:
                    raise RuntimeError("Generated code may only use in-memory DuckDB")
                return GuardedConnection(real_connect(database=":memory:"))

        duckdb_proxy = DuckDBProxy()
        return duckdb_proxy

    safe_builtins = {
        name: getattr(builtins, name)
        for name in _ALLOWED_BUILTIN_CALLS
        if hasattr(builtins, name)
    }
    safe_builtins.update(
        {
            "__import__": safe_import,
            "ArithmeticError": ArithmeticError,
            "Exception": Exception,
            "RuntimeError": RuntimeError,
            "ValueError": ValueError,
        }
    )

    real_close = plt.close
    real_close("all")

    def no_output(*args: Any, **kwargs: Any) -> None:
        return None

    plt.show = no_output
    plt.close = no_output
    plt.savefig = no_output
    Figure.savefig = no_output
    namespace: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": "__baseline_programmatic_evaluation__",
    }
    exec(compile(code, str(code_path), "exec"), namespace, namespace)

    figures: Dict[int, Any] = {}
    for number in plt.get_fignums():
        figure = plt.figure(number)
        figures[id(figure)] = figure
    for value in namespace.values():
        if isinstance(value, Figure):
            figures[id(value)] = value
    if len(figures) != 1:
        raise RuntimeError(
            "Generated code must leave exactly one live Matplotlib Figure; "
            f"found {len(figures)}"
        )
    figure = next(iter(figures.values()))
    reversed_labels = _reverse_aliases(figure, alias_path)

    from app.evaluation import evaluate_figure

    result = evaluate_figure(
        figure,
        source_df,
        expectation,
        config.get("metric_config"),
    ).to_dict()
    real_figure_savefig(
        figure,
        render_path,
        format="png",
        bbox_inches="tight",
    )
    _write_json(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "alias_labels_reversed": reversed_labels,
            },
            sort_keys=True,
        )
    )
    real_close(figure)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args(argv)
    return _worker(args.worker)


if __name__ == "__main__":
    raise SystemExit(main())
