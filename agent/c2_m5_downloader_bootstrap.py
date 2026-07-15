"""Run the retained frozen downloader under an explicit guarded import closure."""

from __future__ import annotations

import sys

if (
    not sys.flags.isolated
    or not sys.flags.no_site
    or not sys.flags.dont_write_bytecode
):
    print(
        "M5 downloader bootstrap refused: invoke Python with -I -S -B",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> int:
    import builtins
    import importlib.util
    import itertools
    import multiprocessing
    from pathlib import Path

    workspace = Path(__file__).resolve().parent
    sys.path.insert(0, str(workspace / "dependencies"))
    guard_path = workspace / "network_guard" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("sitecustomize", guard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("M5 downloader network guard is unavailable")
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    original_zip = builtins.zip

    def compatible_zip(*iterables, strict=False):
        if not strict:
            return original_zip(*iterables)

        def strict_iterator():
            sentinel = object()
            for values in itertools.zip_longest(*iterables, fillvalue=sentinel):
                if any(value is sentinel for value in values):
                    raise ValueError("zip() arguments have unequal lengths")
                yield values

        return strict_iterator()

    builtins.zip = compatible_zip
    sys.path.insert(0, str(workspace))
    multiprocessing.set_start_method("fork", force=True)
    downloader_path = workspace / "frozen_downloader.py"
    payload = downloader_path.read_bytes()
    sys.argv = [str(downloader_path), *sys.argv[1:]]
    namespace = globals()
    namespace["__file__"] = str(downloader_path)
    namespace["__package__"] = None
    exec(compile(payload, str(downloader_path), "exec"), namespace)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"M5 downloader bootstrap refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
