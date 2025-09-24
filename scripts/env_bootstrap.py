#!/usr/bin/env python3
"""
Lightweight environment probe and runner:
- Prints current Python and (if available) conda environment info
- Delegates to scripts/nature_cli.py, installing minimal deps if needed
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def print_env_info():
    print(f"Python: {sys.version.split()[0]} | exe: {sys.executable}")
    conda_exe = shutil.which("conda")
    if conda_exe:
        try:
            out = subprocess.check_output([conda_exe, "info", "--json"], text=True)
            data = json.loads(out)
            env_name = data.get("active_prefix_name") or os.environ.get("CONDA_DEFAULT_ENV")
            print(f"Conda: detected | active env: {env_name}")
        except Exception as e:
            print(f"Conda: detected but could not query info ({e})")
    else:
        print("Conda: not detected in PATH; proceeding with current Python env")


def main():
    print_env_info()
    # delegate to CLI with same args
    script = Path(__file__).parent / "nature_cli.py"
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

