from __future__ import annotations

from pathlib import Path
import shutil

import pytest


@pytest.fixture
def workdir() -> Path:
    path = Path(__file__).parent / "_runtime"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
