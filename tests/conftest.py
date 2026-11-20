import os
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No CVAT_* variables from the shell and no .env: each test sets what it needs."""
    for name in [n for n in os.environ if n.startswith("CVAT_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    # subprocess tests must import this checkout's code, not whatever is installed
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")]))
    )
