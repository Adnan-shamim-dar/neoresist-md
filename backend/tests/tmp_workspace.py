from __future__ import annotations

import shutil
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _root_tmp_dir() -> Path:
    root = _repo_root() / "local_state" / "test_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def temp_workspace(prefix: str = "tmp"):
    """
    Stable per-test temporary workspace under repo-local ``local_state/test_tmp``.

    We avoid ``tempfile.TemporaryDirectory`` here because some Windows environments
    intermittently deny writes/cleanup under the OS temp directory during full-suite
    discovery runs.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = _root_tmp_dir() / f"{prefix}_{stamp}_{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

