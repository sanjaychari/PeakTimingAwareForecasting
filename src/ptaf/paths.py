from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the external data directory.

    By default the project expects data in ../data relative to the repository,
    as used by the current MLSys workspace. Set PTAF_DATA_ROOT to override it.
    """
    override = os.environ.get("PTAF_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repository_root().parent / "data"


def results_root() -> Path:
    return repository_root() / "results"
