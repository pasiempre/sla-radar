# tests/conftest.py
import os
import sys
from pathlib import Path

import pytest


# Ensure project root (sla-radar/) is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def db_path(project_root: Path) -> Path:
    """
    Points at data/warehouse.db. If it's missing, tests that depend
    on it will be skipped instead of hard-failing.
    """
    db = project_root / "data" / "warehouse.db"
    if not db.exists():
        pytest.skip(f"warehouse.db not found at {db}")
    return db


@pytest.fixture
def db_conn(db_path: Path):
    """
    Simple sqlite3 connection fixture for DB-related tests.
    """
    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        yield con
    finally:
        con.close()