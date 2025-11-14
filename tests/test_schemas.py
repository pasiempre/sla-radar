from pathlib import Path
import sqlite3

# Use the same DB path the drift / models layer uses
from src.models.drift import DB_PATH as DB
from src.ingest.loader import load_all


def _ensure_db_built():
    """
    Helper: if the warehouse DB doesn't exist yet, build it via load_all(reset=True).
    """
    if not DB.exists():
        load_all(reset=True)


def test_db_created_and_views_exist():
    """
    Ensure that the warehouse DB exists and core views are present.
    """
    _ensure_db_built()
    assert DB.exists(), f"warehouse.db not found even after running load_all() at {DB}"

    con = sqlite3.connect(DB)
    try:
        cur = con.cursor()
        expected_views = [
            "v_interval_enriched",
            "v_sla_policy_current",
            "v_staffing_per_interval",
            "v_model_inputs",
        ]
        for view in expected_views:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='view' AND name = ?",
                (view,),
            )
            row = cur.fetchone()
            assert row is not None, f"View {view} is missing from schema"
    finally:
        con.close()