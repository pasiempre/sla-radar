# tests/test_schemas.py
import sqlite3
from pathlib import Path

def test_db_created_and_views_exist():
    db = Path("data/warehouse.db")
    assert db.exists(), "warehouse.db not found – run loader first."

    con = sqlite3.connect(db)
    cur = con.cursor()

    for view in ["v_interval_enriched", "v_staffing_per_interval", "v_model_inputs", "v_sla_policy_current"]:
        cur.execute("SELECT name FROM sqlite_master WHERE type='view' AND name=?", (view,))
        assert cur.fetchone(), f"Missing view: {view}"

    cur.execute("SELECT COUNT(*) FROM v_model_inputs")
    assert cur.fetchone()[0] > 0, "v_model_inputs should have rows"
    con.close()