from pathlib import Path

from src.ingest.loader import load_all


def test_load_all_runs_and_creates_tables():
    """
    Smoke test: ensure load_all() runs and reports rows for all core tables.

    It should return either:
      - a flat dict like {"tickets_fact": 8530, ...}
      - or {"loaded_rows": {...}, "db": ".../warehouse.db"}

    We normalize both shapes into `rows` and assert on that.
    """
    summary = load_all(reset=False)

    # Normalize shape
    if "loaded_rows" in summary:
        rows = summary["loaded_rows"]
    else:
        rows = summary

    assert isinstance(rows, dict)

    expected_tables = [
        "tickets_fact",
        "interval_load",
        "agent_schedule",
        "sla_policy",
        "ops_cost",
        "drift_events",
    ]

    for table in expected_tables:
        assert table in rows, f"{table} not reported in load_all() summary"
        assert rows[table] > 0, f"{table} row count should be > 0"