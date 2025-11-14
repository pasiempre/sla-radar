from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

# Project root and paths
ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "warehouse.db"
SEED_DIR = ROOT / "data" / "seed"
VIEWS_SQL = ROOT / "sql" / "views.sql"


def _load_csv(
    con: sqlite3.Connection,
    table: str,
    csv_name: str,
    parse_dates: Iterable[str] | None = None,
) -> int:
    """
    Load a CSV from data/seed into a SQLite table (replace if exists).

    - Does NOT pass dtype= to pandas (keeps things simple & safe).
    - Applies datetime conversion *after* reading, and only for columns
      that actually exist in the CSV.
    """
    csv_path = SEED_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing seed CSV: {csv_path}")

    # Plain CSV read; no dtype, no parse_dates here
    df = pd.read_csv(csv_path)

    # Apply datetime conversion only where columns exist
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
            else:
                # Optional: you can comment this out if it's noisy
                print(f"[loader] Warning: parse_dates col '{col}' not in {csv_name}")

    df.to_sql(table, con, if_exists="replace", index=False)
    return len(df)


def _drop_existing_tables(con: sqlite3.Connection) -> None:
    """Drop the core fact/dim tables if they exist."""
    cur = con.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS tickets_fact;
        DROP TABLE IF EXISTS interval_load;
        DROP TABLE IF EXISTS agent_schedule;
        DROP TABLE IF EXISTS sla_policy;
        DROP TABLE IF EXISTS ops_cost;
        DROP TABLE IF EXISTS drift_events;
        """
    )
    con.commit()


def _apply_views(con: sqlite3.Connection) -> None:
    """Run sql/views.sql to (re)create views."""
    if not VIEWS_SQL.exists():
        # Don't hard-fail if someone hasn't added views yet
        print(f"[loader] Warning: views.sql not found at {VIEWS_SQL}")
        return

    sql_text = VIEWS_SQL.read_text()
    cur = con.cursor()
    cur.executescript(sql_text)
    con.commit()


def load_all(reset: bool = False) -> dict[str, int]:
    """
    Load all seed CSVs into the warehouse DB.

    Returns a dict of {table_name: row_count}.
    """
    con = sqlite3.connect(DB)
    try:
        if reset:
            _drop_existing_tables(con)

        loaded_rows: dict[str, int] = {}

        # tickets_fact: datetime columns if present
        loaded_rows["tickets_fact"] = _load_csv(
            con,
            table="tickets_fact",
            csv_name="tickets_fact.csv",
            parse_dates=["created_at", "first_response_at", "resolved_at"],
        )

        # interval_load
        loaded_rows["interval_load"] = _load_csv(
            con,
            table="interval_load",
            csv_name="interval_load.csv",
            parse_dates=["interval_start"],
        )

        # agent_schedule
        loaded_rows["agent_schedule"] = _load_csv(
            con,
            table="agent_schedule",
            csv_name="agent_schedule.csv",
            parse_dates=["interval_start"],
        )

        # sla_policy
        loaded_rows["sla_policy"] = _load_csv(
            con,
            table="sla_policy",
            csv_name="sla_policy.csv",
            parse_dates=None,
        )

        # ops_cost
        loaded_rows["ops_cost"] = _load_csv(
            con,
            table="ops_cost",
            csv_name="ops_cost.csv",
            parse_dates=None,
        )

        # drift_events
        loaded_rows["drift_events"] = _load_csv(
            con,
            table="drift_events",
            csv_name="drift_events.csv",
            parse_dates=["event_ts"],
        )

        # Rebuild views
        _apply_views(con)

        return loaded_rows

    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load seed CSVs into warehouse.db")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing tables before loading",
    )
    args = parser.parse_args()

    summary = load_all(reset=args.reset)
    print(
        json.dumps(
            {
                "loaded_rows": summary,
                "db": str(DB),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()