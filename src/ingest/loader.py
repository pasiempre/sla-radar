from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "warehouse.db"
SEED_DIR = ROOT / "data" / "seed"
VIEWS_SQL = ROOT / "sql" / "views.sql"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def _drop_tables(con: sqlite3.Connection) -> None:
    """
    Drop core tables if they exist.
    This makes --reset idempotent and safe to run multiple times.
    """
    tables = [
        "tickets_fact",
        "interval_load",
        "agent_schedule",
        "sla_policy",
        "ops_cost",
        "drift_events",
    ]
    cur = con.cursor()
    for t in tables:
        cur.execute(f"DROP TABLE IF EXISTS {t}")
    con.commit()


def _load_csv(
    con: sqlite3.Connection,
    table: str,
    csv_name: str,
    dtypes: Dict[str, str] | None = None,
    parse_dates: list[str] | None = None,
) -> int:
    """
    Load a CSV from data/seed into a table, replacing existing content.
    """
    csv_path = SEED_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Seed file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=dtypes, parse_dates=parse_dates)
    df.to_sql(table, con, if_exists="replace", index=False)
    return len(df)


def _apply_views(con: sqlite3.Connection) -> None:
    """
    Execute all view definitions from sql/views.sql.
    Assumes statements are separated by ';'.
    """
    if not VIEWS_SQL.exists():
        # Not fatal; just skip if views file isn't there.
        return

    sql_text = VIEWS_SQL.read_text()
    cur = con.cursor()
    # Simple split on ';' – fine for our purposes.
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        cur.execute(stmt)
    con.commit()


def load_all(reset: bool = False) -> dict:
    """
    Core entry point: rebuild database from seed CSVs.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _connect(DB_PATH) as con:
        if reset:
            _drop_tables(con)

        loaded_rows: Dict[str, int] = {}

        # NOTE: if your actual CSV column names differ, we can tighten dtypes later.
        loaded_rows["tickets_fact"] = _load_csv(
            con,
            table="tickets_fact",
            csv_name="tickets_fact.csv",
            parse_dates=["created_at", "first_response_at", "resolved_at"],
        )

        loaded_rows["interval_load"] = _load_csv(
            con,
            table="interval_load",
            csv_name="interval_load.csv",
            parse_dates=["interval_start"],
        )

        loaded_rows["agent_schedule"] = _load_csv(
            con,
            table="agent_schedule",
            csv_name="agent_schedule.csv",
            parse_dates=["interval_start"],
        )

        loaded_rows["sla_policy"] = _load_csv(
            con,
            table="sla_policy",
            csv_name="sla_policy.csv",
        )

        loaded_rows["ops_cost"] = _load_csv(
            con,
            table="ops_cost",
            csv_name="ops_cost.csv",
            parse_dates=["interval_start"],
        )

        loaded_rows["drift_events"] = _load_csv(
            con,
            table="drift_events",
            csv_name="drift_events.csv",
            parse_dates=["start_ts", "end_ts"],
        )

        _apply_views(con)

    return {
        "loaded_rows": loaded_rows,
        "db": str(DB_PATH),
    }


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="(Re)load SLA Radar warehouse from seed CSVs."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate tables before loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_all(reset=args.reset)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

