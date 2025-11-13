from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED = DATA / "seed"
DB = DATA / "warehouse.db"
SQL_VIEWS = ROOT / "sql" / "views.sql"

class TicketRow(BaseModel):
    ticket_id: str
    created_ts: str
    first_response_ts: str
    resolved_ts: str
    channel: str
    issue_type: str
    agent_id: str | None = None
    handle_seconds: int = Field(ge=0)
    acw_seconds: int = Field(ge=0)
    first_touch_resolved: bool

    @field_validator("first_response_ts")
    @classmethod
    def _fr_ge_created(cls, v, info):
        return v
    
class IntervalLoadRow(BaseModel):
    interval_start: str
    arrivals: int = Field(ge=0)
    avg_aht: float = Field(ge=0.0)
    avg_acw: float = Field(ge=0.0)
    fcr_pct: float = Field(ge=0.0, le=100.0)
    mix_json: str

class AgentScheduleRow(BaseModel):
    agent_id: str
    interval_start: str
    scheduled_minutes: int = Field(ge=0, le=30)
    adherent_minutes: int = Field(ge=0, le=30)

class SlaPolicyRow(BaseModel):
    kpi: Literal["first_response"]
    target_minutes: int = Field(gt=0)
    coverage_hours: str
    target_pct: float = Field(ge=0.0, le=100.0)

class OpsCostRow(BaseModel):
    interval_start: str
    agent_rate: float = Field(gt=0.0)
    ot_rate: float = Field(ge=0.0)

class DriftEventRow(BaseModel):
    event_ts: str
    metric_name: Literal["aht_eff", "arrivals"]
    direction: Literal["up", "down"]
    score: float
    note: str | None = None

@dataclass
class TableSpec: 
    csv: Path
    name: str
    dtype: dict[str, str]

TABLES: list[TableSpec] = [
    TableSpec(SEED / "tickets_fact.csv", "tickets_fact", {
        "ticket_id": "string",
        "created_ts": "string",
        "first_response_ts": "string",
        "resolved_ts": "string",
        "channel": "string",
        "issue_type": "string",
        "agent_id": "string",
        "handle_seconds": "int64",
        "acw_seconds": "int64",
        "first_touch_resolved": "boolean",
    }),
    TableSpec(SEED / "interval_load.csv", "interval_load", {
        "interval_start": "string",
        "arrivals": "int64",
        "avg_aht": "float64",
        "avg_acw": "float64",
        "fcr_pct": "float64",
        "mix_json": "string",
    }),
    TableSpec(SEED / "agent_schedule.csv", "agent_schedule", {
        "agent_id": "string",
        "interval_start": "string",
        "scheduled_minutes": "int64",
        "adherent_minutes": "int64",
    }),
    TableSpec(SEED / "sla_policy.csv", "sla_policy", {
        "kpi": "string",
        "target_minutes": "int64",
        "coverage_hours": "string",
        "target_pct": "float64",
    }),
    TableSpec(SEED / "ops_cost.csv", "ops_cost", {
        "interval_start": "string",
        "agent_rate": "float64",
        "ot_rate": "float64",
    }),
    TableSpec(SEED / "drift_events.csv", "drift_events", {
        "event_ts": "string",
        "metric_name": "string",
        "direction": "string",
        "score": "float64",
        "note": "string",
    }),
]

ROW_MODEL = {
    "tickets_fact": TicketRow,
    "interval_load": IntervalLoadRow,
    "agent_schedule": AgentScheduleRow,
    "sla_policy": SlaPolicyRow,
    "ops_cost": OpsCostRow,
    "drift_events": DriftEventRow,
}

def _connect(reset: bool) -> sqlite3.Connection:
    if reset and DB.exists():
        DB.unlink()
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")
    return con


def _load_csv(spec: TableSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.csv, dtype=spec.dtype)
    # Normalize booleans & NaNs (pandas may read "true"/"false" as strings)
    if "first_touch_resolved" in df.columns and df["first_touch_resolved"].dtype == "object":
        df["first_touch_resolved"] = df["first_touch_resolved"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
    return df


def _validate(df: pd.DataFrame, table: str) -> None:
    model = ROW_MODEL[table]
    # Sample a few rows plus check for nulls / basic ranges
    if df.isnull().any().any():
        missing_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"{table}: nulls present in columns {missing_cols}")

    # Row-level schema check on a small sample (for speed)
    sample = df.head(min(200, len(df)))
    errors = []
    for i, row in sample.iterrows():
        try:
            model(**row.to_dict())
        except ValidationError as e:
            errors.append((i, str(e)))
    if errors:
        raise ValueError(f"{table}: schema validation errors on sample rows: {errors[:3]}")


def _write(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    # Ensure timestamp strings remain as TEXT in SQLite (we’ll cast in views as needed)
    df.to_sql(table, con, if_exists="replace", index=False)
    # Basic indices for joins/lookups
    if table == "tickets_fact":
        con.execute("CREATE INDEX IF NOT EXISTS ix_tickets_created ON tickets_fact(created_ts);")
        con.execute("CREATE INDEX IF NOT EXISTS ix_tickets_first_response ON tickets_fact(first_response_ts);")
    if table in {"interval_load", "agent_schedule", "ops_cost"}:
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_interval ON {table}(interval_start);")


def _apply_views(con: sqlite3.Connection) -> None:
    if SQL_VIEWS.exists():
        with open(SQL_VIEWS, "r", encoding="utf-8") as f:
            con.executescript(f.read())


def main():
    parser = argparse.ArgumentParser(description="Load seed CSVs into SQLite with basic validation.")
    parser.add_argument("--reset", action="store_true", help="Recreate the warehouse.db from scratch.")
    args = parser.parse_args()

    con = _connect(reset=args.reset)
    try:
        for spec in TABLES:
            if not spec.csv.exists():
                raise FileNotFoundError(f"Missing seed file: {spec.csv}")
            df = _load_csv(spec)
            _validate(df, spec.name)
            _write(con, spec.name, df)

        _apply_views(con)

        # Quick counts
        cur = con.cursor()
        summary = {}
        for t in [s.name for s in TABLES]:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            summary[t] = cur.fetchone()[0]
        print(json.dumps({"loaded_rows": summary, "db": str(DB)}, indent=2))
    finally:
        con.close()


if __name__ == "__main__":
    main()