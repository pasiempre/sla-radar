# src/models/drift.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd

DB_PATH = Path("data/warehouse.db")


@dataclass
class DriftConfig:
    alpha: float = 0.3          # EWMA smoothing
    hysteresis_k: float = 2.0   # trigger when |z| >= k
    min_run: int = 2            # require >=2 consecutive intervals to fire
    which: Tuple[str, ...] = ("aht_eff_seconds", "arrivals")  # which metrics to monitor


def _ewma(x: pd.Series, alpha: float) -> pd.Series:
    return x.ewm(alpha=alpha, adjust=False).mean()


def _zscore(x: pd.Series, baseline: pd.Series) -> pd.Series:
    # robust z = (x - baseline) / MAD
    resid = x - baseline
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad == 0:
        mad = max(1e-6, np.std(resid))
    return resid / (mad if mad != 0 else 1.0)


def load_inputs_for_drift(db: Path = DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            """
            SELECT interval_start, arrivals, aht_eff_seconds
            FROM v_model_inputs
            ORDER BY interval_start
            """,
            con,
            parse_dates=["interval_start"],
        )
    finally:
        con.close()
    return df


def detect_drift(df: pd.DataFrame, cfg: DriftConfig = DriftConfig()) -> pd.DataFrame:
    """
    Compute EWMA baseline & robust z, then emit sustained excursions.
    Returns DataFrame with columns: event_ts, metric_name, direction, score, note
    """
    out_rows = []
    df = df.set_index("interval_start").sort_index()

    for metric in cfg.which:
        series = df[metric].astype(float)
        baseline = _ewma(series, cfg.alpha)
        z = _zscore(series, baseline)

        # hysteresis: sustained |z| >= k for >= min_run
        over = (z.abs() >= cfg.hysteresis_k).astype(int)
        # find runs
        run_id = (over.diff().fillna(0) == 1).cumsum()
        # consider only runs where over==1
        mask = over.astype(bool)
        groups = z[mask].groupby(run_id[mask])

        for _, g in groups:
            if len(g) >= cfg.min_run:
                ts = g.index[-1]  # mark by last point in the run
                direction = "up" if g.iloc[-1] > 0 else "down"
                score = float(g.abs().mean())  # average |z| during run
                out_rows.append(
                    dict(
                        event_ts=ts,
                        metric_name=metric,
                        direction=direction,
                        score=score,
                        note=f"EWMA α={cfg.alpha}, k={cfg.hysteresis_k}, run={len(g)}",
                    )
                )

    return pd.DataFrame(out_rows).sort_values("event_ts").reset_index(drop=True)


def write_drift_events(events: pd.DataFrame, db: Path = DB_PATH) -> None:
    if events.empty:
        return
    con = sqlite3.connect(db)
    try:
        events.to_sql("drift_events", con, if_exists="append", index=False)
    finally:
        con.close()


def run_drift(db: Path = DB_PATH, cfg: DriftConfig = DriftConfig()) -> pd.DataFrame:
    df = load_inputs_for_drift(db)
    events = detect_drift(df, cfg)
    write_drift_events(events, db)
    return events