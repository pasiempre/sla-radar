# src/models/forecast.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


DB_PATH = Path("data/warehouse.db")


@dataclass
class ForecastConfig:
    horizon_intervals: int = 16  # next business day (16x 30-min intervals for 09:00-17:00)
    min_history: int = 7 * 16    # at least 1 week of intervals
    use_ets: bool = True         # try ETS; if fails, use seasonal mean only


def load_model_inputs(db: Path = DB_PATH) -> pd.DataFrame:
    """
    Load v_model_inputs from SQLite.

    Returns columns:
      interval_start (datetime64[ns]), arrivals, avg_aht_seconds, avg_acw_seconds, aht_eff_seconds, fcr_pct, agents_effective
    """
    con = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(
            """
            SELECT interval_start, arrivals, avg_aht_seconds, avg_acw_seconds, aht_eff_seconds, fcr_pct, agents_effective
            FROM v_model_inputs
            ORDER BY interval_start
            """,
            con,
            parse_dates=["interval_start"],
        )
    finally:
        con.close()

    # Fill staffing if null (shouldn't happen, but be defensive)
    if "agents_effective" in df.columns:
        df["agents_effective"] = df["agents_effective"].fillna(0).astype(int)

    return df


def seasonal_keys(ts: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Return weekday (0=Mon) and half-hour slot index per day (0..n-1) for a DateTimeIndex.
    Assumes intervals are strictly aligned (e.g., 09:00..17:00 every 30m).
    """
    wk = ts.index.dayofweek
    slot = (ts.index.hour * 60 + ts.index.minute) // 30
    return wk, slot


def seasonal_mean_forecast(y: pd.Series, horizon: int) -> pd.Series:
    """
    Simple seasonal-mean forecast using weekday x slot averages.
    """
    # Build keys
    wk, slot = seasonal_keys(y)

    # Seasonal table
    df = pd.DataFrame({"y": y.values, "wk": wk, "slot": slot})
    seasonal = df.groupby(["wk", "slot"])["y"].mean()

    # Next horizon index
    last = y.index.max()
    freq = pd.infer_freq(y.index)
    if freq is None:
        # default to 30min (updated from deprecated '30T')
        freq = "30min"
    future_index = pd.date_range(last + pd.Timedelta(freq), periods=horizon, freq=freq)

    wk_f, slot_f = seasonal_keys(pd.Series(index=future_index, dtype=float))
    preds = []
    for w, s in zip(wk_f, slot_f):
        preds.append(float(seasonal.get((w, s), y.mean())))

    return pd.Series(preds, index=future_index, name="arrivals_forecast")


def ets_forecast(y: pd.Series, horizon: int) -> Optional[pd.Series]:
    """
    ETS (Holt-Winters) with multiplicative seasonality if enough data; fall back if it fails.
    """
    try:
        # Try to infer a seasonal period ~ a business day length in steps (commonly 16 for 8 hours @ 30-min)
        # If irregular, fallback quickly.
        freq = pd.infer_freq(y.index)
        if freq is None:
            return None

        # Estimate daily cycle length by counting unique slots in a day
        one_day = y.index.normalize().value_counts().index[:1]
        # Safer: use 16 by default
        sp = 16

        model = ExponentialSmoothing(
            y.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=sp,
            initialization_method="estimated",
        ).fit(optimized=True)

        fcst = model.forecast(horizon)
        fcst.name = "arrivals_forecast"
        return fcst.clip(lower=0)
    except Exception:
        return None


def forecast_arrivals(series: pd.Series, cfg: ForecastConfig) -> pd.Series:
    """
    Main entry: produce arrivals forecast for next cfg.horizon_intervals.
    Strategy:
      - If history >= min_history and ETS works, blend: 50% ETS, 50% seasonal mean
      - Else, seasonal mean only
    """
    y = series.dropna().astype(float)
    if len(y) < cfg.min_history:
        return seasonal_mean_forecast(y, cfg.horizon_intervals)

    smean = seasonal_mean_forecast(y, cfg.horizon_intervals)
    if cfg.use_ets:
        ets = ets_forecast(y, cfg.horizon_intervals)
    else:
        ets = None

    if ets is None or not np.isfinite(ets.values).all():
        return smean

    # Simple blend (could learn weights later)
    out = 0.5 * smean.values + 0.5 * ets.values
    return pd.Series(out, index=smean.index, name="arrivals_forecast")


def next_day_arrivals_forecast(db: Path = DB_PATH, cfg: ForecastConfig = ForecastConfig()) -> pd.DataFrame:
    """
    Convenience: load inputs, forecast arrivals next business day, return a small DataFrame
    with index=forecast intervals, columns: arrivals_forecast.
    """
    df = load_model_inputs(db)
    df = df.set_index("interval_start").sort_index()
    y = df["arrivals"]
    fcst = forecast_arrivals(y, cfg)
    return fcst.to_frame()