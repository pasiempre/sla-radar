from __future__ import annotations
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "warehouse.db"

@dataclass
class WhatIfConfig:
    acw_reduction_pct: float = 0.0          # 10 -> -10% ACW
    fcr_uplift_pct: float = 0.0             # 5  -> +5% FCR (reduces recontacts)
    extra_agents: int = 0                   # flat agents delta per interval (can be negative)
    ot_hourly: float = 0.0                  # cost calc (optional)
    arrivals_multiplier: float = 1.0        # traffic scale (0.5 .. 2.0)
    aht_change_pct: float = 0.0             # +%/-% applied to TALK time (avg_aht_seconds)
    target_minutes_override: float | None = None  # if set, use this instead of DB policy

def _load_inputs(con: sqlite3.Connection) -> pd.DataFrame:
    sql = """
        SELECT
            e.interval_start,
            e.arrivals,                 -- arrivals in the 30-min bucket
            e.avg_aht_seconds,          -- talk time (sec)
            e.avg_acw_seconds,          -- ACW (sec)
            e.aht_eff_seconds,          -- talk + ACW (sec)
            e.fcr_pct,                  -- 0..1
            COALESCE(s.agents_effective, 0) AS agents_effective
        FROM v_interval_enriched e
        LEFT JOIN v_staffing_per_interval s
          ON s.interval_start = e.interval_start
        ORDER BY 1
    """
    return pd.read_sql_query(sql, con, parse_dates=["interval_start"])

# Numerically-stable Erlang C helpers
def _fact(n: int) -> float:
    return math.prod(range(1, n + 1)) if n > 1 else 1.0

def _erlang_c_wait_prob(lambda_rate: float, mu: float, s: int) -> float:
    """
    lambda_rate: arrivals per second
    mu:          service rate per second (1 / AHT_eff_seconds)
    s:           agents
    Returns Pw = P(customer waits > 0)
    """
    if s <= 0 or mu <= 0 or lambda_rate <= 0:
        return 1.0
    a = lambda_rate / mu  # traffic in erlangs
    rho = a / s
    if rho >= 1.0:
        return 1.0

    # Sum_{k=0}^{s-1} a^k/k!
    summ = sum((a**k) / _fact(k) for k in range(s))
    top = (a**s) / (_fact(s) * (1 - rho))
    return top / (summ + top)

def _service_level(lambda_rate: float, mu: float, s: int, t_seconds: float) -> float:
    """
    Returns % answered within t_seconds using Erlang C:
      SL(t) = 1 - Pw * exp(-(s*mu - lambda) * t)
    """
    if s <= 0 or mu <= 0 or lambda_rate <= 0:
        return 0.0
    a = lambda_rate / mu
    rho = a / s
    if rho >= 1.0:
        return 0.0  # unstable queue
    Pw = _erlang_c_wait_prob(lambda_rate, mu, s)
    beta = (s * mu) - lambda_rate
    # Guard for numeric safety
    if beta <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - Pw * math.exp(-beta * t_seconds))))

def simulate_what_if(cfg: WhatIfConfig, db: Path = DB) -> pd.DataFrame:
    con = sqlite3.connect(db)
    try:
        df = _load_inputs(con)

        # Apply AHT change to TALK time first, then recompute effective AHT with ACW change
        talk_new = df["avg_aht_seconds"] * (1 + cfg.aht_change_pct / 100.0)
        acw_new  = df["avg_acw_seconds"] * (1 - cfg.acw_reduction_pct / 100.0)
        aht_eff_new = talk_new + acw_new

        # 2) FCR uplift reduces re-contacts (dampened to 80%)
        arrivals_new = (
            df["arrivals"]
            * cfg.arrivals_multiplier
            * (1 - 0.8 * cfg.fcr_uplift_pct / 100.0)
        )

        # 3) Agents per interval
        agents_new = (df["agents_effective"].fillna(0).astype(int) + int(cfg.extra_agents)).clip(lower=0)

        # 4) SLA target in SECONDS (prefer seconds; else minutes*60; fallback 900s = 15 min)
        sla_target_seconds = 900.0
        # Allow explicit override from UI
        if cfg.target_minutes_override is not None:
            sla_target_seconds = float(cfg.target_minutes_override) * 60.0
        # Try to read a seconds-based policy first
        try:
            df_sec = pd.read_sql_query(
                "SELECT target_seconds FROM v_sla_policy_current LIMIT 1",
                con
            )
            if not df_sec.empty and pd.notna(df_sec.iloc[0, 0]):
                sla_target_seconds = float(df_sec.iloc[0, 0])
        except Exception:
            pass

        # If seconds not available, try minutes and convert to seconds
        if sla_target_seconds == 900.0:
            try:
                df_min = pd.read_sql_query(
                    "SELECT target_minutes FROM v_sla_policy_current LIMIT 1",
                    con
                )
                if not df_min.empty and pd.notna(df_min.iloc[0, 0]):
                    sla_target_seconds = float(df_min.iloc[0, 0]) * 60.0
            except Exception:
                pass

        # 5) Compute per-interval Service Level using Erlang C
        # Convert arrivals in 30-min bins to per-second rate
        lam_per_sec = arrivals_new / (30.0 * 60.0)
        # Service rate per second (guard against divide-by-zero)
        mu_per_sec = 1.0 / aht_eff_new.replace(0, np.nan)

        # Utilization rho = lambda / (s * mu). Clean inf/nan for display
        rho = (lam_per_sec / (agents_new * mu_per_sec)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Service level at the policy target
        sla = [
            _service_level(l, m, int(s), sla_target_seconds)
            if (l > 0 and m > 0 and s > 0) else 0.0
            for l, m, s in zip(lam_per_sec, mu_per_sec, agents_new)
        ]

        out = pd.DataFrame({
            "interval_start": df["interval_start"],
            "arrivals": arrivals_new,
            "aht_eff_seconds": aht_eff_new,
            "agents": agents_new,
            "utilization_rho": rho,
            "sla_attainment_pct": np.array(sla) * 100.0,
        }).sort_values("interval_start")

        # Optional quick cost proxy
        if cfg.ot_hourly > 0:
            out["incremental_cost"] = (cfg.extra_agents * 0.5 * cfg.ot_hourly)
        else:
            out["incremental_cost"] = 0.0

        # Round only for display
        return out.assign(
            arrivals=lambda d: d["arrivals"].round(2),
            aht_eff_seconds=lambda d: d["aht_eff_seconds"].round(2),
            utilization_rho=lambda d: d["utilization_rho"].round(3),
            sla_attainment_pct=lambda d: d["sla_attainment_pct"].round(2),
        )
    finally:
        con.close()