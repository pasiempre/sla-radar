from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# add project root (sla-radar/) to sys.path so "src" imports work
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.forecast import next_day_arrivals_forecast
from src.models.drift import run_drift, DriftConfig
from src.scenarios.what_if import simulate_what_if, WhatIfConfig
from src.reports.pdf_export import export_ops_brief

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "warehouse.db"
REPORTS = ROOT / "data" / "reports" / "ops_brief.pdf"

@st.cache_data(show_spinner=False)
def load_interval_inputs():
    con = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query("""
            SELECT e.interval_start,
                   e.arrivals,
                   e.avg_aht_seconds,
                   e.avg_acw_seconds,
                   e.aht_eff_seconds,
                   e.fcr_pct,
                   COALESCE(s.agents_effective, 0) AS agents_effective
            FROM v_interval_enriched e
            LEFT JOIN v_staffing_per_interval s
              ON s.interval_start = e.interval_start
            ORDER BY 1
        """, con, parse_dates=["interval_start"])
        return df
    finally:
        con.close()

st.set_page_config(page_title="SLA Drift Radar", layout="wide")
st.title("SLA Drift Radar & Staffing What-Ifs")

tab_radar, tab_why, tab_whatif = st.tabs(["📡 Radar (Today)", "🔍 Why (Drift)", "🧮 What-If"])

# --- RADAR TAB ---
with tab_radar:
    df = load_interval_inputs()
    st.subheader("SLA Headline & Forecast")
    # Headline proxy: compute simple modeled SLA today with current staffing
    # (quick reuse of what_if with 0 deltas)
    policy = 15.0
    fdf = next_day_arrivals_forecast(DB)
    fig_fc = px.line(fdf.reset_index(), x="index", y="arrivals_forecast", title="Arrivals: Next-Day Forecast")
    st.plotly_chart(fig_fc, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("AHT (sec)", f"{df['avg_aht_seconds'].tail(16).mean():.0f}")
    k2.metric("ACW (sec)", f"{df['avg_acw_seconds'].tail(16).mean():.0f}")
    k3.metric("FCR (%)", f"{df['fcr_pct'].tail(16).mean():.1f}")

# --- WHY TAB ---
with tab_why:
    st.subheader("Drift Timeline (EWMA)")
    events = run_drift(DB, DriftConfig(alpha=0.3, hysteresis_k=1.8, min_run=2))
    if events.empty:
        st.info("No sustained drift events detected with current thresholds.")
    else:
        st.dataframe(events.tail(20), use_container_width=True)
        # Quick timeline figure
        ef = events.copy()
        ef["event_ts"] = pd.to_datetime(ef["event_ts"])
        fig_ev = px.scatter(ef, x="event_ts", y="score", color="metric_name",
                            symbol="direction", title="Detected Drift Events")
        st.plotly_chart(fig_ev, use_container_width=True)

    # Simple contribution waterfall proxy
    st.subheader("Contribution Breakdown (Δ vs. baseline)")
    # For MVP: show last-day averages vs. prior-day averages for AHT/ACW/Arrivals
    df["day"] = df["interval_start"].dt.date
    last_day = df["day"].max()
    prev_day = sorted(df["day"].unique())[-2]
    agg = (df.groupby("day")[["avg_aht_seconds","avg_acw_seconds","arrivals"]].mean()
             .loc[[prev_day,last_day]].diff().iloc[-1].rename({"avg_aht_seconds":"ΔAHT","avg_acw_seconds":"ΔACW","arrivals":"ΔArrivals"}))
    wf = pd.DataFrame({"component": agg.index, "delta": agg.values})
    fig_wf = px.bar(wf, x="component", y="delta", title=f"Contributions: {prev_day} → {last_day}")
    st.plotly_chart(fig_wf, use_container_width=True)

# --- WHAT-IF TAB ---
with tab_whatif:
    st.subheader("Scenario Controls")
    c1, c2, c3, c4 = st.columns(4)
    extra_agents = c1.slider(
        "Δ agents per interval",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        help="Negative: fewer agents; positive: more agents"
    )
    acw_red = c2.slider(
        "ACW change (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Negative: more ACW (worse); positive: less ACW (better)"
    )
    fcr_up = c3.slider(
        "FCR change (%)",
        min_value=-20,
        max_value=20,
        value=0,
        step=2,
        help="Negative: lower FCR; positive: higher FCR"
    )
    ot_rate = c4.number_input("OT $/hour (optional)", min_value=0.0, value=33.0, step=1.0)

    cfg_base = WhatIfConfig(
        acw_reduction_pct=0.0,
        fcr_uplift_pct=0.0,
        extra_agents=0,
        ot_hourly=ot_rate,
    )
    base = simulate_what_if(cfg_base, db=DB)
    base = base.rename(columns={"sla_attainment_pct": "sla_base"})

    # Scenario (user controls)
    cfg = WhatIfConfig(
        acw_reduction_pct=acw_red,
        fcr_uplift_pct=fcr_up,
        extra_agents=extra_agents,
        ot_hourly=ot_rate,
    )
    out = simulate_what_if(cfg, db=DB).rename(columns={"sla_attainment_pct": "sla_scenario"})

    merged = base[["interval_start", "sla_base"]].merge(
        out[["interval_start", "sla_scenario", "agents", "aht_eff_seconds"]],
        on="interval_start",
        how="inner",
    )

    st.subheader("Preview of Projected Data (Baseline vs Scenario)")
    st.dataframe(
        merged[["interval_start", "agents", "aht_eff_seconds", "sla_base", "sla_scenario"]].head(10),
        use_container_width=True,
    )

    st.subheader("Projected SLA by Interval")

    fig_sla = px.line(
        merged,
        x="interval_start",
        y=["sla_base", "sla_scenario"],
        title="SLA Attainment %: Baseline vs Scenario",
        labels={"value": "SLA %", "variable": "Curve"},
    )
    st.plotly_chart(fig_sla, use_container_width=True)

    st.write(
        f"**Baseline mean SLA (last day):** {merged['sla_base'].tail(16).mean():.1f}%  "
        f"**Scenario mean SLA (last day):** {merged['sla_scenario'].tail(16).mean():.1f}%"
    )

    # Export Ops Brief (simple HTML -> PDF)
    if st.button("Export Ops Brief (PDF)"):
        html = f"""
        <h1>SLA Ops Brief</h1>
        <p><b>Scenario:</b> Δ agents={extra_agents}, ACW {acw_red}%, FCR {fcr_up}%</p>
        <p>Baseline mean SLA (last day): {merged['sla_base'].tail(16).mean():.1f}%</p>
        <p>Scenario mean SLA (last day): {merged['sla_scenario'].tail(16).mean():.1f}%</p>
        """
        path = export_ops_brief(html, REPORTS)
        st.success(f"Exported: {path}")