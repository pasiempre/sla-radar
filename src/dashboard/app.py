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

SLA_TARGET = 80.0

# Predefined scenario presets for the What-If tab
PRESETS = {
    "Custom (freeform)": {
        "extra_agents": None,
        "acw_red": None,
        "fcr_up": None,
    },
    "Staffing bump: +2 agents": {
        "extra_agents": 2,
        "acw_red": 0,
        "fcr_up": 0,
    },
    "ACW discipline: +10% fewer ACW seconds": {
        "extra_agents": 0,
        "acw_red": 10,   # +10% = less ACW (better)
        "fcr_up": 0,
    },
    "FCR uplift: +5%": {
        "extra_agents": 0,
        "acw_red": 0,
        "fcr_up": 5,
    },
    "Balanced improvement: +2 agents, +5% ACW reduction, +5% FCR": {
        "extra_agents": 2,
        "acw_red": 5,
        "fcr_up": 5,
    },
}

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

    # --- Session state setup for presets + sliders ---
    if "scenario_preset" not in st.session_state:
        st.session_state.scenario_preset = "Custom (freeform)"

    if "slider_extra_agents" not in st.session_state:
        st.session_state.slider_extra_agents = 0
    if "slider_acw_red" not in st.session_state:
        st.session_state.slider_acw_red = 0
    if "slider_fcr_up" not in st.session_state:
        st.session_state.slider_fcr_up = 0

    preset_names = list(PRESETS.keys())
    default_preset_index = preset_names.index(st.session_state.scenario_preset)

    # Choose a scenario preset (can still tweak sliders afterward)
    preset_name = st.selectbox(
        "Scenario preset",
        options=preset_names,
        index=default_preset_index,
        help="Pick a common scenario, then fine-tune with the sliders below.",
    )

    # If preset changed, update slider state based on preset values
    if preset_name != st.session_state.scenario_preset:
        st.session_state.scenario_preset = preset_name
        preset = PRESETS[preset_name]
        if preset["extra_agents"] is not None:
            st.session_state.slider_extra_agents = preset["extra_agents"]
        if preset["acw_red"] is not None:
            st.session_state.slider_acw_red = preset["acw_red"]
        if preset["fcr_up"] is not None:
            st.session_state.slider_fcr_up = preset["fcr_up"]

    c1, c2, c3, c4 = st.columns(4)
    extra_agents = c1.slider(
        "Δ agents per interval",
        min_value=-20,
        max_value=20,
        step=1,
        key="slider_extra_agents",
        help="Negative: fewer agents staffed vs baseline; positive: more agents per 30-min interval.",
    )
    acw_red = c2.slider(
        "ACW change vs baseline (%)",
        min_value=-50,
        max_value=50,
        step=5,
        key="slider_acw_red",
        help="Positive values mean *less* ACW (better). Negative values mean *more* ACW (worse).",
    )
    fcr_up = c3.slider(
        "FCR change vs baseline (%)",
        min_value=-20,
        max_value=20,
        step=2,
        key="slider_fcr_up",
        help="Positive values increase FCR (fewer re-contacts). Negative values reduce FCR.",
    )
    ot_rate = c4.number_input(
        "OT $/hour (optional)",
        min_value=0.0,
        value=33.0,
        step=1.0,
        help="Used only for incremental cost estimates in the scenario.",
    )

    scenario_label = (
        "Custom scenario" if preset_name == "Custom (freeform)" else preset_name
    )
    st.caption(
        f"Scenario: {scenario_label} — Δ agents = {extra_agents:+d}, "
        f"ACW change = {acw_red:+.0f}%, FCR change = {fcr_up:+.0f}%."
    )

    # --- Baseline (no changes) ---
    cfg_base = WhatIfConfig(
        acw_reduction_pct=0.0,
        fcr_uplift_pct=0.0,
        extra_agents=0,
        ot_hourly=ot_rate,
    )
    base = simulate_what_if(cfg_base, db=DB).rename(
        columns={"sla_attainment_pct": "sla_base"}
    )

    # --- Scenario (user controls) ---
    cfg = WhatIfConfig(
        acw_reduction_pct=acw_red,
        fcr_uplift_pct=fcr_up,
        extra_agents=extra_agents,
        ot_hourly=ot_rate,
    )
    out = simulate_what_if(cfg, db=DB).rename(
        columns={"sla_attainment_pct": "sla_scenario"}
    )

    merged = base[["interval_start", "sla_base"]].merge(
        out[["interval_start", "sla_scenario", "agents", "aht_eff_seconds"]],
        on="interval_start",
        how="inner",
    )

    # --- Focus on "today" only (last calendar date) ---
    merged["day"] = merged["interval_start"].dt.date
    today = merged["day"].max()
    today_df = merged[merged["day"] == today].copy()

    # --- Delta cards (today baseline vs scenario) ---
    base_today = today_df["sla_base"].mean()
    scenario_today = today_df["sla_scenario"].mean()
    delta_today = scenario_today - base_today

    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Baseline SLA (today)",
        f"{base_today:.1f}%",
    )
    k2.metric(
        "Scenario SLA (today)",
        f"{scenario_today:.1f}%",
        delta=f"{delta_today:+.1f} pts",
    )
    k3.metric(
        "Δ agents per interval",
        f"{extra_agents:+d}",
        help="Scenario staffing change vs baseline.",
    )

    st.subheader("Preview of Projected Data (Baseline vs Scenario)")
    st.dataframe(
        merged[
            ["interval_start", "agents", "aht_eff_seconds", "sla_base", "sla_scenario"]
        ].head(10),
        use_container_width=True,
    )

    st.subheader("Projected SLA by Interval")

    # Use nicer legend labels
    plot_df = merged.rename(
        columns={
            "sla_base": "Baseline SLA",
            "sla_scenario": "Scenario SLA",
        }
    )

    fig_sla = px.line(
        plot_df,
        x="interval_start",
        y=["Baseline SLA", "Scenario SLA"],
        title="SLA Attainment %: Baseline vs Scenario",
        labels={"value": "SLA %", "variable": "Curve"},
    )
    st.plotly_chart(fig_sla, use_container_width=True)

    st.write(
        f"**Baseline mean SLA (last day):** {base_today:.1f}%  "
        f"**Scenario mean SLA (last day):** {scenario_today:.1f}%  "
        f"(**Δ:** {delta_today:+.1f} pts)"
    )

    # --- At-risk intervals (scenario view) ---
    st.subheader(f"At-risk intervals (Scenario < {SLA_TARGET:.0f}% SLA)")

    at_risk = today_df[today_df["sla_scenario"] < SLA_TARGET].copy()

    if at_risk.empty:
        st.success("No intervals below the SLA target under this scenario. ✅")
    else:
        # Keep only a few key columns for readability
        at_risk_view = at_risk[
            ["interval_start", "agents", "aht_eff_seconds", "sla_base", "sla_scenario"]
        ].sort_values("interval_start")

        st.dataframe(
            at_risk_view,
            use_container_width=True,
        )

    # Export Ops Brief (simple HTML -> PDF)
    if st.button("Export Ops Brief (PDF)"):
        html = f"""
        <h1>SLA Ops Brief</h1>
        <p><b>Scenario:</b> Δ agents={extra_agents}, ACW {acw_red}%, FCR {fcr_up}%</p>
        <p>Baseline mean SLA (last day): {base_today:.1f}%</p>
        <p>Scenario mean SLA (last day): {scenario_today:.1f}%</p>
        <p>Delta: {delta_today:+.1f} percentage points</p>
        """
        path = export_ops_brief(html, REPORTS)
        st.success(f"Exported: {path}")