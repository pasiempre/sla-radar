from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

@st.cache_data(show_spinner=False)
def compute_daily_modeled_sla(db: Path) -> pd.Series:
    """
    Run the baseline What-If model (no deltas) and return
    modeled mean SLA per calendar day.
    """
    cfg = WhatIfConfig(
        acw_reduction_pct=0.0,
        fcr_uplift_pct=0.0,
        extra_agents=0,
        ot_hourly=0.0,  # cost not needed for attribution
    )
    df = simulate_what_if(cfg, db=db)
    if df.empty:
        return pd.Series(dtype=float)

    df["day"] = df["interval_start"].dt.date
    daily_sla = df.groupby("day")["sla_attainment_pct"].mean()
    return daily_sla


st.set_page_config(page_title="Support Analytics Dashboard", layout="wide")
st.title("Support Analytics Dashboard")

tab_radar, tab_why, tab_whatif = st.tabs(["📡 Radar (Today)", "🔍 Why (Drift)", "🧮 What-If"])

# --- RADAR TAB ---
with tab_radar:
    st.subheader("📡 Radar: Today at a Glance")

    df_all = load_interval_inputs()
    if df_all.empty:
        st.info("No interval data available yet for the Radar view.")
    else:
        # Add calendar day for grouping
        df_all["day"] = df_all["interval_start"].dt.date
        days = sorted(df_all["day"].unique())

        if len(days) == 0:
            st.info("No dated interval data available for the Radar view.")
        else:
            today = days[-1]
            previous_days = [d for d in days if d < today]
            baseline_days = previous_days[-6:] if previous_days else []

            # --------------------------
            # 1) SLA Headline Strip
            # --------------------------
            st.markdown("### SLA Headline (modeled)")

            daily_sla = compute_daily_modeled_sla(DB)
            if daily_sla.empty or today not in daily_sla.index:
                st.info(
                    "Modeled daily SLA is not available yet; "
                    "headline SLA metrics will appear once data is populated."
                )
            else:
                sla_today = float(daily_sla.loc[today])

                # Yesterday (if available in modeled SLA series)
                delta_vs_yesterday = None
                if previous_days:
                    yesterday = previous_days[-1]
                    if yesterday in daily_sla.index:
                        sla_yesterday = float(daily_sla.loc[yesterday])
                        delta_vs_yesterday = sla_today - sla_yesterday

                # 7-day baseline (modeled SLA)
                baseline_sla_days = [d for d in baseline_days if d in daily_sla.index]
                delta_vs_baseline = None
                if baseline_sla_days:
                    sla_baseline = float(daily_sla.loc[baseline_sla_days].mean())
                    delta_vs_baseline = sla_today - sla_baseline
                else:
                    sla_baseline = None

                c1, c2, c3 = st.columns(3)

                # Today SLA
                if delta_vs_yesterday is not None:
                    c1.metric(
                        "SLA (today, modeled)",
                        f"{sla_today:.1f}%",
                        delta=f"{delta_vs_yesterday:+.1f} pts vs yesterday",
                    )
                else:
                    c1.metric("SLA (today, modeled)", f"{sla_today:.1f}%")

                # Vs 7-day baseline
                if delta_vs_baseline is not None and sla_baseline is not None:
                    c2.metric(
                        "Δ vs 7-day baseline",
                        f"{delta_vs_baseline:+.1f} pts",
                        help=f"Baseline modeled SLA: {sla_baseline:.1f}%",
                    )
                else:
                    c2.metric(
                        "Δ vs 7-day baseline",
                        "n/a",
                        help="Not enough modeled SLA history for a 7-day baseline.",
                    )

                # Target status
                if sla_today >= SLA_TARGET:
                    icon = "🟢"
                    status = "On target"
                elif sla_today >= SLA_TARGET - 5:
                    icon = "🟡"
                    status = "Slightly below target"
                else:
                    icon = "🔴"
                    status = "Off target"

                c3.metric(
                    "SLA target",
                    f"{SLA_TARGET:.0f}%",
                    help=f"{icon} {status} vs target.",
                )

            # --------------------------
            # 2) KPI Cards (today vs 7-day norm)
            # --------------------------
            st.markdown("### Key Ops KPIs (today vs 7-day norm)")

            df_today = df_all[df_all["day"] == today].copy()
            if df_today.empty:
                st.info("No intervals found for today yet.")
            else:
                baseline_df = (
                    df_all[df_all["day"].isin(baseline_days)].copy()
                    if baseline_days
                    else pd.DataFrame()
                )

                kpi_cols = st.columns(4)

                def render_kpi(idx, col_name, label, fmt="{:.0f}"):
                    col = kpi_cols[idx]
                    today_mean = df_today[col_name].mean()

                    if not baseline_df.empty and baseline_df[col_name].mean() > 0:
                        base_mean = baseline_df[col_name].mean()
                        pct_diff = (today_mean - base_mean) / base_mean
                        delta_str = f"{pct_diff:+.0%} vs 7-day norm"
                        col.metric(label, fmt.format(today_mean), delta=delta_str)
                    else:
                        col.metric(label, fmt.format(today_mean))

                render_kpi(0, "aht_eff_seconds", "AHT eff (sec)")
                render_kpi(1, "avg_acw_seconds", "ACW (sec)")
                render_kpi(2, "fcr_pct", "FCR (%)", fmt="{:.1f}")
                render_kpi(3, "arrivals", "Arrivals / 30m", fmt="{:.1f}")

            # --------------------------
            # 3) SLA Today by Interval
            # --------------------------
            st.markdown("### SLA Today by Interval")

            try:
                cfg_headline = WhatIfConfig(
                    acw_reduction_pct=0.0,
                    fcr_uplift_pct=0.0,
                    extra_agents=0,
                    ot_hourly=0.0,
                )
                sla_df = simulate_what_if(cfg_headline, db=DB)
            except Exception:
                sla_df = pd.DataFrame()

            if sla_df.empty:
                st.info(
                    "Could not compute interval-level modeled SLA for today yet."
                )
            else:
                sla_df["day"] = sla_df["interval_start"].dt.date
                today_sla = sla_df[sla_df["day"] == today].copy()

                if today_sla.empty:
                    st.info("No modeled SLA intervals found for today.")
                else:
                    plot_sla = today_sla.rename(
                        columns={"sla_attainment_pct": "SLA %"}
                    )

                    fig_sla = px.line(
                        plot_sla,
                        x="interval_start",
                        y="SLA %",
                        title="SLA Today by Interval (vs target)",
                        labels={"interval_start": "Interval start"},
                    )

                    # Add target line
                    fig_sla.add_trace(
                        go.Scatter(
                            x=plot_sla["interval_start"],
                            y=[SLA_TARGET] * len(plot_sla),
                            name=f"SLA target ({SLA_TARGET:.0f}%)",
                            mode="lines",
                            line=dict(dash="dash"),
                        )
                    )

                    fig_sla.update_layout(
                        hovermode="x unified",
                        margin=dict(l=40, r=40, t=60, b=40),
                    )

                    st.plotly_chart(fig_sla, use_container_width=True)

            # --------------------------
            # 4) Arrivals Forecast
            # --------------------------
            st.markdown("### Arrivals Forecast")

            try:
                fdf = next_day_arrivals_forecast(DB)
            except Exception:
                fdf = pd.DataFrame()

            if fdf is None or len(fdf) == 0:
                st.info("Arrivals forecast is not available yet.")
            else:
                fc_plot = fdf.reset_index().rename(
                    columns={"index": "interval_start"}
                )
                fig_fc = px.line(
                    fc_plot,
                    x="interval_start",
                    y="arrivals_forecast",
                    title="Arrivals forecast (next day, avg per 30m)",
                    labels={
                        "interval_start": "Forecast interval",
                        "arrivals_forecast": "Arrivals (forecast)",
                    },
                )
                fig_fc.update_layout(
                    hovermode="x unified",
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                st.caption(
                    "This shows the modeled arrivals for the next day. "
                    "In production, intraday peaks will highlight upcoming load risk."
                )

# --- WHY TAB ---
with tab_why:
    st.subheader("Drift Timeline: Metrics vs EWMA Score")

    # Load interval metrics and restrict to recent window
    df_why = load_interval_inputs().sort_values("interval_start")
    if df_why.empty:
        st.info("No interval data available to compute drift.")
    else:
        last_ts = df_why["interval_start"].max()
        cutoff = last_ts - pd.Timedelta(days=7)  # last 7 days
        ts = df_why[df_why["interval_start"] >= cutoff].copy()

        # Load drift events and align to same window
        events = run_drift(DB, DriftConfig(alpha=0.3, hysteresis_k=1.8, min_run=2))

        if events.empty:
            st.info("No sustained drift events detected with current thresholds.")
            events_window = pd.DataFrame()
        else:
            events_window = events.copy()
            events_window["event_ts"] = pd.to_datetime(events_window["event_ts"])
            events_window = events_window[events_window["event_ts"] >= cutoff]

        # --- Combined time-series: AHT / ACW / Arrivals vs Drift Score ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Primary axis: operational metrics
        fig.add_trace(
            go.Scatter(
                x=ts["interval_start"],
                y=ts["aht_eff_seconds"],
                name="AHT eff (sec)",
                mode="lines",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=ts["interval_start"],
                y=ts["avg_acw_seconds"],
                name="ACW (sec)",
                mode="lines",
                line=dict(dash="dot"),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=ts["interval_start"],
                y=ts["arrivals"],
                name="Arrivals",
                mode="lines",
                line=dict(dash="dash"),
            ),
            secondary_y=False,
        )

        # Secondary axis: drift score markers (if any)
        if not events_window.empty:
            fig.add_trace(
                go.Scatter(
                    x=events_window["event_ts"],
                    y=events_window["score"],
                    mode="markers",
                    name="Drift score",
                    marker=dict(size=8),
                    text=events_window["metric_name"] + " / " + events_window["direction"],
                    hovertemplate=(
                        "metric=%{text}"
                        "<br>score=%{y:.2f}"
                        "<br>time=%{x}"
                        "<extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

        fig.update_layout(
            title="Drift Timeline (Last 7 Days)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_yaxes(
            title_text="AHT / ACW (sec) & Arrivals",
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Drift score",
            secondary_y=True,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Optional: raw events table in an expander
        if not events_window.empty:
            with st.expander("Raw drift events (debug view)"):
                st.dataframe(
                    events_window.sort_values("event_ts"),
                    use_container_width=True,
                )

        # --- Today So Far – Early Warning ---
        st.subheader("Today So Far – Early Warning")

        # Add day + collect history
        df_why["day"] = df_why["interval_start"].dt.date
        days = sorted(df_why["day"].unique())

        if len(days) < 2:
            st.info("Not enough history to compute early warning baselines yet.")
        else:
            last_day = days[-1]
            previous_days = [d for d in days if d < last_day]
            baseline_days = previous_days[-6:] if previous_days else []

            if not baseline_days:
                st.info(
                    "Only one day of data available; early warning baselines will "
                    "appear once multiple days are loaded."
                )
            else:
                N = 4  # number of most recent intervals to inspect

                today_df = df_why[df_why["day"] == last_day].copy()
                if len(today_df) < N:
                    st.info(
                        "Not enough intervals so far today to compute early warnings."
                    )
                else:
                    # Prepare baseline + today, aligned by time-of-day
                    today_df = today_df.sort_values("interval_start")
                    today_df["time_of_day"] = today_df["interval_start"].dt.time

                    baseline = df_why[df_why["day"].isin(baseline_days)].copy()
                    baseline["time_of_day"] = baseline["interval_start"].dt.time

                    def summarize_metric(col, label, normal_tol=0.10, warn_tol=0.20):
                        """
                        Compare the last N intervals for `col` vs a 7-day baseline
                        at the same time-of-day, and return a human-readable sentence.
                        """
                        base_stats = (
                            baseline.groupby("time_of_day")[col]
                            .mean()
                            .rename("baseline_mean")
                            .reset_index()
                        )

                        recent = (
                            today_df.tail(N)
                            .merge(base_stats, on="time_of_day", how="left")
                        )

                        recent = recent[recent["baseline_mean"] > 0]
                        if recent.empty:
                            return (
                                f"ℹ️ **{label}**: not enough baseline data for "
                                "a reliable comparison."
                            )

                        actual = recent[col].mean()
                        base_mean = recent["baseline_mean"].mean()
                        if base_mean <= 0:
                            return (
                                f"ℹ️ **{label}**: baseline is zero; cannot compare yet."
                            )

                        dev_pct = (actual - base_mean) / base_mean
                        dev_abs = abs(dev_pct)
                        arrow = "above" if dev_pct > 0 else "below"

                        if dev_abs < normal_tol:
                            icon = "🟢"
                            msg = "within normal range"
                        elif dev_abs < warn_tol:
                            icon = "🟡"
                            msg = "slightly out of range"
                        else:
                            icon = "⚠️"
                            msg = "significantly out of range"

                        return (
                            f"{icon} **{label}** is {dev_pct:+.0%} {arrow} its 7-day norm "
                            f"over the last {N} intervals ({msg})."
                        )

                    # AHT & ACW messages
                    aht_msg = summarize_metric("aht_eff_seconds", "AHT")
                    acw_msg = summarize_metric("avg_acw_seconds", "ACW")

                    # Arrivals: slightly different thresholds & wording
                    def summarize_arrivals(normal_tol=0.15, warn_tol=0.30):
                        base_stats = (
                            baseline.groupby("time_of_day")["arrivals"]
                            .mean()
                            .rename("baseline_mean")
                            .reset_index()
                        )

                        recent = (
                            today_df.tail(N)
                            .merge(base_stats, on="time_of_day", how="left")
                        )
                        recent = recent[recent["baseline_mean"] > 0]
                        if recent.empty:
                            return (
                                "ℹ️ **Arrivals**: not enough baseline data for "
                                "a reliable comparison."
                            )

                        actual = recent["arrivals"].mean()
                        base_mean = recent["baseline_mean"].mean()
                        dev_pct = (actual - base_mean) / base_mean
                        dev_abs = abs(dev_pct)
                        arrow = "above" if dev_pct > 0 else "below"

                        if dev_abs < normal_tol:
                            icon = "🟢"
                            msg = "near expected demand"
                        elif dev_abs < warn_tol:
                            icon = "🟡"
                            msg = "moderately off forecast"
                        else:
                            icon = "⚠️"
                            msg = "demand surge" if dev_pct > 0 else "demand drop"

                        return (
                            f"{icon} **Arrivals** are {dev_pct:+.0%} {arrow} expected "
                            f"over the last {N} intervals ({msg})."
                        )

                    arrivals_msg = summarize_arrivals()

                    # Drift score trend from detected events
                    if events_window.empty:
                        drift_msg = (
                            "ℹ️ **Drift score**: no sustained drift events detected "
                            "in the last 7 days."
                        )
                    else:
                        recent_events = events_window.sort_values("event_ts").tail(5)
                        last_score = recent_events["score"].iloc[-1]
                        first_score = recent_events["score"].iloc[0]
                        delta_score = last_score - first_score

                        if last_score < 2.5:
                            icon = "🔵"
                            trend_text = "low and stable"
                        elif last_score < 3.5:
                            if delta_score >= 0:
                                icon = "🟡"
                                trend_text = "moderate and slightly rising"
                            else:
                                icon = "🟡"
                                trend_text = "moderate and easing"
                        else:
                            if delta_score > 0:
                                icon = "⚠️"
                                trend_text = "elevated and rising"
                            else:
                                icon = "🟠"
                                trend_text = "elevated but easing"

                        drift_msg = (
                            f"{icon} **Drift score** is {last_score:.1f} and "
                            f"{trend_text} over the last {len(recent_events)} events."
                        )

                    # Layout: 2 columns of text alerts
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(aht_msg)
                        st.markdown(acw_msg)
                    with c2:
                        st.markdown(arrivals_msg)
                        st.markdown(drift_msg)

        # --- 7-day Contribution Breakdown (Δ vs baseline) ---
        st.subheader("7-Day Contribution Breakdown (Δ vs baseline)")

        df_why["day"] = df_why["interval_start"].dt.date
        days = sorted(df_why["day"].unique())

        if len(days) < 2:
            st.info("Not enough days of data to compute a multi-day contribution breakdown.")
        else:
            last_day = days[-1]
            # Baseline = up to 6 previous days (or all previous if fewer)
            previous_days = [d for d in days if d < last_day]
            baseline_days = previous_days[-6:] if len(previous_days) > 0 else []

            if len(baseline_days) == 0:
                st.info("Only one day of data available; showing simple previous-vs-today view.")
                # Fallback: previous day vs today (if it exists)
                if len(days) >= 2:
                    prev_day = days[-2]
                    daily = (
                        df_why.groupby("day")[["avg_aht_seconds", "avg_acw_seconds", "arrivals"]]
                        .mean()
                    )
                    contrib = (
                        daily.loc[last_day] - daily.loc[prev_day]
                    ).rename(
                        {
                            "avg_aht_seconds": "ΔAHT",
                            "avg_acw_seconds": "ΔACW",
                            "arrivals": "ΔArrivals",
                        }
                    )
                else:
                    contrib = pd.Series(dtype=float)
            else:
                daily = (
                    df_why.groupby("day")[["avg_aht_seconds", "avg_acw_seconds", "arrivals"]]
                    .mean()
                )
                baseline_mean = daily.loc[baseline_days].mean()
                today_vals = daily.loc[last_day]

                contrib = (today_vals - baseline_mean).rename(
                    {
                        "avg_aht_seconds": "ΔAHT (vs baseline)",
                        "avg_acw_seconds": "ΔACW (vs baseline)",
                        "arrivals": "ΔArrivals (vs baseline)",
                    }
                )

            if contrib.empty:
                st.info("No contribution breakdown available yet.")
            else:
                # Waterfall-style contribution view: how today's averages differ from baseline
                components = list(contrib.index)
                values = contrib.values

                # Clean display labels (strip "(vs baseline)" if present)
                display_labels = [
                    name.replace("(vs baseline)", "").strip()
                    for name in components
                ]

                total_delta = float(contrib.sum())

                fig_contrib = go.Figure(
                    go.Waterfall(
                        name="Δ vs baseline",
                        orientation="v",
                        x=display_labels + ["Net Δ"],
                        measure=["relative"] * len(values) + ["total"],
                        y=list(values) + [total_delta],
                        text=[f"{v:+.1f}" for v in values] + [f"{total_delta:+.1f}"],
                        textposition="outside",
                    )
                )

                fig_contrib.update_layout(
                    title=f"7-Day Contribution Breakdown for {last_day}",
                    showlegend=False,
                    margin=dict(l=40, r=40, t=60, b=40),
                    yaxis_title="Δ vs baseline (today - 7-day baseline)",
                )

                st.plotly_chart(fig_contrib, use_container_width=True)

                # --- Estimated SLA impact (today vs 7-day baseline) ---
                try:
                    # We only do this when we have at least one baseline day
                    if len(baseline_days) > 0:
                        daily_sla = compute_daily_modeled_sla(DB)
                        if daily_sla.empty:
                            raise ValueError("No modeled SLA data available")

                        # Restrict baseline_days to days that exist in the modeled SLA series
                        baseline_sla = daily_sla[daily_sla.index.isin(baseline_days)]
                        if baseline_sla.empty or last_day not in daily_sla.index:
                            raise ValueError("Insufficient SLA history for attribution")

                        sla_today = float(daily_sla.loc[last_day])
                        sla_baseline = float(baseline_sla.mean())
                        delta_sla = sla_today - sla_baseline

                        abs_contrib = contrib.abs()
                        total_abs = float(abs_contrib.sum())

                        if total_abs > 0:
                            weights = abs_contrib / total_abs
                            sla_contrib = weights * delta_sla

                            st.markdown("**Estimated SLA impact vs 7-day baseline (last day)**")

                            # Nice, readable bullets
                            for name, val in sla_contrib.items():
                                label = name.replace("(vs baseline)", "").strip()
                                st.markdown(f"- **{label}**: {val:+.1f} pts")

                            st.markdown(
                                f"- **Net change in SLA**: {delta_sla:+.1f} pts "
                                f"(modeled SLA {sla_today:.1f}% vs baseline {sla_baseline:.1f}%)"
                            )
                        else:
                            st.caption(
                                "Estimated SLA impact: metric deltas are negligible vs baseline."
                            )
                except Exception:
                    # Keep this silent in UI; just avoid breaking the tab
                    st.caption(
                        "Estimated SLA impact could not be computed with the current data window."
                    )

        # --- Forecast Deviations (today vs 7-day band) ---
        st.subheader("Forecast Deviations (Today vs 7-day band)")

        # Need at least 2 days of data and at least 1 baseline day
        if len(days) < 2:
            st.info("Not enough days of data to build an expected band for today.")
        else:
            last_day = days[-1]
            previous_days = [d for d in days if d < last_day]
            baseline_days = previous_days[-6:] if len(previous_days) > 0 else []

            if len(baseline_days) == 0:
                st.info("Only one day of data available; cannot build a 7-day baseline band yet.")
            else:
                # Map UI labels to column names
                metric_options = {
                    "AHT eff (sec)": "aht_eff_seconds",
                    "ACW (sec)": "avg_acw_seconds",
                    "Arrivals": "arrivals",
                }
                metric_label = st.selectbox(
                    "Metric",
                    options=list(metric_options.keys()),
                    help="Compare today's intervals against a band built from the last 7 days at the same time-of-day.",
                )
                metric_col = metric_options[metric_label]

                # Add time-of-day so we can align intervals across days
                df_why["time_of_day"] = df_why["interval_start"].dt.time

                baseline = df_why[df_why["day"].isin(baseline_days)].copy()
                today_df = df_why[df_why["day"] == last_day].copy()

                if today_df.empty or baseline.empty:
                    st.info("Not enough data for today or baseline to compute forecast deviations.")
                else:
                    # Baseline mean/std by time-of-day
                    baseline_stats = (
                        baseline.groupby("time_of_day")[metric_col]
                        .agg(["mean", "std"])
                        .rename(columns={"mean": "metric_mean", "std": "metric_std"})
                        .reset_index()
                    )

                    # Join baseline stats onto today's intervals
                    today_df["time_of_day"] = today_df["interval_start"].dt.time
                    dev = today_df.merge(
                        baseline_stats,
                        on="time_of_day",
                        how="left",
                    )

                    # Handle cases with no variance / missing stats
                    dev["metric_std"] = dev["metric_std"].fillna(0.0)
                    dev["metric_mean"] = dev["metric_mean"].fillna(dev[metric_col].mean())

                    band_k = 2.0  # ~95% band
                    dev["lower"] = dev["metric_mean"] - band_k * dev["metric_std"]
                    dev["upper"] = dev["metric_mean"] + band_k * dev["metric_std"]

                    dev["is_anomaly"] = (dev[metric_col] < dev["lower"]) | (
                        dev[metric_col] > dev["upper"]
                    )

                    # Summary line
                    total_intervals = len(dev)
                    anomaly_count = int(dev["is_anomaly"].sum())
                    anomaly_pct = (anomaly_count / total_intervals * 100.0) if total_intervals > 0 else 0.0
                    st.caption(
                        f"Today: {anomaly_count} / {total_intervals} intervals "
                        f"({anomaly_pct:.1f}%) for **{metric_label}** were outside the 2σ band "
                        f"built from the last {len(baseline_days)} day(s)."
                    )

                    # Plot: band + actual + anomaly markers
                    fig_dev = go.Figure()

                    # Upper band
                    fig_dev.add_trace(
                        go.Scatter(
                            x=dev["interval_start"],
                            y=dev["upper"],
                            name="Upper band (mean + 2σ)",
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                    # Lower band (fill to previous)
                    fig_dev.add_trace(
                        go.Scatter(
                            x=dev["interval_start"],
                            y=dev["lower"],
                            name="Expected band",
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            opacity=0.2,
                            hoverinfo="skip",
                        )
                    )

                    # Actuals
                    fig_dev.add_trace(
                        go.Scatter(
                            x=dev["interval_start"],
                            y=dev[metric_col],
                            name="Actual",
                            mode="lines+markers",
                        )
                    )

                    # Anomaly markers
                    anomalies = dev[dev["is_anomaly"]]
                    if not anomalies.empty:
                        fig_dev.add_trace(
                            go.Scatter(
                                x=anomalies["interval_start"],
                                y=anomalies[metric_col],
                                name="Out-of-band",
                                mode="markers",
                                marker=dict(size=8, symbol="circle-open"),
                            )
                        )

                    fig_dev.update_layout(
                        title=f"{metric_label}: Today vs 7-day expected band",
                        hovermode="x unified",
                        margin=dict(l=40, r=40, t=60, b=40),
                        yaxis_title=metric_label,
                        xaxis_title="Interval start (today)",
                    )

                    st.plotly_chart(fig_dev, use_container_width=True)

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

    base_for_merge = base.rename(
        columns={
            "agents": "agents_base",
            "aht_eff_seconds": "aht_base",
            "arrivals": "arrivals_base",
        }
    )

    scenario_for_merge = out.rename(
        columns={
            "agents": "agents_scenario",
            "aht_eff_seconds": "aht_scenario",
            "arrivals": "arrivals_scenario",
        }
    )

    merged = base_for_merge[
        ["interval_start", "sla_base", "agents_base", "aht_base", "arrivals_base"]
    ].merge(
        scenario_for_merge[
            ["interval_start", "sla_scenario", "agents_scenario", "aht_scenario", "arrivals_scenario"]
        ],
        on="interval_start",
        how="inner",
    )

    # --- Focus on "today" only (last calendar date) ---
    merged["day"] = merged["interval_start"].dt.date
    today = merged["day"].max()

    # --- Derived capacity / utilization / risk (baseline & scenario) ---
    # Offered (we treat scenario arrivals as "offered" volume)
    merged["offered_scenario"] = merged["arrivals_scenario"]

    # Safe guards for divisions
    base_valid = (merged["agents_base"] > 0) & (merged["aht_base"] > 0)
    scen_valid = (merged["agents_scenario"] > 0) & (merged["aht_scenario"] > 0)

    # Initialize columns
    merged["util_base"] = 0.0
    merged["util_scenario"] = 0.0
    merged["capacity_scenario"] = 0.0

    # Baseline utilization (ρ_base)
    merged.loc[base_valid, "util_base"] = (
        (merged.loc[base_valid, "arrivals_base"] / 30.0)  # λ_base per minute
        / (
            (60.0 / merged.loc[base_valid, "aht_base"])     # μ_base per agent
            * merged.loc[base_valid, "agents_base"]         # total capacity in services/min
        )
    )

    # Scenario capacity and utilization (ρ_scenario)
    merged.loc[scen_valid, "capacity_scenario"] = (
        (60.0 / merged.loc[scen_valid, "aht_scenario"])    # μ_scenario per agent
        * merged.loc[scen_valid, "agents_scenario"]
        * 30.0                                             # 30-minute window
    )

    merged.loc[scen_valid, "util_scenario"] = (
        (merged.loc[scen_valid, "arrivals_scenario"] / 30.0)  # λ_scenario per minute
        / (
            (60.0 / merged.loc[scen_valid, "aht_scenario"])
            * merged.loc[scen_valid, "agents_scenario"]
        )
    )

    # Risk index (0–100) based on scenario utilization
    rho_safe = 0.85
    rho_max = 1.15
    raw_risk = (merged["util_scenario"] - rho_safe) / (rho_max - rho_safe)
    merged["risk_index"] = raw_risk.clip(lower=0.0, upper=1.0) * 100.0

    # Now slice to today
    today_df = merged[merged["day"] == today].copy()

    # --- Delta cards (today baseline vs scenario) ---
    base_today = today_df["sla_base"].mean()
    scenario_today = today_df["sla_scenario"].mean()
    delta_today = scenario_today - base_today

    # --- At-risk intervals mask (scenario vs SLA target) ---
    # Any interval where scenario SLA drops below the target is considered "at risk".
    at_risk = today_df[
    (today_df["sla_base"] < SLA_TARGET) | (today_df["sla_scenario"] < SLA_TARGET)].copy()

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

    # --- Utilization cards (today) ---
    util_base_today = today_df["util_base"].mean() * 100.0
    util_scen_today = today_df["util_scenario"].mean() * 100.0
    peak_util_scen = today_df["util_scenario"].max() * 100.0

    u1, u2, u3 = st.columns(3)
    u1.metric(
        "Avg Util (baseline, today)",
        f"{util_base_today:.1f}%",
    )
    u2.metric(
        "Avg Util (scenario, today)",
        f"{util_scen_today:.1f}%",
        help="Average scenario utilization across today's intervals.",
    )
    u3.metric(
        "Peak Util (scenario, today)",
        f"{peak_util_scen:.1f}%",
        help="Highest scenario utilization across today's intervals.",
    )

    st.subheader("Preview of Projected Data (Baseline vs Scenario)")
    st.dataframe(
        merged[
            [
                "interval_start",
                "agents_base",
                "agents_scenario",
                "aht_base",
                "aht_scenario",
                "sla_base",
                "sla_scenario",
            ]
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

    # --- Load vs Capacity (scenario, today) ---
    st.subheader("Load vs Capacity (scenario, today)")

    load_view = today_df[
        [
            "interval_start",
            "offered_scenario",
            "capacity_scenario",
            "util_scenario",
            "risk_index",
        ]
    ].copy()

    load_view = load_view.rename(
        columns={
            "offered_scenario": "Offered",
            "capacity_scenario": "Capacity (30m)",
            "util_scenario": "Utilization",
            "risk_index": "Risk index",
        }
    )

    # Convert utilization from fraction to %
    load_view["Utilization"] = (load_view["Utilization"] * 100.0).round(1)
    load_view["Capacity (30m)"] = load_view["Capacity (30m)"].round(1)
    load_view["Risk index"] = load_view["Risk index"].round(1)

    st.dataframe(
        load_view.sort_values("interval_start"),
        use_container_width=True,
    )

    # --- At-risk intervals (scenario view) ---
    if at_risk.empty:
        st.success("No intervals below the SLA target under this scenario. ✅")
    else:
        # Keep only a few key columns for readability
        at_risk_view = at_risk[
            [
                "interval_start",
                "agents_base",
                "agents_scenario",
                "aht_base",
                "aht_scenario",
                "sla_base",
                "sla_scenario",
                "risk_index",
            ]
        ].sort_values("interval_start")

        at_risk_view = at_risk_view.rename(
            columns={
                "agents_base": "Agents (base)",
                "agents_scenario": "Agents (scenario)",
                "aht_base": "AHT base (sec)",
                "aht_scenario": "AHT scenario (sec)",
                "sla_base": "SLA base (%)",
                "sla_scenario": "SLA scenario (%)",
                "risk_index": "Risk index",
            }
        )

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