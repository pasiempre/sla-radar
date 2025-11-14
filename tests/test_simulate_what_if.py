# tests/test_simulate_what_if.py
import pytest

from src.scenarios.what_if import simulate_what_if, WhatIfConfig


@pytest.mark.usefixtures("db_path")
def test_simulate_what_if_baseline_runs(db_path):
    cfg = WhatIfConfig(
        acw_reduction_pct=0.0,
        fcr_uplift_pct=0.0,
        extra_agents=0,
        ot_hourly=0.0,
    )

    df = simulate_what_if(cfg, db=db_path)

    # Basic structural checks
    assert not df.empty
    for col in ("interval_start", "arrivals", "aht_eff_seconds", "agents", "sla_attainment_pct"):
        assert col in df.columns

    # SLA should always be between 0 and 100
    assert df["sla_attainment_pct"].between(0.0, 100.0).all()


@pytest.mark.usefixtures("db_path")
def test_simulate_what_if_extreme_staffing_changes_sla(db_path):
    """
    When we apply a very aggressive staffing change, SLA curves should change
    in at least some intervals – unless the system is already totally degenerate
    (all 0 or all 100).
    """
    base_cfg = WhatIfConfig(
        acw_reduction_pct=0.0,
        fcr_uplift_pct=0.0,
        extra_agents=0,
        ot_hourly=0.0,
    )
    base = simulate_what_if(base_cfg, db=db_path)

    low_cfg = WhatIfConfig(
        acw_reduction_pct=0.0,
        fcr_uplift_pct=0.0,
        extra_agents=-20,   # aggressively under-staff
        ot_hourly=0.0,
    )
    low = simulate_what_if(low_cfg, db=db_path)

    # If base SLA is non-degenerate, the curves should differ
    base_sla = base["sla_attainment_pct"]
    low_sla = low["sla_attainment_pct"]

    if base_sla.nunique() == 1 and base_sla.iloc[0] in (0.0, 100.0):
        # In degenerate case (everything 0 or 100), don't assert direction
        pytest.xfail("Base SLA is fully saturated (all 0 or all 100); cannot test direction safely.")
    else:
        assert not base_sla.equals(low_sla), "Expected SLA to change under extreme staffing shift"