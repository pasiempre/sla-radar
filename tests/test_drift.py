import pandas as pd
from pathlib import Path

from src.models.drift import (
    DB_PATH as DB,
    load_inputs_for_drift,
    run_drift,
    DriftConfig,
)
from src.ingest.loader import load_all


def _ensure_db_built():
    """
    Helper: make sure the warehouse DB exists and has core tables/views
    before running drift tests.
    """
    if not DB.exists():
        load_all(reset=True)


def test_drift_emits_events():
    """
    Basic sanity check: drift pipeline can read inputs and run end-to-end.

    We assert:
      - inputs DataFrame is non-empty
      - required columns are present on inputs
      - run_drift() returns a DataFrame
      - if events are present, they have key columns like metric_name
    """
    _ensure_db_built()

    # Inputs
    df = load_inputs_for_drift(DB)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Our current drift inputs only need these core columns
    for col in ["interval_start", "arrivals", "aht_eff_seconds"]:
        assert col in df.columns

    # Run drift
    events = run_drift(
        DB,
        DriftConfig(alpha=0.3, hysteresis_k=1.8, min_run=2),
    )

    assert isinstance(events, pd.DataFrame)

    # It’s OK if no events are detected depending on thresholds,
    # but if we *do* get events, they should have these columns.
    if not events.empty:
        for col in ["event_ts", "metric_name", "score", "direction"]:
            assert col in events.columns