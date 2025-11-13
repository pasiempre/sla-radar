# tests/test_drift.py
from pathlib import Path
from src.models.drift import load_inputs_for_drift, detect_drift, DriftConfig

def test_drift_emits_events():
    db = Path("data/warehouse.db")
    df = load_inputs_for_drift(db)
    assert len(df) > 0
    # lower thresholds so we're guaranteed to see something in synthetic incident
    cfg = DriftConfig(alpha=0.3, hysteresis_k=1.5, min_run=2)
    events = detect_drift(df, cfg)
    # We expect at least one event across aht_eff_seconds or arrivals for the bad-release window
    assert events.shape[0] >= 1
    assert set(events["metric_name"]).issubset({"aht_eff_seconds", "arrivals"})