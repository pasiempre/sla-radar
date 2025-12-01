# tests/test_forecast.py
from pathlib import Path
from src.models.forecast import load_model_inputs, next_day_arrivals_forecast, ForecastConfig

def test_forecast_basic():
    db = Path("data/warehouse.db")
    df = load_model_inputs(db)
    assert len(df) > 0
    fc = next_day_arrivals_forecast(db, ForecastConfig(horizon_intervals=16))
    assert len(fc) == 16
    assert (fc["arrivals_forecast"] >= 0).all()