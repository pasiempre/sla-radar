from pathlib import Path
from .forecast import next_day_arrivals_forecast, ForecastConfig

if __name__ == "__main__":
    df = next_day_arrivals_forecast(Path("data/warehouse.db"), ForecastConfig())
    print(df.head(20).to_string())
