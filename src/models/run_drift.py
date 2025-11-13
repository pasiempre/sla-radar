from pathlib import Path
from .drift import run_drift, DriftConfig

if __name__ == "__main__":
    events = run_drift(Path("data/warehouse.db"), DriftConfig())
    print(events.tail(10).to_string(index=False))
