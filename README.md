# SLA Performance & Drift Radar

A production-ready operational analytics toolkit for call center SLA monitoring, drift detection, staffing optimization, and scenario simulation.

## 🚀 Overview

SLA Radar is a comprehensive analytics stack designed for Support Operations and Workforce Management teams who need:

- **Real-time SLA health monitoring** with visual gauges and alerts
- **Drift detection** using EWMA statistical methods with hysteresis
- **Staffing recommendations** based on Erlang C queueing theory
- **Scenario simulation** to model operational changes before implementation
- **Cost impact analysis** for budget planning and optimization

### Core Components

| Component | Description |
|-----------|-------------|
| **ETL Pipeline** | Loads CSV data into SQLite warehouse with SQL views |
| **Erlang C Engine** | Queue theory calculations for SLA predictions |
| **Drift Detection** | EWMA-based anomaly detection with attribution |
| **Scenario Simulator** | What-if analysis for staffing/efficiency changes |
| **Streamlit Dashboard** | Interactive 3-tab interface with production features |

### Key Features

- 📊 **SLA Health Gauge** - Real-time visual SLA status with configurable thresholds
- 👥 **Staffing Recommendations** - Erlang C-based agent optimization per interval
- ⚠️ **Interval Alerts** - Proactive warnings for at-risk periods
- 📞 **Channel Mix Breakdown** - Distribution analysis across contact channels
- 💰 **Cost Impact Calculator** - Hourly cost projections for staffing changes

---

## 📂 Project Structure

```
sla-radar/
├── src/
│   ├── dashboard/
│   │   └── app.py              # Streamlit interface (3 tabs + features)
│   ├── ingest/
│   │   └── loader.py           # ETL: CSV → SQLite warehouse
│   ├── models/
│   │   ├── drift.py            # EWMA drift detection & attribution
│   │   ├── erlangc.py          # Erlang C queue theory calculations
│   │   ├── forecast.py         # Time-series forecasting
│   │   ├── run_drift.py        # Drift analysis runner
│   │   └── run_forecast.py     # Forecast runner
│   ├── reports/
│   │   └── pdf_export.py       # PDF report generation
│   └── scenarios/
│       └── what_if.py          # Scenario simulation engine
│
├── data/
│   ├── seed/                   # Input CSV files
│   │   ├── interval_load.csv   # 30-min interval data
│   │   ├── agent_schedule.csv  # Agent availability
│   │   ├── sla_policy.csv      # SLA targets
│   │   ├── ops_cost.csv        # Operational costs
│   │   └── config.yml          # Configuration
│   ├── reports/                # Generated reports
│   └── warehouse.db            # Auto-generated SQLite database
│
├── sql/
│   └── views.sql               # SQL view definitions
│
├── tests/                      # pytest test suite
│   ├── test_drift.py
│   ├── test_erlang_core.py
│   ├── test_erlangc.py
│   ├── test_forecast.py
│   ├── test_ingest_views.py
│   ├── test_schemas.py
│   └── test_simulate_what_if.py
│
├── pyproject.toml
├── pytest.ini
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-org/sla-radar.git
cd sla-radar

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- Key dependencies: streamlit, plotly, pandas, numpy, scipy, statsmodels

---

## 🚀 Quick Start

### 1. Load Data (ETL)

```bash
python -m src.ingest.loader --reset
```

This builds the SQLite warehouse from CSV files in `data/seed/`.

### 2. Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 📊 Dashboard Features

### Tab 1: SLA Performance Radar

| Feature | Description |
|---------|-------------|
| **SLA Health Gauge** | Real-time circular gauge showing current SLA percentage |
| **Multi-metric Chart** | Combined view of AHT, ACW, Arrivals, Drift over time |
| **7-Day SLA Waterfall** | Daily contribution breakdown |
| **Channel Mix Donut** | Contact distribution by channel |
| **KPI Cards** | At-a-glance operational metrics |

### Tab 2: Why Drift? (Root Cause Analysis)

| Feature | Description |
|---------|-------------|
| **Contribution Analysis** | Metric-level attribution (AHT, ACW, Arrivals) |
| **Drift Timeline** | Temporal view of drift events |
| **Early Warning Panel** | "Today So Far" alerts |
| **Staffing Recommendations** | Erlang C-based agent suggestions |
| **Interval Alerts** | At-risk period warnings |

### Tab 3: Scenario Simulator (What If)

| Feature | Description |
|---------|-------------|
| **Agent Slider** | Simulate additional staffing |
| **AHT/ACW Inputs** | Model efficiency improvements |
| **FCR Adjustments** | First-call resolution changes |
| **Cost Calculator** | Hourly cost impact projections |
| **Delta Cards** | Before/after comparison |
| **At-Risk Table** | Intervals requiring attention |

### Sidebar Configuration

- **SLA Threshold** - Warning/critical levels (default: 80%)
- **Alert Sensitivity** - Minimum gap to trigger alerts
- **Cost Per Agent Hour** - For cost calculations
- **Auto-Refresh** - Enable/disable periodic data refresh

---

## 🔍 Drift Detection Engine

The drift engine (`src/models/drift.py`) uses EWMA (Exponentially Weighted Moving Average) with hysteresis for robust anomaly detection:

### How It Works

1. **EWMA Calculation** - Smooths metric values over configurable spans
2. **Baseline Comparison** - Compares current vs 7-day rolling baseline
3. **Hysteresis Control** - Prevents alert flapping with enter/exit thresholds
4. **Attribution** - Calculates each metric's contribution to SLA drift

### Output Metrics

| Metric | Description |
|--------|-------------|
| `ewma_score` | Smoothed drift indicator |
| `metric_contribution` | Per-metric attribution |
| `drift_direction` | Improving/degrading trend |
| `alert_state` | Current hysteresis state |

### Example Attribution Output

```
AHT contributed −6.8 points (degrading)
ACW contributed −1.2 points (degrading)
Arrivals contributed +0.4 points (positive impact)
```

---

## 🎛️ Scenario Simulation (What-If Engine)

The simulation engine (`src/scenarios/what_if.py`) enables testing operational changes before implementation:

### Adjustable Levers

| Lever | Description | Impact |
|-------|-------------|--------|
| **Staffing** | Add/remove agents | Direct capacity change |
| **AHT** | Handle time efficiency | Throughput improvement |
| **ACW** | After-call work | Agent availability |
| **FCR** | First-call resolution | Reduces repeat contacts |

### Simulation Outputs

- **Forecasted SLA** - Projected SLA under scenario
- **Utilization** - Agent occupancy rate
- **Load vs Capacity** - Demand/supply ratio
- **Risk Index** - Probability of SLA miss
- **Cost Delta** - Financial impact of changes

---

## 📐 Erlang C Queue Theory

The Erlang C module (`src/models/erlangc.py`) provides mathematically-sound staffing calculations:

```python
from src.models.erlangc import estimate_sla_attainment

sla = estimate_sla_attainment(
    arrivals=120,        # Calls per interval
    aht_seconds=300,     # Average handle time
    agents=15,           # Available agents
    target_seconds=20    # SLA target (answer in X seconds)
)
# Returns: 0.85 (85% predicted SLA)
```

### Key Functions

- `erlang_c_prob()` - Probability of waiting
- `expected_wait_time()` - Average queue time
- `estimate_sla_attainment()` - SLA percentage prediction
- `required_agents()` - Agents needed for target SLA

---

## 🧪 Testing

Run the full test suite:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

### Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_drift.py` | Drift detection & EWMA |
| `test_erlang_core.py` | Core queue math |
| `test_erlangc.py` | Erlang C integration |
| `test_forecast.py` | Time-series forecasting |
| `test_ingest_views.py` | ETL & SQL views |
| `test_schemas.py` | Data validation |
| `test_simulate_what_if.py` | Scenario simulation |

Current status: **10 tests passing, 1 xfailed (expected)**

---

## 📋 Data Model

### Interval Data (`interval_load.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `interval_start` | datetime | Start of 30-min interval |
| `arrivals` | int | Inbound contact count |
| `aht_seconds` | float | Average handle time |
| `acw_seconds` | float | After-call work time |
| `fcr` | float | First-call resolution rate |
| `channel` | string | Contact channel (voice, chat, email) |

### Agent Schedule (`agent_schedule.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `interval_start` | datetime | Start of 30-min interval |
| `agents_available` | int | Scheduled agents |

### SLA Policy (`sla_policy.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `target_pct` | float | SLA target percentage |
| `target_seconds` | int | Answer time threshold |

---

## 🔧 Production Features

The dashboard includes production-ready enhancements:

- **Logging** - Structured logging for debugging and monitoring
- **Data Freshness** - Automated staleness detection with warnings
- **Configurable Thresholds** - Sidebar controls for runtime adjustment
- **TTL Caching** - Efficient data caching with 5-minute TTL
- **Auto-Refresh** - Optional periodic data updates
- **Error Handling** - Graceful degradation with user-friendly messages

---

## 🛠️ Roadmap

### ✅ Completed

- ETL pipeline with SQLite warehouse
- EWMA drift detection with hysteresis
- Erlang C staffing calculations
- Scenario simulator (What-If engine)
- SLA Health Gauge
- Staffing Recommendations
- Interval Alerts
- Channel Mix Breakdown
- Cost Impact Calculator
- Production hardening (logging, caching, config)

### 🔜 Upcoming

- KPI glossary/tooltips
- Multi-day trend reports
- Advanced risk scoring
- SLA sensitivity analysis
- PDF report export
- API endpoints for integration

---

## 📄 License
cd
MIT License

---

## 🙌 Acknowledgements

This project combines Erlang C queueing theory, EWMA statistical methods, and modern analytics UX patterns into a production-ready operational toolkit. Built to support real-world call center operations and workforce management strategy.
