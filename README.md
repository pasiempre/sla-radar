

# SLA Performance & Drift Radar  
A lightweight operational analytics toolkit that detects SLA drift, explains root causes, and simulates corrective scenarios.

## 🚀 Overview
SLA Radar is a full mini-analytics stack designed for Support Operations and Workforce Management teams who need a clear view of:  
- How SLA is trending  
- Why operational drift is happening  
- What interventions (staffing, AHT/ACW, FCR) would recover SLA  
- How much improvement each lever produces  

It includes:  
- **ETL pipeline** → loads raw interval data into a SQLite “warehouse”  
- **Modeling layer** → drift detection, feature attribution, scenario simulation  
- **Interactive dashboard** → Streamlit app with 3 core tabs  
- **Full test suite** → pytest-based validation of ingest, models, and drift logic  

---

## 📂 Project Structure

```
sla-radar/
│
├── src/
│   ├── ingest/
│   │   ├── loader.py        # ETL: loads CSV → warehouse.db
│   │   └── ...
│   ├── models/
│   │   ├── drift.py         # Drift detection, EWMA scores
│   │   ├── what_if.py       # Scenario simulation engine
│   │   └── erlang/          # Erlang C core + queue math
│   ├── dashboard/
│   │   └── app.py           # Streamlit interface
│   └── utils/
│
├── data/
│   ├── raw/                 # Input CSVs
│   └── warehouse.db         # Auto-generated database
│
├── tests/
│   ├── test_drift.py
│   ├── test_erlangc.py
│   ├── test_ingest_views.py
│   ├── test_what_if.py
│   └── ...
│
├── sql/
│   └── views.sql            # Logical warehouse views
│
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## ⚙️ ETL Pipeline

The ETL flow is intentionally simple and robust:

1. Raw CSVs are placed in `data/raw/`
2. Run:

```
python -m src.ingest.loader --reset
```

3. Loader builds a fresh SQLite warehouse with:
   - **interval_inputs**
   - **sla_policy**
   - **metrics history**
   - **view definitions** from `sql/views.sql`

4. All downstream components use only SQL views—never raw tables.

ETL is stable and complete for current scope.

---

## 🔍 Drift Detection

The drift engine (in `models/drift.py`) produces:

- **EWMA drift score**
- **Metric-level contribution**
- **Drift timeline**
- **Past 7-day baseline comparison**
- **Daily aggregates**

Drift attribution works by calculating *how much each metric contributed* to SLA degradation.

Example outputs:
- “AHT contributed −6.8 points”
- “ACW contributed −1.2 points”
- “Arrivals contributed +0.4 points (positive impact)”

---

## 🎛️ Scenario Simulation (“What If” Engine)

Powered by `models/what_if.py`, the simulation engine allows users to test changes in:

- Staffing (agents)
- AHT (efficiency)
- ACW
- FCR
- Combined operational levers

Outputs include:
- Forecasted SLA under the scenario  
- Utilization  
- Load vs capacity  
- Risk index  
- Delta vs baseline  

This is used by the dashboard’s “What If” tab.

---

## 📊 Dashboard (Streamlit)

Launch with:

```
streamlit run src/dashboard/app.py
```

### Tabs:

#### 1️⃣ **SLA Performance Radar**
Includes:
- Combined multi-metric time-series (AHT, ACW, Arrivals, Drift)
- Forecast deviation analysis
- 7‑day SLA waterfall
- At-a-glance SLA KPIs

#### 2️⃣ **Why Drift? (Root Cause Analysis)**
Includes:
- Contribution analysis (AHT / ACW / Arrivals)
- Past-day drift timeline
- Early warning panel (“Today So Far”)
- Drift event detection
- Deviation flags and anomaly markers

#### 3️⃣ **Scenario Simulator (What If)**
Includes:
- Extra agents slider
- AHT/ACW change inputs
- FCR improvements
- Daily delta cards
- At-risk‑intervals table
- Load vs capacity visualization

---

## 🧪 Test Suite

Run all tests:

```
pytest
```

Covers:
- ETL correctness
- SQL views
- Drift engine
- Erlang C queueing logic
- Scenario simulation math

All current tests are passing.

---

## 📦 Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🛠 Roadmap

### Completed
- ETL pipeline  
- Drift engine  
- Scenario simulator  
- Radar tab overhaul  
- Forecast deviations  
- Waterfall chart  
- Warning panel  

### Upcoming Enhancements
- KPI glossary  
- Auto-detection of outlier days  
- Multi-day trend reports  
- Advanced risk scoring  
- SLA sensitivity analysis  
- Full documentation site  

---

## 📄 License
MIT License (or TBD based on project direction)

---

## 🙌 Acknowledgements
This project blends queueing theory, operational science, and modern analytics UX patterns into a compact engineering toolkit. Built to support real-world support operations and staffing strategy.