# Production Fleet Intelligence Platform

Production-oriented machine-learning platform for fleet operations: leakage-safe vehicle failure prediction first, followed by ETA prediction, fuel forecasting, geospatial optimization, streaming telemetry, and monitored deployment.

## Status

**v0.1 — reliability baseline.** The first milestone intentionally starts with one measurable problem: predict whether a vehicle will require maintenance within a future horizon without leaking future telemetry into training.

The repository is being built around reproducible evaluation rather than demo accuracy. Model promotion will require a frozen test set, business-aware thresholds, calibration checks, and regression gates.

## Why this project exists

Fleet ML is not one model. A useful production system eventually needs several connected capabilities:

- vehicle failure-risk prediction
- ETA and delay prediction
- fuel-consumption forecasting
- anomaly detection over telemetry
- geospatial route optimization
- batch and online inference
- drift and data-quality monitoring
- experiment/model tracking
- streaming telemetry ingestion

The first release establishes the evaluation discipline and training contract that later capabilities will reuse.

## v0.1 architecture

```text
CSV telemetry/history
        |
        v
schema + leakage checks
        |
        v
time-aware split
        |
        v
feature pipeline
        |
        v
Logistic Regression baseline
        |
        v
probability calibration metrics
        |
        v
business-cost threshold search
        |
        v
JSON benchmark report
```

## Dataset contract

The baseline expects one row per vehicle observation with at least:

| Column | Type | Meaning |
|---|---|---|
| `vehicle_id` | string | Stable vehicle identifier |
| `timestamp` | datetime | Observation timestamp |
| `mileage_km` | numeric | Odometer reading |
| `engine_temp_c` | numeric | Engine temperature |
| `oil_pressure_kpa` | numeric | Oil pressure |
| `battery_voltage` | numeric | Battery voltage |
| `vibration_rms` | numeric | Vibration signal summary |
| `fuel_rate_lph` | numeric | Fuel consumption rate |
| `hours_since_service` | numeric | Operating hours since last service |
| `failure_within_horizon` | 0/1 | Label known only after the prediction horizon |

A real fleet dataset may contain additional fields. The baseline deliberately avoids identifiers and post-event fields as model features.

## Leakage policy

Random row splitting is not allowed for the benchmark. Observations are sorted by `timestamp`; the newest partition is held out. This reduces temporal leakage and better approximates future deployment.

The following are never model features:

- `vehicle_id`
- `timestamp`
- target labels
- maintenance outcome fields created after the prediction time

Future versions will add group-aware temporal evaluation so repeated observations from the same vehicle can be stress-tested separately.

## Metrics

Accuracy is not a release metric because maintenance failures are typically imbalanced. The benchmark records:

- PR-AUC
- ROC-AUC
- precision
- recall
- F1
- Brier score
- false positives / false negatives
- threshold selected by business cost

The default threshold objective makes a false negative five times more expensive than a false positive. This is configurable and is not presented as a universal fleet cost model.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/generate_synthetic_fleet.py --output data/synthetic_fleet.csv
python -m fleet_intelligence.cli train \
  --data data/synthetic_fleet.csv \
  --report artifacts/baseline_report.json
```

Run tests:

```bash
pytest -q
ruff check .
```

## Planned progression

### v0.1 — failure-risk baseline

- dataset/schema contract
- time-aware split
- leakage-safe preprocessing
- Logistic Regression baseline
- business-cost threshold optimization
- reproducible JSON evaluation report
- tests + CI

### v0.2 — stronger tabular ML

- gradient-boosted candidate model
- probability calibration
- frozen holdout promotion gate
- SHAP/error analysis
- model artifact metadata

### v0.3 — telemetry and anomaly detection

- rolling/windowed features
- streaming-compatible feature transforms
- anomaly detector
- drift/data-quality reports

### v0.4 — ETA + geospatial optimization

- ETA/delay model
- route graph representation
- OR-Tools constrained routing benchmark

### v0.5 — production MLOps integration

- MLflow tracking/registry
- feature-store integration
- FastAPI online inference
- Kafka telemetry path
- Prometheus metrics
- container/Kubernetes deployment
- Terraform cloud stack

## Portfolio target

The final platform should demonstrate classical ML, time-series features, anomaly detection, geospatial ML, optimization, streaming/data engineering, MLOps, cloud deployment, and measurable operational trade-offs in one coherent system.

## License

MIT. Third-party datasets, models, libraries, and services retain their own licenses.