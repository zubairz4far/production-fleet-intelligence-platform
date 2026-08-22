# Production Fleet Intelligence Platform

[![CI](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml)

Production-oriented machine-learning platform for fleet operations: leakage-safe vehicle failure prediction first, followed by telemetry anomaly detection, ETA prediction, geospatial optimization, streaming data, and monitored deployment.

## Status

**v0.2 — calibrated model promotion pipeline.**

The project now compares a balanced Logistic Regression baseline with a balanced histogram gradient-boosting candidate using a chronological **train → development → untouched test** protocol.

Model complexity alone does not justify promotion. Calibration and operating thresholds are fixed from the development window before the test set is evaluated.

## Measured v0.2 CI result

The deterministic synthetic CI fixture produced:

| Model | PR-AUC | ROC-AUC | Brier | Recall | Business cost |
|---|---:|---:|---:|---:|---:|
| Logistic Regression baseline | 0.373 | **0.884** | 0.140 | **0.696** | **63** |
| Calibrated HistGradientBoosting | **0.391** | 0.878 | **0.071** | 0.609 | 76 |

**Promotion decision: REJECT.**

The candidate improved PR-AUC and probability calibration, but it increased the configured business cost from **63 to 76**, so the promotion gate rejected it.

This is intentional evidence that the project does not equate a stronger model class or a better single metric with a production improvement.

> These scores come from a deterministic synthetic software-regression fixture. They are **not** evidence of real fleet performance.

Machine-readable evidence: [`evals/results/model_comparison_v0.2_synthetic.json`](evals/results/model_comparison_v0.2_synthetic.json).

## Why this project exists

Fleet ML is not one model. A useful production system eventually needs connected capabilities:

- vehicle failure-risk prediction
- telemetry anomaly detection
- ETA and delay prediction
- fuel-consumption forecasting
- geospatial route optimization
- batch and online inference
- drift and data-quality monitoring
- experiment/model tracking
- streaming telemetry ingestion

The repository is built around evaluation and operational trade-offs rather than demo accuracy.

## v0.2 architecture

```text
Fleet telemetry/history
        |
        v
schema + leakage checks
        |
        v
chronological split
        |
        +-------------------+
        |                   |
        v                   v
Logistic baseline     HistGradientBoosting
        |                   |
        |             Platt calibration
        |                   |
        +---------+---------+
                  |
                  v
       development threshold search
                  |
                  v
          untouched test window
                  |
                  v
      PR-AUC / Brier / business cost
                  |
                  v
           PROMOTE or REJECT
```

## Dataset contract

The failure-risk task expects one row per vehicle observation with at least:

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

Identifiers, timestamps, labels, and post-event maintenance fields are not model features.

## Evaluation discipline

Random row splitting is prohibited for reported promotion results.

v0.2 uses:

```text
oldest                                               newest
|---------------- train ----------------|--- dev ---|--- test ---|
```

- **train** fits model parameters;
- **development** fits Platt calibration and selects business-cost thresholds;
- **test** is evaluated after those choices are fixed.

Promotion currently requires all of:

1. candidate PR-AUC >= baseline PR-AUC;
2. candidate Brier score <= baseline Brier score;
3. candidate configured business cost <= baseline business cost.

See [`docs/evaluation-policy.md`](docs/evaluation-policy.md) and [`docs/model-promotion-v0.2.md`](docs/model-promotion-v0.2.md).

## Metrics

Accuracy is intentionally not a release metric. Reports include:

- PR-AUC
- ROC-AUC
- precision
- recall
- F1
- Brier score
- false positives / false negatives
- business-cost operating point
- development-selected threshold

The default example weights a false negative at 5× a false positive. This is configurable and is not presented as universal fleet economics.

## Model artifacts

The v0.2 comparison can persist:

```text
artifacts/models/
├── baseline_logistic.joblib
├── candidate_calibrated.joblib
└── metadata.json
```

The candidate artifact contains the fitted gradient-boosting model, Platt calibrator, selected threshold, and feature contract.

Generated artifacts are ignored by Git.

## Post-test interpretation

After the promotion decision, the pipeline computes permutation importance using average precision scoring. It is explicitly marked as **post-test analysis** and is not used for model selection.

On the synthetic CI fixture, the three largest average PR-AUC drops were associated with:

1. `hours_since_service`
2. `vibration_rms`
3. `oil_pressure_kpa`

Those values reflect only the synthetic generator and must not be interpreted as real-world fleet feature importance.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/generate_synthetic_fleet.py --output data/synthetic_fleet.csv

python -m fleet_intelligence.cli compare \
  --data data/synthetic_fleet.csv \
  --report artifacts/model_comparison_v0.2.json \
  --artifact-dir artifacts/models
```

Compatibility v0.1 baseline:

```bash
python -m fleet_intelligence.cli train \
  --data data/synthetic_fleet.csv \
  --report artifacts/baseline_report.json
```

Run quality gates:

```bash
ruff check .
pytest -q
```

## Progression

### v0.1 — failure-risk baseline ✅

- dataset/schema contract
- chronological holdout
- leakage-safe preprocessing
- balanced Logistic Regression
- business-cost threshold optimization
- reproducible JSON report
- tests + CI

### v0.2 — calibrated model promotion ✅

- chronological train/dev/test split
- HistGradientBoosting candidate
- balanced sample weighting
- Platt probability calibration
- development-only threshold selection
- untouched test promotion gate
- PR-AUC / Brier / business-cost criteria
- persisted model artifacts + metadata
- post-test permutation importance
- 9-test regression suite + CI

### v0.3 — telemetry and anomaly detection

- rolling/windowed telemetry features
- streaming-compatible feature transforms
- anomaly detection baseline + candidate
- feature-missingness robustness
- drift/data-quality reports
- vehicle-group temporal stress test

### v0.4 — ETA + geospatial optimization

- ETA/delay model
- geospatial feature pipeline
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

The final platform should demonstrate classical ML, time-series feature engineering, anomaly detection, geospatial ML, optimization, streaming/data engineering, MLOps, cloud deployment, and measurable operational trade-offs in one coherent system.

## License

MIT. Third-party datasets, models, libraries, and services retain their own licenses.
