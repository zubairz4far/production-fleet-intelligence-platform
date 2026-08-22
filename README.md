# Production Fleet Intelligence Platform

[![CI](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml)

Production-oriented ML platform for fleet operations: leakage-safe failure prediction, causal telemetry anomaly detection, robustness evaluation, and a planned path to ETA/geospatial optimization, streaming, and production MLOps.

## Status

**v0.3 — causal telemetry anomaly detection + robustness evaluation.**

The repository now contains two evaluated ML tracks:

1. **vehicle failure-risk prediction** — Logistic Regression baseline versus calibrated histogram gradient boosting;
2. **telemetry anomaly detection** — robust statistical detector versus Isolation Forest using causal rolling features.

The project is built around explicit train/development/test separation, operational thresholds, reproducible reports, and model-promotion gates rather than demo accuracy.

> All headline numbers below come from deterministic synthetic CI fixtures. They validate software and evaluation behavior only. They are **not evidence of real fleet performance**.

## Measured v0.3 anomaly CI result

Synthetic anomaly fixture: **1,200 observations, 60 vehicles, 4.83% labeled anomalies**. The first 60% of the timeline is anomaly-free for unsupervised fitting; anomalies are injected only into development/test windows.

| Detector | PR-AUC | ROC-AUC | Recall | False negatives | Business cost |
|---|---:|---:|---:|---:|---:|
| Robust z-score baseline | 0.197 | **0.688** | 0.931 | 2 | 155 |
| Isolation Forest candidate | **0.210** | 0.641 | **0.966** | **1** | **153** |

**Predeclared v0.3 promotion decision: PROMOTE.**

The Isolation Forest candidate satisfied all criteria declared before the test run:

- PR-AUC did not regress;
- configured business cost did not regress;
- 10% missing-sensor-family stress produced less than 0.10 absolute PR-AUC drop.

### Robustness result

With 10% of each sensor family masked at test time:

| Metric | Clean | Masked |
|---|---:|---:|
| PR-AUC | 0.210 | 0.201 |
| Recall | 0.966 | 0.793 |
| Business cost | 153 | 177 |

The PR-AUC drop was only **0.009**, so the predeclared robustness criterion passed. However, recall dropped by **0.172** and business cost increased to **177**. That degradation is retained as a warning. The promotion rule is not changed after observing the test set; a stricter recall/cost robustness gate would require a new untouched holdout.

### Unseen-vehicle stress test

Every fifth vehicle identity was excluded from model fitting. On the synthetic held-out-vehicle test slice:

- 12 held-out vehicles
- 48 test observations
- PR-AUC: **0.286**
- ROC-AUC: **0.791**
- recall: **1.000**
- false negatives: **0**

The model does not see held-out vehicle identities during fitting. Their own prior telemetry remains available for causal rolling features, matching a deployment where a newly onboarded vehicle accumulates history over time.

### Drift check

Population Stability Index (PSI) compares the training distribution with the chronological test window and a deterministic simulated-drift scenario.

- clean maximum PSI: **0.456**
- simulated maximum PSI: **5.520**
- high-drift threshold: **0.25**

The clean synthetic test window already shows high fuel-rate PSI because the generator contains a temporal mileage/fuel trend. The simulated scenario produces strong shift across engine temperature, oil pressure, battery voltage, vibration, and fuel rate. PSI is treated as a distribution warning, not proof of causal model degradation.

Machine-readable evidence: [`evals/results/anomaly_detection_v0.3_synthetic.json`](evals/results/anomaly_detection_v0.3_synthetic.json).

Protocol: [`docs/telemetry-anomaly-v0.3.md`](docs/telemetry-anomaly-v0.3.md).

## Previous v0.2 failure-risk result

| Model | PR-AUC | ROC-AUC | Brier | Recall | Business cost |
|---|---:|---:|---:|---:|---:|
| Logistic Regression baseline | 0.373 | **0.884** | 0.140 | **0.696** | **63** |
| Calibrated HistGradientBoosting | **0.391** | 0.878 | **0.071** | 0.609 | 76 |

**v0.2 promotion decision: REJECT.**

The stronger classifier improved PR-AUC and calibration but worsened configured business cost, so it was not promoted.

Machine-readable evidence: [`evals/results/model_comparison_v0.2_synthetic.json`](evals/results/model_comparison_v0.2_synthetic.json).

## v0.3 architecture

```text
Fleet telemetry/history
        |
        v
schema + chronology validation
        |
        v
per-vehicle causal features
shift(1) -> rolling 3/6 windows
        |
        v
chronological train / dev / test
        |
        +-------------------------+
        |                         |
        v                         v
Robust z-score              Isolation Forest
baseline                    candidate
        |                         |
        +------------+------------+
                     |
                     v
          development calibration
          + threshold selection
                     |
                     v
             untouched test
                     |
      +--------------+---------------+
      |              |               |
      v              v               v
 model gate     missing sensors   unseen vehicles
                                      |
                                      v
                              robustness report
                     |
                     v
               PSI drift checks
                     |
                     v
        PROMOTE / REJECT + artifacts
```

## Causal telemetry features

For each of five telemetry signals:

- current value
- lag-1 value
- past 3-observation mean/std
- current minus past-3 mean
- past 6-observation mean/std

Mileage and hours since service are included as operating context, producing **37 anomaly features**.

Historical windows use `shift(1)` before rolling. Future observations cannot enter earlier feature rows. A regression test mutates a future observation and verifies that an earlier row remains unchanged.

## Dataset contracts

Failure-risk data requires:

| Column | Meaning |
|---|---|
| `vehicle_id` | stable vehicle identifier |
| `timestamp` | observation timestamp |
| `mileage_km` | odometer reading |
| `engine_temp_c` | engine temperature |
| `oil_pressure_kpa` | oil pressure |
| `battery_voltage` | battery voltage |
| `vibration_rms` | vibration summary |
| `fuel_rate_lph` | fuel consumption rate |
| `hours_since_service` | operating hours since service |
| `failure_within_horizon` | future failure label |

The anomaly benchmark adds `telemetry_anomaly` as an evaluation label. That label is never part of the detector feature matrix.

## Evaluation discipline

Reported model-selection experiments use chronological partitions:

```text
oldest                                               newest
|---------------- train ----------------|--- dev ---|--- test ---|
```

- **train** fits model parameters;
- **development** fits score/probability calibration and operating thresholds;
- **test** is evaluated only after those choices are fixed.

Random row splitting is prohibited for reported promotion results.

See:

- [`docs/evaluation-policy.md`](docs/evaluation-policy.md)
- [`docs/model-promotion-v0.2.md`](docs/model-promotion-v0.2.md)
- [`docs/telemetry-anomaly-v0.3.md`](docs/telemetry-anomaly-v0.3.md)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the v0.3 synthetic anomaly pipeline:

```bash
python scripts/generate_synthetic_fleet.py \
  --output data/synthetic_fleet_anomaly.csv \
  --with-anomalies

python -m fleet_intelligence.cli anomaly \
  --data data/synthetic_fleet_anomaly.csv \
  --report artifacts/anomaly_report_v0.3.json \
  --artifact-dir artifacts/anomaly-model
```

Run the v0.2 failure-risk comparison:

```bash
python scripts/generate_synthetic_fleet.py --output data/synthetic_fleet.csv

python -m fleet_intelligence.cli compare \
  --data data/synthetic_fleet.csv \
  --report artifacts/model_comparison_v0.2.json \
  --artifact-dir artifacts/models
```

Quality gates:

```bash
ruff check .
pytest -q
```

Current CI suite: **14 tests** plus end-to-end v0.1, v0.2, and v0.3 command validation.

## Generated model artifacts

v0.2:

```text
artifacts/models/
├── baseline_logistic.joblib
├── candidate_calibrated.joblib
└── metadata.json
```

v0.3:

```text
artifacts/anomaly-model/
├── anomaly_isolation_forest.joblib
└── anomaly_metadata.json
```

Generated model binaries are ignored by Git; benchmark evidence and protocol files remain in source control.

## Progression

### v0.1 — failure-risk baseline ✅

- dataset/schema contract
- chronological holdout
- balanced Logistic Regression
- business-cost thresholding
- JSON evaluation report
- tests + CI

### v0.2 — calibrated model promotion ✅

- train/dev/test discipline
- HistGradientBoosting candidate
- probability calibration
- development-only thresholds
- untouched test promotion gate
- model artifact metadata
- post-test interpretation

### v0.3 — telemetry anomaly detection ✅

- causal rolling telemetry features
- robust statistical baseline
- Isolation Forest candidate
- score calibration + operational thresholds
- missing-sensor robustness
- unseen-vehicle stress test
- PSI drift simulation
- anomaly model artifacts
- 14-test suite + end-to-end CI

### v0.4 — ETA + geospatial optimization

- ETA/delay baseline and candidate
- geospatial feature pipeline
- route graph representation
- OR-Tools constrained-routing benchmark
- business objective: lateness + distance + capacity constraints

### v0.5 — production MLOps integration

- MLflow tracking/registry
- feature-store integration
- FastAPI online inference
- Kafka telemetry path
- Prometheus metrics
- container/Kubernetes deployment
- Terraform cloud stack

## Portfolio target

The final platform is intended to demonstrate classical ML, sequential/time-series feature engineering, anomaly detection, geospatial ML, optimization, streaming/data engineering, MLOps, cloud deployment, and measurable operational trade-offs in one coherent system.

## License

MIT. Third-party datasets, models, libraries, and services retain their own licenses.
