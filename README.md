# Production Fleet Intelligence Platform

[![CI](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml)

Production-oriented ML platform for fleet operations: leakage-safe failure prediction, causal telemetry anomaly detection, ETA prediction, constrained route optimization, robustness evaluation, and a planned path to production MLOps.

## Status

**v0.4 — ETA prediction + capacitated route optimization.**

The repository now contains four evaluated capabilities:

1. **vehicle failure-risk prediction** — Logistic Regression baseline versus calibrated histogram gradient boosting;
2. **telemetry anomaly detection** — robust statistical baseline versus Isolation Forest with causal rolling features;
3. **ETA prediction** — train-fitted median-speed baseline versus histogram gradient boosting;
4. **route optimization** — deterministic greedy routing versus Google OR-Tools under vehicle-capacity constraints.

All reported release decisions use explicit evaluation contracts rather than assuming a more complex model or optimizer is automatically better.

> All headline numbers below come from deterministic synthetic CI fixtures. They validate software, evaluation, and optimization behavior only. They are **not evidence of real fleet performance, real ETA accuracy, road-network savings, or production cost reduction**.

## Measured v0.4 CI result

Synthetic mobility fixture: **1,600 ETA trips**, **18 delivery stops + one depot**, **3 vehicles**, capacity **6 units per vehicle**, seed **84**.

### ETA prediction

The ETA task uses a chronological **train → development → untouched test** split with 960 / 320 / 320 rows and 13 dispatch-time features.

| Model | MAE ↓ | RMSE ↓ | p90 absolute error ↓ | Mean bias |
|---|---:|---:|---:|---:|
| Train-fitted median-speed baseline | 7.787 min | 10.201 min | 16.322 min | -0.685 min |
| HistGradientBoosting candidate | **2.888 min** | **3.892 min** | **5.804 min** | **-0.106 min** |

**Predeclared ETA promotion decision: PROMOTE.**

The candidate improved every frozen test criterion: MAE, RMSE, and p90 absolute error.

ETA features include planned distance, Haversine distance, detour ratio, traffic index, weather severity, vehicle load, remaining stops, cyclic hour/day-of-week encodings, and route-bearing encodings. `actual_travel_minutes` is excluded from the feature matrix.

### Capacitated routing

The routing benchmark requires one depot, every stop to be served exactly once, and every vehicle route to stay within capacity.

| Solver | Total geometric distance ↓ | Max route distance ↓ | Stops served | Capacity violations |
|---|---:|---:|---:|---:|
| Nearest-feasible greedy baseline | 68.366 km | 25.206 km | 18/18 | 0 |
| OR-Tools guided local search | **63.666 km** | **22.582 km** | **18/18** | **0** |

**Predeclared routing promotion decision: PROMOTE.**

On this deterministic fixture, OR-Tools reduced Haversine route distance by **6.87%** while serving every stop and preserving capacity feasibility.

This is a geometric synthetic benchmark. v0.4 does **not** claim live road-network optimization, real traffic-aware routing, delivery time-window feasibility, or a 6.87% production saving.

### v0.4 release gate

```text
ETA gate       PASS
Routing gate   PASS
-------------------
v0.4 release   PROMOTE
```

Machine-readable evidence: [`evals/results/mobility_v0.4_synthetic.json`](evals/results/mobility_v0.4_synthetic.json).

Protocol: [`docs/mobility-v0.4.md`](docs/mobility-v0.4.md).

## Previous evaluated milestones

### v0.3 telemetry anomaly detection

| Detector | PR-AUC | ROC-AUC | Recall | False negatives | Business cost |
|---|---:|---:|---:|---:|---:|
| Robust z-score baseline | 0.197 | **0.688** | 0.931 | 2 | 155 |
| Isolation Forest candidate | **0.210** | 0.641 | **0.966** | **1** | **153** |

**v0.3 decision: PROMOTE** under its predeclared rule.

The missing-sensor stress test remains an explicit warning: masking 10% of each sensor family reduced recall from **0.966 to 0.793** and increased synthetic business cost from **153 to 177**, even though the predeclared PR-AUC robustness gate passed. The historical gate was not changed after test observation.

Evidence: [`evals/results/anomaly_detection_v0.3_synthetic.json`](evals/results/anomaly_detection_v0.3_synthetic.json).

### v0.2 failure-risk model promotion

| Model | PR-AUC | ROC-AUC | Brier | Recall | Business cost |
|---|---:|---:|---:|---:|---:|
| Logistic Regression baseline | 0.373 | **0.884** | 0.140 | **0.696** | **63** |
| Calibrated HistGradientBoosting | **0.391** | 0.878 | **0.071** | 0.609 | 76 |

**v0.2 decision: REJECT.** The stronger classifier improved ranking/calibration but worsened the configured business cost.

Evidence: [`evals/results/model_comparison_v0.2_synthetic.json`](evals/results/model_comparison_v0.2_synthetic.json).

## v0.4 architecture

```text
                         Fleet platform
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
failure-risk ML       telemetry anomaly ML       mobility data
       |                      |                      |
       |              causal rolling features       |
       |                      |              +-------+-------+
       |                      |              |               |
       v                      v              v               v
model promotion       anomaly + drift      ETA ML       routing stops
                                             |               |
                                   chronological split       |
                                             |               |
                                  median-speed baseline      |
                                             vs              |
                                   HistGradientBoosting      |
                                             |               |
                                             v               v
                                          ETA gate      greedy baseline
                                                          vs OR-Tools
                                                             |
                                                             v
                                                        routing gate
                                             \               /
                                              +------+------+ 
                                                     |
                                                     v
                                             v0.4 release gate
```

## ETA evaluation contract

Random row splitting is prohibited. Trip rows are sorted by timestamp:

```text
oldest                                               newest
|---------------- train ----------------|--- dev ---|--- test ---|
```

The baseline's effective speed is estimated from training outcomes only. The gradient-boosting candidate is fitted on training data with fixed hyperparameters. The final test window is evaluated after the model and release criteria are frozen.

ETA promotion requires all three:

1. lower MAE;
2. lower RMSE;
3. lower p90 absolute error.

## Routing evaluation contract

v0.4 solves a deterministic **capacitated vehicle-routing problem** over a Haversine distance matrix.

Constraints:

- exactly one depot;
- every delivery stop served exactly once;
- each route starts/ends at the depot;
- vehicle capacity cannot be exceeded.

The baseline greedily chooses the nearest currently feasible stop. The candidate uses OR-Tools `RoutingModel`, a capacity dimension, `PATH_CHEAPEST_ARC` initialization, and guided local search.

Routing promotion requires:

1. all stops served;
2. zero capacity violations;
3. total geometric distance no worse than the greedy baseline.

See [`docs/evaluation-policy.md`](docs/evaluation-policy.md) for the full release-governance rules.

## Causal telemetry features

The v0.3 anomaly track remains active. For each telemetry signal it builds current value, lag-1, past 3/6-observation mean/std, and current-minus-past mean. Historical windows apply `shift(1)` before rolling, preventing future observations from entering earlier feature rows.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run v0.4 mobility evaluation:

```bash
python scripts/generate_synthetic_mobility.py \
  --trips-output data/synthetic_eta_trips.csv \
  --stops-output data/synthetic_routing_stops.csv \
  --rows 1600 \
  --stops 18 \
  --seed 84

python -m fleet_intelligence.cli eta-routing \
  --trips data/synthetic_eta_trips.csv \
  --stops data/synthetic_routing_stops.csv \
  --report artifacts/mobility_report_v0.4.json \
  --artifact-dir artifacts/mobility \
  --vehicle-count 3 \
  --vehicle-capacity 6
```

Run v0.3 anomaly evaluation:

```bash
python scripts/generate_synthetic_fleet.py \
  --output data/synthetic_fleet_anomaly.csv \
  --with-anomalies

python -m fleet_intelligence.cli anomaly \
  --data data/synthetic_fleet_anomaly.csv \
  --report artifacts/anomaly_report_v0.3.json \
  --artifact-dir artifacts/anomaly-model
```

Quality gates:

```bash
ruff check .
pytest -q
```

Current v0.4 CI suite: **21 tests** plus end-to-end v0.1, v0.2, v0.3, and v0.4 evaluation/artifact validation. The suite includes a regression test that validates the committed v0.4 evidence contract.

## Generated model artifacts

```text
artifacts/
├── models/
│   ├── baseline_logistic.joblib
│   ├── candidate_calibrated.joblib
│   └── metadata.json
├── anomaly-model/
│   ├── anomaly_isolation_forest.joblib
│   └── anomaly_metadata.json
└── mobility/
    └── eta/
        ├── eta_hist_gradient_boosting.joblib
        └── eta_metadata.json
```

Generated model binaries are ignored by Git. Protocols and benchmark evidence remain version-controlled.

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
- HistGradientBoosting classifier candidate
- probability calibration
- development-only thresholds
- untouched test promotion gate
- model artifacts and evidence

### v0.3 — telemetry anomaly detection ✅

- causal rolling telemetry features
- robust statistical baseline
- Isolation Forest candidate
- missing-sensor robustness
- unseen-vehicle stress test
- PSI drift simulation
- anomaly artifacts and evidence

### v0.4 — ETA + constrained routing ✅

- Haversine/geospatial feature pipeline
- chronological ETA benchmark
- median-speed ETA baseline
- HistGradientBoosting ETA candidate
- p90-tail-error release criterion
- deterministic greedy route baseline
- OR-Tools capacitated VRP candidate
- combined ETA + routing release gate
- committed machine-readable evidence regression test
- 21-test suite + end-to-end CI

### v0.5 — production MLOps integration

- MLflow experiment tracking/model registry
- FastAPI online inference
- feature-store contract
- Kafka telemetry ingestion path
- Prometheus metrics
- Docker/Kubernetes deployment
- Terraform cloud infrastructure

## Portfolio target

The final platform is intended to demonstrate classical ML, sequential/time-series feature engineering, anomaly detection, geospatial ML, operations research/optimization, streaming/data engineering, MLOps, cloud deployment, and measurable operational trade-offs in one coherent system.

## License

MIT. Third-party datasets, models, libraries, and services retain their own licenses.
