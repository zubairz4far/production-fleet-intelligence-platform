# Production Fleet Intelligence Platform

[![CI](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zubairz4far/production-fleet-intelligence-platform/actions/workflows/ci.yml)

Production-oriented fleet ML platform spanning failure-risk prediction, telemetry anomaly detection, ETA prediction, constrained routing, online serving, model registry lineage, streaming ingestion, observability, containers, Kubernetes, and Terraform.

## Status

**v0.5 — production MLOps integration around the promoted v0.4 ETA artifact.**

The platform now contains five release layers:

1. **v0.1** — leakage-aware vehicle failure baseline;
2. **v0.2** — calibrated model-promotion comparison;
3. **v0.3** — causal telemetry anomaly detection + robustness/drift checks;
4. **v0.4** — ETA prediction + capacitated OR-Tools routing;
5. **v0.5** — FastAPI serving, Prometheus metrics, MLflow registry lineage, Kafka ingestion, Docker, Kubernetes, Terraform, and CI release validation.

> v0.4 headline model/routing numbers come from deterministic synthetic CI fixtures. They validate software and evaluation mechanics only. v0.5 adds deployable integration code but does **not** claim a live production deployment, real fleet accuracy, real routing savings, or production SLOs.

## v0.5 architecture

```text
                       evaluated fleet platform
                               |
        +----------------------+-----------------------+
        |                      |                       |
        v                      v                       v
 failure-risk ML        anomaly / drift ML       v0.4 mobility
                                                       |
                                             promoted ETA artifact
                                                       |
                              +------------------------+------------------+
                              |                        |                  |
                              v                        v                  v
                        MLflow lineage           FastAPI API        Kafka worker
                     tracking + registry      /predict /metrics     manual commit
                                                       |
                                               Prometheus metrics
                                                       |
                              +------------------------+------------------+
                              |                                           |
                              v                                           v
                         Docker image                               model volume
                              |                                           |
                              +-------------------+-----------------------+
                                                  |
                                                  v
                                             Kubernetes
                                                  |
                                                  v
                                              Terraform
```

## Online ETA API

The API serves the existing promoted ETA joblib artifact. It verifies that the persisted feature list exactly matches the 13-feature serving contract before readiness can pass.

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | process liveness |
| `GET /ready` | model/artifact readiness |
| `POST /predict` | ETA inference |
| `GET /metrics` | Prometheus exposition |

Example:

```bash
fleet-intelligence serve \
  --model-path artifacts/mobility/eta/eta_hist_gradient_boosting.joblib \
  --port 8000
```

Prediction requests contain only dispatch-time inputs; `actual_travel_minutes` is not accepted or required for inference.

## MLflow release lineage

v0.5 can log the promoted ETA artifact and its frozen evaluation evidence to MLflow, with optional registration as a model version:

```bash
pip install -e ".[mlops]"

fleet-intelligence register \
  --model-path artifacts/mobility/eta/eta_hist_gradient_boosting.joblib \
  --evaluation-report artifacts/mobility_report_v0.4.json \
  --tracking-uri sqlite:///mlflow.db \
  --experiment fleet-intelligence-eta \
  --registered-model-name fleet-intelligence-eta
```

The command rejects non-promoted evaluation evidence before registry logging.

## Kafka-compatible ingestion

The streaming worker uses the same JSON schema as the HTTP API and disables auto-commit. A message offset is committed only after validation and successful ETA scoring.

```bash
fleet-intelligence consume \
  --model-path artifacts/mobility/eta/eta_hist_gradient_boosting.joblib \
  --bootstrap-servers localhost:9092 \
  --topic fleet.eta.requests \
  --group-id fleet-intelligence-eta-v0.5
```

Malformed messages or failed scoring are not acknowledged by this worker. Dead-letter/retry policy is intentionally left deployment-specific.

## Docker

The API image runs as a non-root user. The model artifact is mounted at runtime instead of being baked into the image.

```bash
docker compose up --build
```

Local Compose expects:

```text
artifacts/mobility/eta/eta_hist_gradient_boosting.joblib
```

and mounts it read-only at `/models/eta_hist_gradient_boosting.joblib`.

## Kubernetes

Manifests live under [`infra/k8s`](infra/k8s):

- `deployment.yaml`
- `service.yaml`
- `model-pvc.yaml`

The Deployment includes:

- two replicas;
- liveness and readiness probes;
- CPU/memory requests and limits;
- non-root execution;
- dropped Linux capabilities;
- read-only root filesystem;
- read-only model PVC mount.

The PVC is intentionally empty in source control. The promoted model artifact must be delivered to it before `/ready` succeeds.

## Terraform

[`infra/terraform`](infra/terraform) manages the Kubernetes namespace, model PVC, Deployment, and ClusterIP Service through the Kubernetes provider.

```bash
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan
```

This assumes an existing Kubernetes cluster and kubeconfig. v0.5 does not claim that a cloud cluster, DNS/TLS, Kafka cluster, or MLflow server has been provisioned.

## Measured v0.4 synthetic ETA result

Synthetic mobility fixture: **1,600 ETA trips**, **18 delivery stops + one depot**, **3 vehicles**, capacity **6 units per vehicle**, seed **84**.

| Model | MAE ↓ | RMSE ↓ | p90 absolute error ↓ | Mean bias |
|---|---:|---:|---:|---:|
| Train-fitted median-speed baseline | 7.787 min | 10.201 min | 16.322 min | -0.685 min |
| HistGradientBoosting candidate | **2.888 min** | **3.892 min** | **5.804 min** | **-0.106 min** |

**ETA promotion decision: PROMOTE.**

## Measured v0.4 synthetic routing result

| Solver | Total geometric distance ↓ | Max route distance ↓ | Stops served | Capacity violations |
|---|---:|---:|---:|---:|
| Nearest-feasible greedy baseline | 68.366 km | 25.206 km | 18/18 | 0 |
| OR-Tools guided local search | **63.666 km** | **22.582 km** | **18/18** | **0** |

**Routing promotion decision: PROMOTE.** OR-Tools reduced Haversine route distance by **6.87%** on this deterministic fixture.

This is not a production savings claim and does not use a live road graph or delivery time windows.

Machine-readable v0.4 evidence: [`evals/results/mobility_v0.4_synthetic.json`](evals/results/mobility_v0.4_synthetic.json).

## Previous evaluated milestones

### v0.3 anomaly detection

| Detector | PR-AUC | ROC-AUC | Recall | False negatives | Business cost |
|---|---:|---:|---:|---:|---:|
| Robust z-score baseline | 0.197 | **0.688** | 0.931 | 2 | 155 |
| Isolation Forest candidate | **0.210** | 0.641 | **0.966** | **1** | **153** |

**v0.3 decision: PROMOTE.** Missing-sensor robustness still carries the documented recall/cost warning.

### v0.2 failure-risk promotion

| Model | PR-AUC | ROC-AUC | Brier | Recall | Business cost |
|---|---:|---:|---:|---:|---:|
| Logistic Regression baseline | 0.373 | **0.884** | 0.140 | **0.696** | **63** |
| Calibrated HistGradientBoosting | **0.391** | 0.878 | **0.071** | 0.609 | 76 |

**v0.2 decision: REJECT.** The candidate improved ranking/calibration but worsened the configured business cost.

## Evaluation discipline

Model-selection experiments use chronological partitions when deployment is chronological:

```text
oldest                                               newest
|---------------- train ----------------|--- dev ---|--- test ---|
```

Release rules are frozen before test observation. A more complex model is not promoted automatically, and synthetic fixtures are never represented as real-world performance evidence.

See:

- [`docs/evaluation-policy.md`](docs/evaluation-policy.md)
- [`docs/telemetry-anomaly-v0.3.md`](docs/telemetry-anomaly-v0.3.md)
- [`docs/mobility-v0.4.md`](docs/mobility-v0.4.md)
- [`docs/production-mlops-v0.5.md`](docs/production-mlops-v0.5.md)

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,mlops]"

ruff check .
pytest -q
```

CI additionally validates:

- Docker Compose configuration;
- Docker image build;
- Terraform format/init/validate;
- v0.1–v0.4 evaluation compatibility;
- MLflow tracking + registered-model creation from the promoted ETA artifact;
- Confluent Kafka client import.

## Generated artifacts

```text
artifacts/
├── models/
├── anomaly-model/
└── mobility/
    └── eta/
        ├── eta_hist_gradient_boosting.joblib
        └── eta_metadata.json
```

Generated model binaries remain ignored by Git. Frozen protocols and benchmark evidence remain version-controlled.

## Next evidence gap

v0.5 is an integration milestone, not proof of a live production system. The next meaningful step is environment-backed validation: deploy to an actual cluster, attach a real artifact store/registry, exercise Kafka end-to-end, measure API latency/throughput, define an SLO, and preserve those results as deployment evidence.

## License

MIT. Third-party datasets, models, libraries, container images, and services retain their own licenses.
