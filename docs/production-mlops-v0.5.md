# v0.5 Production MLOps Integration

v0.5 turns the evaluated v0.4 ETA artifact into a deployable service boundary. It does not retrain or change the v0.4 model-selection result.

## Release scope

v0.5 adds:

- FastAPI online ETA inference;
- strict artifact/feature-contract validation;
- liveness and readiness separation;
- Prometheus request and latency metrics;
- MLflow tracking plus model-registry integration;
- Kafka-compatible event ingestion with manual commit after successful scoring;
- Docker packaging;
- Kubernetes deployment/service/model-volume manifests;
- Terraform-managed Kubernetes deployment;
- CI validation of serving, container, Terraform, MLflow, Kafka imports, and all previous evaluation milestones.

## Architecture

```text
promoted v0.4 ETA artifact
          |
          +--------------------+
          |                    |
          v                    v
     MLflow lineage       model volume
  tracking + registry          |
                               v
                         FastAPI service
                         /health /ready
                         /predict /metrics
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
             HTTP client               Kafka worker
                                              |
                                              v
                                     manual offset commit
                                     after successful score

container image -> Kubernetes -> Terraform-managed Kubernetes resources
```

## Serving contract

The API loads the existing joblib bundle and verifies that its persisted feature list exactly matches the current 13-feature ETA serving contract. A mismatched artifact is rejected before readiness passes.

The API intentionally does not need `actual_travel_minutes`. Online feature construction uses only dispatch-time fields:

- trip ID and timestamp;
- origin/destination coordinates;
- planned distance;
- traffic index;
- weather severity;
- vehicle load;
- stops remaining.

The derived feature vector is identical in order to the training-time ETA feature contract.

### Endpoints

- `GET /health` — process liveness only;
- `GET /ready` — succeeds only when a compatible model artifact is loaded;
- `POST /predict` — returns ETA minutes and loaded model class;
- `GET /metrics` — Prometheus exposition endpoint.

Missing or incompatible model artifacts fail closed: the process may remain live, but readiness and prediction return 503.

## MLflow contract

The `register` command accepts the persisted ETA model artifact and the frozen evaluation JSON. It refuses to register a model unless the evaluation decision is `promote`.

A registration run logs:

- release/source-evaluation versions;
- model class and feature count;
- evaluation split strategy;
- the synthetic/production-evidence flag;
- candidate test metrics;
- the evaluation JSON as an artifact;
- the sklearn model itself;
- an optional registered-model version.

This preserves lineage between the release artifact and the evidence that allowed promotion.

## Kafka contract

Kafka events use the same JSON request schema as the HTTP API. The consumer disables auto-commit. An offset is committed synchronously only after:

1. the message is polled without a Kafka error;
2. JSON and schema validation succeed;
3. ETA inference succeeds.

Malformed events or scoring failures are not acknowledged by this worker. A production deployment should add a dead-letter/error-policy layer before choosing retry semantics.

## Container and cluster contract

The image runs as a non-root user and exposes port 8000. The model is not baked into the image. `ETA_MODEL_PATH` defaults to:

```text
/models/eta_hist_gradient_boosting.joblib
```

Docker Compose mounts `artifacts/mobility/eta` read-only for local serving.

The Kubernetes deployment uses:

- two replicas by default;
- HTTP liveness/readiness probes;
- CPU/memory requests and limits;
- non-root execution;
- dropped Linux capabilities;
- read-only root filesystem;
- read-only model PVC mount.

The supplied PVC is empty by design. The promoted artifact must be populated through the deployment environment's artifact-delivery process before `/ready` can pass.

## Terraform scope

Terraform manages the namespace, model PVC, Deployment, and ClusterIP Service through the Kubernetes provider. It assumes an existing Kubernetes cluster and working kubeconfig. It does not provision a cloud provider account, managed Kubernetes control plane, container registry, DNS, TLS, Kafka cluster, or MLflow server.

That boundary is deliberate: v0.5 validates deployable application/IaC mechanics without pretending a cloud environment was actually provisioned.

## CI release gate

The v0.5 PR is acceptable only if CI passes all of the following unchanged:

1. Ruff;
2. complete pytest suite;
3. Docker Compose configuration;
4. Docker image build;
5. Terraform format/init/validate;
6. v0.1 compatibility baseline;
7. v0.2 evaluation/artifacts;
8. v0.3 anomaly evaluation/artifacts;
9. v0.4 ETA/routing evaluation/artifacts;
10. MLflow tracking + registered-model creation from the promoted ETA artifact;
11. Confluent Kafka client import.

## Non-claims

v0.5 does not establish:

- real fleet ETA accuracy;
- production route savings;
- production API latency/SLOs;
- Kafka throughput or exactly-once processing;
- live MLflow server availability;
- live Kubernetes deployment health;
- cloud infrastructure provisioning;
- automated model retraining or rollback.

Those require environment-specific evidence rather than CI fixtures.
