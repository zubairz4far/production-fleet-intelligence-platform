# v0.3 Telemetry Anomaly Protocol

v0.3 adds sequential telemetry anomaly detection without weakening the evaluation discipline established in the failure-risk milestones.

## Goal

Score newly arrived vehicle telemetry for abnormal behavior while avoiding future-data leakage and while measuring robustness to missing sensors, unseen vehicles, and distribution drift.

The CI dataset is synthetic. Its labels and scores exist only to validate software and evaluation behavior.

## Causal feature engineering

For each vehicle, current sensor observations are combined with historical context. Historical rolling statistics always apply `shift(1)` before rolling, so the current row and future rows cannot enter their own historical summaries.

Signals:

- engine temperature
- oil pressure
- battery voltage
- vibration RMS
- fuel rate

For each signal the feature table contains:

- current value
- previous value
- past 3-observation mean
- past 3-observation standard deviation
- current minus past-3 mean
- past 6-observation mean
- past 6-observation standard deviation

Mileage and hours since service are included as operating-context features.

The current sensor value is intentionally available at score time: anomaly detection evaluates the observation that just arrived. Only the historical summaries are constrained to prior observations.

## Temporal partitions

```text
oldest                                               newest
|---------------- train ----------------|--- dev ---|--- test ---|
```

Default split:

- 60% training
- 20% development
- 20% test

The synthetic anomaly fixture injects labeled anomalies only into development and test windows. The early training window remains normal for unsupervised detector fitting.

## Detectors

### Baseline

`RobustZScoreDetector`

- median center
- median absolute deviation scale
- standard-deviation fallback for near-constant features
- 95th percentile of absolute per-feature robust z-scores as the raw anomaly score

### Candidate

`IsolationForest`

- 300 trees
- median imputation
- normal training observations only
- fixed random seed

Both raw detector scores are converted into anomaly probabilities through a one-dimensional logistic score calibrator fitted on the development window.

## Threshold policy

Operating thresholds are selected only on development probabilities.

The default synthetic policy uses:

```text
false negative cost = 8
false positive cost = 1
```

The test labels do not participate in calibration or threshold selection.

## Candidate promotion

The Isolation Forest candidate is promoted only when all of the following hold:

1. test PR-AUC does not regress versus the robust-z baseline;
2. configured test business cost does not regress;
3. a 10% missing-sensor-family stress test causes no more than 0.10 absolute PR-AUC drop.

A more complex anomaly detector may therefore be rejected.

## Missing-sensor robustness

The test stress test removes 10% of observations independently for each sensor family. When a sensor is removed, its related current/lag/rolling features are masked together.

The already-fitted training imputer handles the missing values. No model or threshold is re-fit for this stress test.

Reported robustness includes:

- clean PR-AUC
- masked PR-AUC
- PR-AUC drop
- clean recall
- masked recall
- recall drop
- masked business cost

## Unseen-vehicle stress test

Every fifth sorted vehicle identifier is held out from model fitting. The candidate is then evaluated on those held-out vehicle identities in the chronological test window.

The model never sees those vehicle identities during fitting. Their own earlier sensor history is still permitted when constructing causal rolling features, matching a deployment where a newly onboarded vehicle accumulates its own telemetry history.

If a small synthetic slice does not contain both normal and anomalous labels, the stress test is reported as unavailable instead of manufacturing a score.

## Drift simulation

Population Stability Index (PSI) compares the training sensor distribution with:

1. the clean chronological test window;
2. a deterministic synthetic drift scenario.

The simulated drift shifts engine temperature, vibration, battery voltage, and oil pressure. PSI >= 0.25 is reported as high drift.

PSI is a data-distribution signal. It does not by itself prove model-performance degradation or identify a causal mechanism.

## Artifacts

The anomaly command can persist:

```text
artifacts/anomaly-model/
├── anomaly_isolation_forest.joblib
└── anomaly_metadata.json
```

The model artifact contains:

- fitted imputer + Isolation Forest
- fitted score calibrator
- development-selected threshold
- exact feature contract

## Real-data promotion requirements

Before making any fleet-performance claim, replace the synthetic fixture with a real telemetry dataset and define:

- sensor sampling cadence
- anomaly/maintenance-event semantics
- prediction or detection horizon
- per-vehicle availability and missingness rules
- immutable chronological test window
- business cost assumptions with an operational owner

The synthetic benchmark must remain labeled as CI/software-regression evidence only.
