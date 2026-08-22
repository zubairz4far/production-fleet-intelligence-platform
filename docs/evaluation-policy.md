# Evaluation and Promotion Policy

This project treats model evaluation as a release-control problem, not a notebook score.

## Core rules

- Random row splitting is prohibited for reported promotion results when deployment is chronological.
- Identifiers, timestamps, targets, and post-outcome fields are metadata rather than model features unless a task explicitly justifies otherwise.
- Training fits model parameters.
- Development data may fit calibration and select operating thresholds.
- The final chronological test window is evaluated only after model, preprocessing, calibration, threshold policy, and release criteria are fixed.
- A more complex model is not promoted merely because one metric improves.
- Synthetic fixtures validate software/evaluation behavior only; they are never real-world performance evidence.

## v0.1 failure-risk benchmark

The initial vehicle-failure task uses a chronological holdout. Its metrics include:

- PR-AUC
- ROC-AUC
- Brier score
- precision
- recall
- F1
- false positives
- false negatives
- configurable business cost

Accuracy is not a promotion metric.

The original v0.1 compatibility path selects its threshold on the same holdout used for reporting. It is retained for reproducibility, but later promotion experiments use the stricter train/development/test protocol.

## v0.2 model-promotion contract

The failure-risk comparison uses chronological train/development/test partitions.

Both the Logistic Regression baseline and gradient-boosting candidate select operating thresholds from development predictions. The candidate probability calibrator is also fitted on development predictions. Test labels are not used for either decision.

The v0.2 candidate is promoted only if all predeclared conditions hold on the untouched test window:

1. PR-AUC does not regress versus the baseline;
2. Brier score does not regress;
3. configured business cost does not regress.

Post-test permutation importance is interpretation only. It must not be used to retune against the already-observed test set.

## v0.3 telemetry-anomaly contract

Sequential telemetry features must preserve causality. Historical per-vehicle rolling features apply `shift(1)` before rolling aggregation, so an observation cannot enter its own historical context and future observations cannot affect earlier feature rows. The current sensor value is allowed because anomaly detection scores the observation that has just arrived.

Anomaly detectors fit on the early normal training window. Development labels may be used for anomaly-score calibration and operating-threshold selection. The chronological test labels remain untouched until those choices and the release criteria are frozen.

The v0.3 Isolation Forest candidate is promoted only if all criteria declared before test observation hold:

1. test PR-AUC does not regress versus the robust-z baseline;
2. configured test business cost does not regress;
3. the predeclared missing-sensor stress test causes no more than 0.10 absolute PR-AUC drop.

The benchmark also records missing-sensor recall/cost degradation, unseen-vehicle performance, and PSI drift signals. These diagnostics can expose weaknesses that were not part of the frozen promotion rule.

### No post-hoc gate changes

A diagnostic discovered after test observation may be documented as a warning, but it must not be retroactively inserted into the already-evaluated promotion rule to change the historical result.

For example, the v0.3 synthetic test revealed material recall degradation under missing sensors even though its predeclared PR-AUC robustness criterion passed. That warning is preserved. A future release may add recall or cost robustness to its promotion criteria, but it must evaluate that stricter rule on a new untouched holdout.

This keeps model governance auditable and prevents repeated optimization against the same test labels.

## v0.4 ETA and routing contract

The ETA task uses chronological train/development/test partitions. The target `actual_travel_minutes` is excluded from features. Dispatch-time features include planned and Haversine distance, detour ratio, traffic, weather, vehicle load, remaining stops, cyclic time features, and route bearing.

The ETA baseline estimates a median effective speed from training data only. The fixed histogram-gradient-boosting candidate is promoted only if all predeclared conditions hold on the untouched test window:

1. MAE improves versus the baseline;
2. RMSE improves versus the baseline;
3. p90 absolute error improves versus the baseline.

The routing task is evaluated separately from predictive ETA quality. Its deterministic baseline greedily selects the nearest feasible stop. The OR-Tools candidate must satisfy the declared capacitated vehicle-routing constraints and is promoted only if:

1. every stop is served;
2. no vehicle capacity is violated;
3. total geometric route distance does not exceed the greedy baseline.

The overall v0.4 release gate passes only if both the ETA and routing gates pass.

The v0.4 synthetic routing fixture uses Haversine geometry rather than a live road graph. It does not claim road-network optimality, live traffic savings, time-window feasibility, or real fleet cost reduction.

## Business-cost thresholds

Operating thresholds may be selected independently from 0.5 by minimizing a configurable cost:

```text
cost = false_positives * FP_COST + false_negatives * FN_COST
```

The synthetic failure-risk example uses `FN_COST=5` and `FP_COST=1`; the synthetic anomaly example uses `FN_COST=8` and `FP_COST=1`. These are evaluation assumptions, not claims about universal fleet economics.

Real deployments require cost weights to be agreed with an operational owner and tied to actual intervention consequences.

## Drift policy

Distribution drift is monitored separately from predictive performance. v0.3 uses Population Stability Index (PSI) as a simple deterministic signal.

A drift alert means the observed feature distribution changed relative to the reference distribution. It does not establish concept drift, causal model degradation, or a need for automatic retraining.

Production drift handling should distinguish:

- data-quality failures;
- expected seasonality or fleet-composition change;
- covariate drift;
- label/performance degradation once outcomes arrive;
- genuine concept drift.

## Synthetic data

Deterministic synthetic generators exist for CI and software regression testing. Scores on synthetic data must always be labeled accordingly and must never be presented as evidence of production model quality.

A synthetic fixture may test:

- chronology and leakage controls;
- model/artifact reproducibility;
- promotion-rule behavior;
- missingness robustness;
- drift detectors;
- optimization constraint handling;
- command-line and CI integration.

It cannot establish real-world generalization.

## Real-data promotion sequence

When a suitable real fleet dataset is introduced:

1. define the prediction/detection timestamp and outcome semantics before feature engineering;
2. define sensor cadence, missingness behavior, and intervention/business costs;
3. freeze development and untouched chronological test partitions;
4. establish simple baselines first;
5. develop stronger models only on training/development data;
6. freeze preprocessing, model hyperparameters, calibration, threshold policy, and release criteria;
7. evaluate the candidate once on the untouched test set;
8. publish both successes and regressions;
9. require a new untouched holdout before changing rules based on post-test findings.

## Continuing stress tests

Future releases should extend evaluation with:

- additional unseen-vehicle and fleet-composition slices;
- seasonal/time-window slices;
- high-mileage / high-service-age slices;
- calibration curves and expected calibration error;
- realistic sensor outage patterns;
- cost sensitivity analysis;
- delayed-label performance monitoring;
- road-network travel matrices and service-time constraints;
- new locked holdouts for stricter robustness gates.

The goal is an auditable promotion process in which evaluation constraints become stricter as the platform becomes more production-like.
