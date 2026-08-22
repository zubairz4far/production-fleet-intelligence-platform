# Evaluation and Promotion Policy

This project treats model evaluation as a release-control problem, not a notebook score.

## v0.1 benchmark contract

The initial vehicle-failure task uses a chronological holdout. Random row splitting is prohibited for reported benchmark results because future observations must not leak into training.

Identifiers and timestamps are metadata only. They are excluded from model features.

Primary probability metrics:

- PR-AUC
- ROC-AUC
- Brier score

Operating-point metrics:

- precision
- recall
- F1
- false positives
- false negatives
- configurable business cost

Accuracy is not a promotion metric.

## Threshold selection

A classification threshold is selected independently from the default 0.5 operating point by minimizing a configurable cost:

```text
cost = false_positives * FP_COST + false_negatives * FN_COST
```

The v0.1 example uses `FN_COST=5` and `FP_COST=1` to demonstrate asymmetric maintenance risk. These numbers are assumptions for evaluation mechanics, not a claim about universal fleet economics.

## Synthetic data

The deterministic synthetic generator exists for CI and software regression testing only. Scores on synthetic data must never be presented as evidence of real-world model quality.

## Real-data promotion sequence

When a suitable real fleet dataset is introduced:

1. define the prediction timestamp and target horizon before feature engineering;
2. freeze a development split and an untouched chronological test split;
3. establish the Logistic Regression result first;
4. develop stronger models only on training/development data;
5. freeze preprocessing, model hyperparameters, calibration, and threshold policy;
6. evaluate the candidate once on the untouched test set;
7. promote only if the candidate improves the declared release criteria without unacceptable calibration or operational regressions.

## Future stress tests

Later releases should add:

- vehicle-group temporal holdout
- unseen-vehicle generalization
- seasonal/time-window slices
- high-mileage / high-service-age slices
- calibration curves and expected calibration error
- drift simulation
- feature-missingness robustness
- cost sensitivity analysis

This policy is intentionally stricter than what the v0.1 synthetic fixture can prove.