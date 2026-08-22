# v0.2 Model Promotion Protocol

v0.2 introduces a stronger tabular candidate without allowing the final test window to become a tuning set.

## Data partitions

All rows are ordered by observation timestamp and divided into:

```text
oldest                                               newest
|---------------- train ----------------|--- dev ---|--- test ---|
```

The default fractions are 60% training, 20% development, and 20% test.

- **train** fits model parameters;
- **development** fits the candidate probability calibrator and selects operating thresholds;
- **test** is evaluated only after the model, calibration method, hyperparameters, and threshold policy are fixed.

Identifiers, timestamps, and labels remain outside the feature matrix.

## Models

Baseline:

- balanced Logistic Regression;
- StandardScaler fitted on training data only.

Candidate:

- `HistGradientBoostingClassifier`;
- balanced training sample weights;
- fixed v0.2 hyperparameters;
- Platt scaling fitted to candidate development predictions.

There is no hyperparameter sweep against the test partition.

## Promotion rule

The calibrated candidate is promoted only when all three conditions hold on the untouched chronological test window:

1. PR-AUC does not regress versus the baseline;
2. Brier score does not regress versus the baseline;
3. configured business cost does not regress versus the baseline.

A candidate can therefore be rejected even when it is more complex.

## Threshold discipline

Both baseline and candidate operating thresholds are selected from development predictions using:

```text
cost = false_positives * FP_COST + false_negatives * FN_COST
```

The test labels never participate in threshold selection.

## Calibration

The gradient-boosted candidate is evaluated twice:

- raw probabilities;
- Platt-calibrated probabilities.

Calibration is intended to improve probability quality rather than ranking. The final promotion comparison uses the calibrated candidate.

## Post-test analysis

Permutation importance with average precision scoring is computed only after the final test evaluation. It is recorded for interpretation and is explicitly marked `used_for_model_selection=false`.

Once a real test set has been observed, those importance results must not be used to retune and repeatedly re-evaluate against that same test set. A new locked holdout is required for a later promotion decision.

## Artifacts

The comparison command can persist:

- `baseline_logistic.joblib`;
- `candidate_calibrated.joblib` containing the model, calibrator, feature contract, and selected threshold;
- `metadata.json` with runtime/model metadata.

These generated artifacts are ignored by Git and are intended for deployment/reproducibility workflows rather than source control.

## Synthetic CI warning

The deterministic synthetic telemetry fixture exists only to test code paths and regression behavior. Its model scores and promotion outcome are not evidence of real fleet performance.
