from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .data import temporal_train_dev_test_split, temporal_train_test_split
from .evaluation import (
    evaluate_at_threshold,
    evaluate_probabilities,
    select_threshold_by_cost,
)
from .schema import FEATURE_COLUMNS, summarize_dataset, validate_dataset


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    @staticmethod
    def _log_odds(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        self.model.fit(self._log_odds(probabilities), np.asarray(y_true, dtype=int))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._log_odds(probabilities))[:, 1]


def build_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_candidate_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def train_baseline(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> tuple[Pipeline, dict[str, object]]:
    data = validate_dataset(frame)
    split = temporal_train_test_split(data, test_fraction=test_fraction)

    model = build_baseline_model()
    model.fit(split.x_train, split.y_train)
    probabilities = model.predict_proba(split.x_test)[:, 1]
    metrics = evaluate_probabilities(
        split.y_test.to_numpy(),
        probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    summary = summarize_dataset(data)

    report: dict[str, object] = {
        "project": "production-fleet-intelligence-platform",
        "version": "0.1.0",
        "task": "failure_within_horizon",
        "model": "logistic_regression_balanced",
        "features": list(FEATURE_COLUMNS),
        "dataset": asdict(summary),
        "split": {
            "strategy": "chronological_holdout",
            "test_fraction": test_fraction,
            "train_rows": len(split.x_train),
            "test_rows": len(split.x_test),
            "train_end": split.train_end,
            "test_start": split.test_start,
        },
        "threshold_policy": {
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
            "search_range": [0.05, 0.95],
            "search_step": 0.01,
        },
        "metrics": metrics.to_dict(),
        "limitations": [
            "v0.1 selects its threshold on the same holdout used for its reported metrics.",
            "Synthetic benchmark data is for CI only and is not production evidence.",
            "Business-cost weights are configurable assumptions, not universal fleet economics.",
        ],
    }
    return model, report


def compare_models(
    frame: pd.DataFrame,
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    data = validate_dataset(frame)
    split = temporal_train_dev_test_split(
        data,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
    )

    baseline = build_baseline_model()
    baseline.fit(split.x_train, split.y_train)
    baseline_dev_probabilities = baseline.predict_proba(split.x_dev)[:, 1]
    baseline_threshold = select_threshold_by_cost(
        split.y_dev.to_numpy(),
        baseline_dev_probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    ).threshold
    baseline_test_probabilities = baseline.predict_proba(split.x_test)[:, 1]
    baseline_metrics = evaluate_at_threshold(
        split.y_test.to_numpy(),
        baseline_test_probabilities,
        threshold=baseline_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    candidate = build_candidate_model()
    sample_weights = compute_sample_weight(class_weight="balanced", y=split.y_train)
    candidate.fit(split.x_train, split.y_train, sample_weight=sample_weights)

    candidate_dev_raw = candidate.predict_proba(split.x_dev)[:, 1]
    raw_threshold = select_threshold_by_cost(
        split.y_dev.to_numpy(),
        candidate_dev_raw,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    ).threshold
    candidate_test_raw = candidate.predict_proba(split.x_test)[:, 1]
    raw_metrics = evaluate_at_threshold(
        split.y_test.to_numpy(),
        candidate_test_raw,
        threshold=raw_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    calibrator = PlattCalibrator().fit(candidate_dev_raw, split.y_dev.to_numpy())
    candidate_dev_calibrated = calibrator.predict(candidate_dev_raw)
    candidate_threshold = select_threshold_by_cost(
        split.y_dev.to_numpy(),
        candidate_dev_calibrated,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    ).threshold
    candidate_test_calibrated = calibrator.predict(candidate_test_raw)
    candidate_metrics = evaluate_at_threshold(
        split.y_test.to_numpy(),
        candidate_test_calibrated,
        threshold=candidate_threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    criteria = {
        "pr_auc_non_regression": candidate_metrics.pr_auc >= baseline_metrics.pr_auc,
        "brier_non_regression": candidate_metrics.brier_score <= baseline_metrics.brier_score,
        "business_cost_non_regression": (
            candidate_metrics.business_cost <= baseline_metrics.business_cost
        ),
    }
    promoted = all(criteria.values())

    importance = permutation_importance(
        candidate,
        split.x_test,
        split.y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
    )
    ranked_importance = sorted(
        (
            {
                "feature": feature,
                "mean_pr_auc_drop": float(mean),
                "std": float(std),
            }
            for feature, mean, std in zip(
                FEATURE_COLUMNS,
                importance.importances_mean,
                importance.importances_std,
                strict=True,
            )
        ),
        key=lambda row: row["mean_pr_auc_drop"],
        reverse=True,
    )

    summary = summarize_dataset(data)
    report: dict[str, object] = {
        "project": "production-fleet-intelligence-platform",
        "version": "0.2.0",
        "task": "failure_within_horizon",
        "features": list(FEATURE_COLUMNS),
        "dataset": asdict(summary),
        "split": {
            "strategy": "chronological_train_dev_test",
            "dev_fraction": dev_fraction,
            "test_fraction": test_fraction,
            "train_rows": len(split.x_train),
            "dev_rows": len(split.x_dev),
            "test_rows": len(split.x_test),
            "train_end": split.train_end,
            "dev_start": split.dev_start,
            "dev_end": split.dev_end,
            "test_start": split.test_start,
        },
        "threshold_policy": {
            "selection_partition": "development",
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
            "search_range": [0.05, 0.95],
            "search_step": 0.01,
        },
        "baseline": {
            "model": "logistic_regression_balanced",
            "threshold_selected_on_dev": baseline_threshold,
            "test_metrics": baseline_metrics.to_dict(),
        },
        "candidate": {
            "model": "hist_gradient_boosting_balanced",
            "hyperparameters": {
                "learning_rate": 0.06,
                "max_iter": 180,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
            },
            "calibration": "platt_scaling_on_development_predictions",
            "raw_test_metrics": raw_metrics.to_dict(),
            "threshold_selected_on_dev": candidate_threshold,
            "calibrated_test_metrics": candidate_metrics.to_dict(),
        },
        "promotion": {
            "decision": "promote" if promoted else "reject",
            "criteria": criteria,
            "test_observed_once": True,
            "rule": (
                "Candidate must not regress PR-AUC, Brier score, or configured business cost "
                "versus the baseline on the untouched chronological test window."
            ),
        },
        "post_test_analysis": {
            "method": "permutation_importance_average_precision",
            "used_for_model_selection": False,
            "feature_importance": ranked_importance,
        },
        "artifact_metadata": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "features": list(FEATURE_COLUMNS),
            "candidate_threshold": candidate_threshold,
            "calibration": "platt",
        },
        "limitations": [
            "Synthetic benchmark data is for CI/software regression only and is not production evidence.",
            "Calibration and threshold selection share the development window in v0.2.",
            "Permutation importance is post-test analysis and must not be used to retune this observed test set.",
            "Business-cost weights are configurable assumptions, not universal fleet economics.",
        ],
    }

    if artifact_dir is not None:
        destination = Path(artifact_dir)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, destination / "baseline_logistic.joblib")
        joblib.dump(
            {
                "model": candidate,
                "calibrator": calibrator,
                "threshold": candidate_threshold,
                "features": list(FEATURE_COLUMNS),
            },
            destination / "candidate_calibrated.joblib",
        )
        (destination / "metadata.json").write_text(
            json.dumps(report["artifact_metadata"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return report


def write_report(report: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
