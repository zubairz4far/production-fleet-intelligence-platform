from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import evaluate_at_threshold, select_threshold_by_cost
from .telemetry import (
    ANOMALY_TARGET,
    TELEMETRY_SIGNALS,
    anomaly_feature_columns,
    build_past_only_telemetry_features,
    population_stability_index,
    simulate_sensor_drift,
    summarize_anomaly_dataset,
    validate_anomaly_dataset,
)


@dataclass(frozen=True)
class AnomalyTemporalSplit:
    x_train: pd.DataFrame
    y_train: pd.Series
    meta_train: pd.DataFrame
    x_dev: pd.DataFrame
    y_dev: pd.Series
    meta_dev: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.Series
    meta_test: pd.DataFrame
    train_end: str
    dev_start: str
    dev_end: str
    test_start: str


class RobustZScoreDetector:
    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame) -> RobustZScoreDetector:
        values = self.imputer.fit_transform(x)
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        robust_scale = 1.4826 * mad
        standard_scale = np.std(values, axis=0)
        scale = np.where(robust_scale > 1e-8, robust_scale, standard_scale)
        self.center_ = center
        self.scale_ = np.where(scale > 1e-8, scale, 1.0)
        return self

    def score_samples(self, x: pd.DataFrame) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Detector must be fitted before scoring")
        values = self.imputer.transform(x)
        absolute_z = np.abs((values - self.center_) / self.scale_)
        return np.quantile(absolute_z, 0.95, axis=1)


class ScoreCalibrator:
    def __init__(self) -> None:
        self.model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    @staticmethod
    def _matrix(scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float).reshape(-1, 1)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> ScoreCalibrator:
        labels = np.asarray(y_true, dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError("Calibration requires both normal and anomalous development labels")
        self.model.fit(self._matrix(scores), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._matrix(scores))[:, 1]


def temporal_anomaly_split(
    feature_table: pd.DataFrame,
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> AnomalyTemporalSplit:
    if not 0.1 <= dev_fraction <= 0.3:
        raise ValueError("dev_fraction must be between 0.1 and 0.3")
    if not 0.1 <= test_fraction <= 0.3:
        raise ValueError("test_fraction must be between 0.1 and 0.3")
    if dev_fraction + test_fraction > 0.5:
        raise ValueError("dev_fraction + test_fraction must leave at least 50% for training")

    data = feature_table.sort_values("timestamp", kind="stable").reset_index(drop=True)
    train_end_index = int(len(data) * (1 - dev_fraction - test_fraction))
    dev_end_index = int(len(data) * (1 - test_fraction))
    train = data.iloc[:train_end_index].copy()
    dev = data.iloc[train_end_index:dev_end_index].copy()
    test = data.iloc[dev_end_index:].copy()

    if min(len(train), len(dev), len(test)) < 20:
        raise ValueError("Dataset is too small for a stable anomaly train/dev/test split")
    if train["timestamp"].max() > dev["timestamp"].min():
        raise AssertionError("Training observations overlap the anomaly development window")
    if dev["timestamp"].max() > test["timestamp"].min():
        raise AssertionError("Development observations overlap the anomaly test window")

    columns = list(anomaly_feature_columns())
    metadata = ["vehicle_id", "timestamp"]
    return AnomalyTemporalSplit(
        x_train=train.loc[:, columns],
        y_train=train[ANOMALY_TARGET],
        meta_train=train.loc[:, metadata],
        x_dev=dev.loc[:, columns],
        y_dev=dev[ANOMALY_TARGET],
        meta_dev=dev.loc[:, metadata],
        x_test=test.loc[:, columns],
        y_test=test[ANOMALY_TARGET],
        meta_test=test.loc[:, metadata],
        train_end=train["timestamp"].max().isoformat(),
        dev_start=dev["timestamp"].min().isoformat(),
        dev_end=dev["timestamp"].max().isoformat(),
        test_start=test["timestamp"].min().isoformat(),
    )


def build_isolation_forest() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    max_samples="auto",
                    contamination="auto",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def isolation_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    transformed = model.named_steps["impute"].transform(x)
    detector = model.named_steps["model"]
    return -detector.decision_function(transformed)


def _fit_and_evaluate_detector(
    detector_name: str,
    detector: RobustZScoreDetector | Pipeline,
    split: AnomalyTemporalSplit,
    *,
    false_negative_cost: float,
    false_positive_cost: float,
) -> tuple[ScoreCalibrator, float, dict[str, object]]:
    normal_mask = split.y_train.to_numpy() == 0
    if int(normal_mask.sum()) < 50:
        raise ValueError("Anomaly training requires at least 50 normal observations")

    detector.fit(split.x_train.loc[normal_mask])
    if isinstance(detector, RobustZScoreDetector):
        dev_scores = detector.score_samples(split.x_dev)
        test_scores = detector.score_samples(split.x_test)
    else:
        dev_scores = isolation_scores(detector, split.x_dev)
        test_scores = isolation_scores(detector, split.x_test)

    calibrator = ScoreCalibrator().fit(dev_scores, split.y_dev.to_numpy())
    dev_probabilities = calibrator.predict(dev_scores)
    threshold = select_threshold_by_cost(
        split.y_dev.to_numpy(),
        dev_probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    ).threshold
    test_probabilities = calibrator.predict(test_scores)
    metrics = evaluate_at_threshold(
        split.y_test.to_numpy(),
        test_probabilities,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    report: dict[str, object] = {
        "model": detector_name,
        "calibration": "logistic_score_calibration_on_development",
        "threshold_selected_on_dev": threshold,
        "test_metrics": metrics.to_dict(),
    }
    return calibrator, threshold, report


def _mask_sensor_families(
    x: pd.DataFrame,
    *,
    missing_rate: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    if not 0.0 < missing_rate < 0.5:
        raise ValueError("missing_rate must be between 0 and 0.5")
    masked = x.copy()
    rng = np.random.default_rng(seed)
    rows_per_signal = max(1, int(round(len(masked) * missing_rate)))

    for signal in TELEMETRY_SIGNALS:
        row_positions = rng.choice(len(masked), size=rows_per_signal, replace=False)
        family = [column for column in masked.columns if column == signal or column.startswith(f"{signal}_")]
        masked.iloc[row_positions, masked.columns.get_indexer(family)] = np.nan
    return masked


def _missingness_robustness(
    model: Pipeline,
    calibrator: ScoreCalibrator,
    threshold: float,
    split: AnomalyTemporalSplit,
    *,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, object]:
    clean_scores = isolation_scores(model, split.x_test)
    clean_probabilities = calibrator.predict(clean_scores)
    clean_metrics = evaluate_at_threshold(
        split.y_test.to_numpy(),
        clean_probabilities,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    masked = _mask_sensor_families(split.x_test, missing_rate=0.10, seed=73)
    masked_scores = isolation_scores(model, masked)
    masked_probabilities = calibrator.predict(masked_scores)
    masked_metrics = evaluate_at_threshold(
        split.y_test.to_numpy(),
        masked_probabilities,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return {
        "missing_rate_per_sensor_family": 0.10,
        "clean_pr_auc": clean_metrics.pr_auc,
        "masked_pr_auc": masked_metrics.pr_auc,
        "pr_auc_drop": max(0.0, clean_metrics.pr_auc - masked_metrics.pr_auc),
        "clean_recall": clean_metrics.recall,
        "masked_recall": masked_metrics.recall,
        "recall_drop": max(0.0, clean_metrics.recall - masked_metrics.recall),
        "masked_business_cost": masked_metrics.business_cost,
    }


def _drift_report(raw_data: pd.DataFrame, split: AnomalyTemporalSplit) -> dict[str, object]:
    train_end = pd.Timestamp(split.train_end)
    test_start = pd.Timestamp(split.test_start)
    train = raw_data.loc[raw_data["timestamp"] <= train_end]
    test = raw_data.loc[raw_data["timestamp"] >= test_start]
    drifted = simulate_sensor_drift(test)

    clean_psi = {
        signal: population_stability_index(train[signal], test[signal])
        for signal in TELEMETRY_SIGNALS
    }
    simulated_psi = {
        signal: population_stability_index(train[signal], drifted[signal])
        for signal in TELEMETRY_SIGNALS
    }
    return {
        "method": "population_stability_index_train_reference",
        "high_drift_threshold": 0.25,
        "clean_test_psi": clean_psi,
        "simulated_drift_psi": simulated_psi,
        "clean_max_psi": max(clean_psi.values()),
        "simulated_max_psi": max(simulated_psi.values()),
        "simulated_high_drift_features": [
            signal for signal, value in simulated_psi.items() if value >= 0.25
        ],
    }


def _unseen_vehicle_stress_test(
    split: AnomalyTemporalSplit,
    *,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, object]:
    vehicle_ids = sorted(
        set(split.meta_train["vehicle_id"])
        | set(split.meta_dev["vehicle_id"])
        | set(split.meta_test["vehicle_id"])
    )
    holdout_ids = set(vehicle_ids[::5])
    train_mask = ~split.meta_train["vehicle_id"].isin(holdout_ids)
    dev_mask = ~split.meta_dev["vehicle_id"].isin(holdout_ids)
    test_mask = split.meta_test["vehicle_id"].isin(holdout_ids)

    train_normal = train_mask & (split.y_train == 0)
    dev_labels = split.y_dev.loc[dev_mask].to_numpy()
    test_labels = split.y_test.loc[test_mask].to_numpy()
    if (
        int(train_normal.sum()) < 50
        or len(np.unique(dev_labels)) < 2
        or len(np.unique(test_labels)) < 2
    ):
        return {
            "available": False,
            "reason": "Synthetic slice did not contain enough normal/anomaly examples.",
        }

    model = build_isolation_forest()
    model.fit(split.x_train.loc[train_normal])
    dev_scores = isolation_scores(model, split.x_dev.loc[dev_mask])
    calibrator = ScoreCalibrator().fit(dev_scores, dev_labels)
    dev_probabilities = calibrator.predict(dev_scores)
    threshold = select_threshold_by_cost(
        dev_labels,
        dev_probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    ).threshold
    test_scores = isolation_scores(model, split.x_test.loc[test_mask])
    test_probabilities = calibrator.predict(test_scores)
    metrics = evaluate_at_threshold(
        test_labels,
        test_probabilities,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return {
        "available": True,
        "held_out_vehicle_count": len(holdout_ids),
        "held_out_vehicle_fraction": len(holdout_ids) / len(vehicle_ids),
        "test_rows": int(test_mask.sum()),
        "metrics": metrics.to_dict(),
        "note": (
            "Held-out vehicle identities are excluded from model fitting; their own past sensor "
            "history is still available for causal rolling features."
        ),
    }


def evaluate_anomaly_detection(
    frame: pd.DataFrame,
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
    false_negative_cost: float = 8.0,
    false_positive_cost: float = 1.0,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    raw_data = validate_anomaly_dataset(frame)
    feature_table = build_past_only_telemetry_features(raw_data)
    split = temporal_anomaly_split(
        feature_table,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
    )

    baseline = RobustZScoreDetector()
    _, _, baseline_report = _fit_and_evaluate_detector(
        "robust_zscore_95th_percentile",
        baseline,
        split,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    candidate = build_isolation_forest()
    candidate_calibrator, candidate_threshold, candidate_report = _fit_and_evaluate_detector(
        "isolation_forest_300",
        candidate,
        split,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )

    baseline_metrics = baseline_report["test_metrics"]
    candidate_metrics = candidate_report["test_metrics"]
    missingness = _missingness_robustness(
        candidate,
        candidate_calibrator,
        candidate_threshold,
        split,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    criteria = {
        "pr_auc_non_regression": candidate_metrics["pr_auc"] >= baseline_metrics["pr_auc"],
        "business_cost_non_regression": (
            candidate_metrics["business_cost"] <= baseline_metrics["business_cost"]
        ),
        "missingness_pr_auc_drop_lte_0_10": missingness["pr_auc_drop"] <= 0.10,
    }
    promoted = all(criteria.values())

    summary = summarize_anomaly_dataset(raw_data)
    report: dict[str, object] = {
        "project": "production-fleet-intelligence-platform",
        "version": "0.3.0",
        "task": "telemetry_anomaly_detection",
        "dataset": asdict(summary),
        "feature_engineering": {
            "feature_count": len(anomaly_feature_columns()),
            "causal_rolling_policy": "shift(1) before rolling statistics per vehicle",
            "windows": [3, 6],
            "current_signals_available_at_score_time": list(TELEMETRY_SIGNALS),
        },
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
        },
        "baseline": baseline_report,
        "candidate": candidate_report,
        "promotion": {
            "decision": "promote" if promoted else "reject",
            "criteria": criteria,
            "test_observed_once": True,
        },
        "robustness": {
            "missing_sensor_families": missingness,
            "unseen_vehicle": _unseen_vehicle_stress_test(
                split,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            ),
        },
        "drift": _drift_report(raw_data, split),
        "artifact_metadata": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "feature_count": len(anomaly_feature_columns()),
            "candidate_threshold": candidate_threshold,
            "candidate": "isolation_forest_300",
        },
        "limitations": [
            "Synthetic anomaly labels are for CI/software regression only, not production evidence.",
            "Development labels calibrate anomaly scores and choose operating thresholds.",
            "PSI detects distribution shift but does not prove a causal performance regression.",
            "A real fleet deployment needs sensor-specific sampling and maintenance-event semantics.",
        ],
    }

    if artifact_dir is not None:
        destination = Path(artifact_dir)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": candidate,
                "calibrator": candidate_calibrator,
                "threshold": candidate_threshold,
                "features": list(anomaly_feature_columns()),
            },
            destination / "anomaly_isolation_forest.joblib",
        )
        (destination / "anomaly_metadata.json").write_text(
            json.dumps(report["artifact_metadata"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return report
