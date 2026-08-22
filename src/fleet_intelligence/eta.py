from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .geospatial import (
    ETA_FEATURE_COLUMNS,
    ETA_TARGET,
    build_eta_features,
    summarize_eta_dataset,
)


@dataclass(frozen=True)
class EtaTemporalSplit:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_dev: pd.DataFrame
    y_dev: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    train_end: str
    dev_start: str
    dev_end: str
    test_start: str


@dataclass(frozen=True)
class EtaMetrics:
    mae_minutes: float
    rmse_minutes: float
    p90_absolute_error_minutes: float
    mean_bias_minutes: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def temporal_eta_split(
    feature_table: pd.DataFrame,
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> EtaTemporalSplit:
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

    if min(len(train), len(dev), len(test)) < 30:
        raise ValueError("ETA dataset is too small for train/dev/test evaluation")
    if train["timestamp"].max() > dev["timestamp"].min():
        raise AssertionError("ETA training observations overlap development")
    if dev["timestamp"].max() > test["timestamp"].min():
        raise AssertionError("ETA development observations overlap test")

    columns = list(ETA_FEATURE_COLUMNS)
    return EtaTemporalSplit(
        x_train=train.loc[:, columns],
        y_train=train[ETA_TARGET],
        x_dev=dev.loc[:, columns],
        y_dev=dev[ETA_TARGET],
        x_test=test.loc[:, columns],
        y_test=test[ETA_TARGET],
        train_end=train["timestamp"].max().isoformat(),
        dev_start=dev["timestamp"].min().isoformat(),
        dev_end=dev["timestamp"].max().isoformat(),
        test_start=test["timestamp"].min().isoformat(),
    )


def evaluate_eta_predictions(y_true: np.ndarray, predictions: np.ndarray) -> EtaMetrics:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    absolute_error = np.abs(predicted - actual)
    return EtaMetrics(
        mae_minutes=float(mean_absolute_error(actual, predicted)),
        rmse_minutes=float(np.sqrt(mean_squared_error(actual, predicted))),
        p90_absolute_error_minutes=float(np.quantile(absolute_error, 0.90)),
        mean_bias_minutes=float(np.mean(predicted - actual)),
    )


class MedianSpeedEtaBaseline:
    def __init__(self) -> None:
        self.median_speed_kph_: float | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> MedianSpeedEtaBaseline:
        hours = np.maximum(np.asarray(y, dtype=float) / 60.0, 1e-6)
        speed = np.asarray(x["planned_distance_km"], dtype=float) / hours
        valid = speed[np.isfinite(speed) & (speed > 1.0)]
        if len(valid) < 20:
            raise ValueError("Not enough valid training rows to estimate baseline speed")
        self.median_speed_kph_ = float(np.median(valid))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.median_speed_kph_ is None:
            raise RuntimeError("Baseline must be fitted before prediction")
        distance = np.asarray(x["planned_distance_km"], dtype=float)
        return np.maximum(distance / self.median_speed_kph_ * 60.0, 1.0)


def build_eta_candidate() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=24,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def evaluate_eta_models(
    frame: pd.DataFrame,
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    feature_table = build_eta_features(frame)
    split = temporal_eta_split(
        feature_table,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
    )

    baseline = MedianSpeedEtaBaseline().fit(split.x_train, split.y_train)
    candidate = build_eta_candidate()
    candidate.fit(split.x_train, split.y_train)

    baseline_dev = evaluate_eta_predictions(split.y_dev.to_numpy(), baseline.predict(split.x_dev))
    candidate_dev = evaluate_eta_predictions(split.y_dev.to_numpy(), candidate.predict(split.x_dev))

    baseline_test = evaluate_eta_predictions(
        split.y_test.to_numpy(),
        baseline.predict(split.x_test),
    )
    candidate_test = evaluate_eta_predictions(
        split.y_test.to_numpy(),
        candidate.predict(split.x_test),
    )

    criteria = {
        "mae_improves": candidate_test.mae_minutes < baseline_test.mae_minutes,
        "rmse_improves": candidate_test.rmse_minutes < baseline_test.rmse_minutes,
        "p90_error_improves": (
            candidate_test.p90_absolute_error_minutes
            < baseline_test.p90_absolute_error_minutes
        ),
    }
    promoted = all(criteria.values())
    summary = summarize_eta_dataset(frame)

    report: dict[str, object] = {
        "task": "eta_prediction",
        "dataset": asdict(summary),
        "feature_engineering": {
            "feature_count": len(ETA_FEATURE_COLUMNS),
            "features": list(ETA_FEATURE_COLUMNS),
            "target_excluded_from_features": True,
            "geospatial_distance": "haversine_great_circle_km",
            "time_encoding": "cyclic_hour_and_day_of_week",
        },
        "split": {
            "strategy": "chronological_train_dev_test",
            "train_rows": len(split.x_train),
            "dev_rows": len(split.x_dev),
            "test_rows": len(split.x_test),
            "train_end": split.train_end,
            "dev_start": split.dev_start,
            "dev_end": split.dev_end,
            "test_start": split.test_start,
        },
        "baseline": {
            "model": "train_median_effective_speed",
            "median_speed_kph": baseline.median_speed_kph_,
            "dev_metrics": baseline_dev.to_dict(),
            "test_metrics": baseline_test.to_dict(),
        },
        "candidate": {
            "model": "hist_gradient_boosting_regressor",
            "hyperparameters": {
                "learning_rate": 0.06,
                "max_iter": 220,
                "max_leaf_nodes": 24,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
            },
            "dev_metrics": candidate_dev.to_dict(),
            "test_metrics": candidate_test.to_dict(),
        },
        "promotion": {
            "decision": "promote" if promoted else "reject",
            "criteria": criteria,
            "test_observed_once": True,
            "rule": (
                "Candidate must improve MAE, RMSE, and p90 absolute error versus the "
                "train-fitted median-speed baseline on the untouched chronological test window."
            ),
        },
        "artifact_metadata": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "feature_count": len(ETA_FEATURE_COLUMNS),
            "candidate": "hist_gradient_boosting_regressor",
        },
        "limitations": [
            "Synthetic trip outcomes are CI regression evidence only, not fleet performance.",
            "The road network is represented by planned distance rather than a live map graph.",
            "Traffic and weather are synthetic operational covariates in the CI fixture.",
        ],
    }

    if artifact_dir is not None:
        destination = Path(artifact_dir)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": candidate,
                "features": list(ETA_FEATURE_COLUMNS),
            },
            destination / "eta_hist_gradient_boosting.joblib",
        )
        (destination / "eta_metadata.json").write_text(
            json.dumps(report["artifact_metadata"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return report
