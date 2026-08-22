from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import FEATURE_COLUMNS, TARGET, validate_dataset


@dataclass(frozen=True)
class TemporalSplit:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    train_end: str
    test_start: str


@dataclass(frozen=True)
class TemporalTrainDevTestSplit:
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


def temporal_train_test_split(frame: pd.DataFrame, test_fraction: float = 0.25) -> TemporalSplit:
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")

    data = validate_dataset(frame)
    split_index = int(len(data) * (1 - test_fraction))
    if split_index < 10 or len(data) - split_index < 5:
        raise ValueError("Dataset is too small for a stable temporal split")

    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    if train["timestamp"].max() > test["timestamp"].min():
        raise AssertionError("Temporal split leaked future observations into training")

    return TemporalSplit(
        x_train=train.loc[:, FEATURE_COLUMNS],
        y_train=train[TARGET],
        x_test=test.loc[:, FEATURE_COLUMNS],
        y_test=test[TARGET],
        train_end=train["timestamp"].max().isoformat(),
        test_start=test["timestamp"].min().isoformat(),
    )


def temporal_train_dev_test_split(
    frame: pd.DataFrame,
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> TemporalTrainDevTestSplit:
    if not 0.1 <= dev_fraction <= 0.3:
        raise ValueError("dev_fraction must be between 0.1 and 0.3")
    if not 0.1 <= test_fraction <= 0.3:
        raise ValueError("test_fraction must be between 0.1 and 0.3")
    if dev_fraction + test_fraction > 0.5:
        raise ValueError("dev_fraction + test_fraction must leave at least 50% for training")

    data = validate_dataset(frame)
    train_end_index = int(len(data) * (1 - dev_fraction - test_fraction))
    dev_end_index = int(len(data) * (1 - test_fraction))

    train = data.iloc[:train_end_index].copy()
    dev = data.iloc[train_end_index:dev_end_index].copy()
    test = data.iloc[dev_end_index:].copy()

    if min(len(train), len(dev), len(test)) < 10:
        raise ValueError("Dataset is too small for a stable train/dev/test temporal split")
    if train["timestamp"].max() > dev["timestamp"].min():
        raise AssertionError("Training observations overlap the development window")
    if dev["timestamp"].max() > test["timestamp"].min():
        raise AssertionError("Development observations overlap the test window")

    return TemporalTrainDevTestSplit(
        x_train=train.loc[:, FEATURE_COLUMNS],
        y_train=train[TARGET],
        x_dev=dev.loc[:, FEATURE_COLUMNS],
        y_dev=dev[TARGET],
        x_test=test.loc[:, FEATURE_COLUMNS],
        y_test=test[TARGET],
        train_end=train["timestamp"].max().isoformat(),
        dev_start=dev["timestamp"].min().isoformat(),
        dev_end=dev["timestamp"].max().isoformat(),
        test_start=test["timestamp"].min().isoformat(),
    )
