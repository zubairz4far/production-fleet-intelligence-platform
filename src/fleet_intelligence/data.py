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
