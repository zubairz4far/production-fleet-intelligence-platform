from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    cost: float
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class ClassificationMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    threshold: float
    false_positives: int
    false_negatives: int
    business_cost: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def select_threshold_by_cost(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> ThresholdResult:
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Business costs must be positive")

    best: ThresholdResult | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = probabilities >= threshold
        false_positives = int(((predictions == 1) & (y_true == 0)).sum())
        false_negatives = int(((predictions == 0) & (y_true == 1)).sum())
        cost = false_positives * false_positive_cost + false_negatives * false_negative_cost
        candidate = ThresholdResult(float(threshold), float(cost), false_positives, false_negatives)
        if best is None or (candidate.cost, candidate.threshold) < (best.cost, best.threshold):
            best = candidate

    if best is None:
        raise RuntimeError("Threshold search produced no candidate")
    return best


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation requires both positive and negative labels")

    selected = select_threshold_by_cost(
        y_true,
        probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    predictions = (probabilities >= selected.threshold).astype(int)

    return ClassificationMetrics(
        pr_auc=float(average_precision_score(y_true, probabilities)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        threshold=selected.threshold,
        false_positives=selected.false_positives,
        false_negatives=selected.false_negatives,
        business_cost=selected.cost,
    )
