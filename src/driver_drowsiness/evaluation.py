"""Dependency-light evaluation helpers."""

from __future__ import annotations

from collections import Counter


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_classification_report(
    y_true: list[str],
    y_pred: list[str],
    *,
    positive_label: str = "drowsy",
) -> dict[str, object]:
    """Compute a compact binary classification summary."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("Cannot compute metrics on an empty set.")

    tp = tn = fp = fn = 0
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if truth == positive_label and prediction == positive_label:
            tp += 1
        elif truth == positive_label and prediction != positive_label:
            fn += 1
        elif truth != positive_label and prediction == positive_label:
            fp += 1
        else:
            tn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, len(y_true))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": Counter(y_true),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
