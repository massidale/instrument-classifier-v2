"""Multi-label metrics for the IRMAS benchmark.

IRMAS is reported with micro- and macro-averaged precision/recall/F1. Micro
pools every (sample, class) decision; macro averages the per-class F1 and so
weights rare instruments equally with common ones.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .data.labels import IRMAS_CLASSES


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute micro/macro precision, recall and F1 for binary multi-label arrays.

    ``y_true`` and ``y_pred`` are ``(n_samples, n_classes)`` with values in {0, 1}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {}
    for avg in ("micro", "macro"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"{avg}_precision"] = float(p)
        metrics[f"{avg}_recall"] = float(r)
        metrics[f"{avg}_f1"] = float(f1)
    return metrics


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """F1 per IRMAS class — the 'which instruments are hard' table.

    ``y_true`` and ``y_pred`` are ``(n_samples, 11)`` with values in {0, 1}.
    Returns a dict keyed by IRMAS class code, in ``IRMAS_CLASSES`` order.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {code: float(s) for code, s in zip(IRMAS_CLASSES, scores)}


def tune_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    candidates: np.ndarray,
    optimize: str = "micro_f1",
) -> tuple[float, float]:
    """Pick the single global threshold (from ``candidates``) that maximizes a metric.

    Returns ``(best_threshold, best_metric_value)``. Ties keep the first (lowest)
    threshold, which favours recall.
    """
    best_t = float(candidates[0])
    best_val = -1.0
    for t in candidates:
        y_pred = (y_scores >= t).astype(np.float32)
        val = multilabel_metrics(y_true, y_pred)[optimize]
        if val > best_val:
            best_val = val
            best_t = float(t)
    return best_t, best_val
