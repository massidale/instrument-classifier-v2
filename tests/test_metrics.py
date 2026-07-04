import numpy as np
import pytest

from instrument_classifier.metrics import (
    multilabel_metrics,
    tune_threshold,
)


def test_perfect_prediction_scores_one():
    y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    m = multilabel_metrics(y_true, y_true)
    assert m["micro_f1"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_micro_and_macro_f1_hand_computed():
    # sample 0: true={0}, pred={0}  -> TP class0
    # sample 1: true={1}, pred={}   -> FN class1
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.array([[1, 0], [0, 0]], dtype=np.float32)
    m = multilabel_metrics(y_true, y_pred)
    # micro: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=0.6667
    assert m["micro_precision"] == pytest.approx(1.0)
    assert m["micro_recall"] == pytest.approx(0.5)
    assert m["micro_f1"] == pytest.approx(2 / 3, abs=1e-4)
    # macro: class0 F1=1.0, class1 F1=0.0 -> mean 0.5
    assert m["macro_f1"] == pytest.approx(0.5, abs=1e-4)


def test_tune_threshold_picks_value_maximizing_micro_f1():
    # Class present iff score high. A threshold near 0.5 separates perfectly.
    y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y_scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]], dtype=np.float32)
    best_t, best_f1 = tune_threshold(
        y_true, y_scores, candidates=np.linspace(0.1, 0.9, 9)
    )
    assert 0.3 <= best_t <= 0.65
    assert best_f1 == pytest.approx(1.0)


def test_tune_threshold_returns_scalar_in_candidate_set():
    y_true = np.array([[1, 0]], dtype=np.float32)
    y_scores = np.array([[0.8, 0.3]], dtype=np.float32)
    candidates = np.array([0.2, 0.5, 0.7])
    best_t, _ = tune_threshold(y_true, y_scores, candidates=candidates)
    assert best_t in candidates
