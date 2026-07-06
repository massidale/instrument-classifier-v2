import json

import numpy as np
import torch

from instrument_classifier.data.dataset import IRMASTestDataset
from instrument_classifier.evaluate import (
    clip_scores, evaluate_scores, gather_test_scores, windows_to_inputs,
)
from instrument_classifier.features import extract_all
from instrument_classifier.models.multibranch import MultiBranchNet
from instrument_classifier.windowing import sliding_windows
from conftest import SR, TEST_FC

ACTIVE = ["mel", "wave"]


def _stats(features_dir):
    return json.loads((features_dir / "stats.json").read_text())


def test_windows_to_inputs_matches_preprocess_path(preprocessed_features, synthetic_irmas):
    """Same clip through eval path == through preprocess path (float16 rounding aside)."""
    stats = _stats(preprocessed_features)
    npz_path = sorted(preprocessed_features.rglob("*.npz"))[0]
    wav_path = (synthetic_irmas / "IRMAS-TrainingData" / npz_path.parent.name /
                (npz_path.stem + ".wav"))
    from instrument_classifier.features import load_audio, normalize
    wav = torch.from_numpy(load_audio(wav_path, SR))
    windows = sliding_windows(wav, TEST_FC.clip_len, TEST_FC.clip_len)
    inputs = windows_to_inputs(windows, TEST_FC, stats, ["mel"])
    stored = np.load(npz_path)["mel"].astype(np.float32)
    expected = normalize({"mel": stored}, {"mel": stats["mel"]})["mel"]
    np.testing.assert_allclose(inputs["mel"][0, 0].numpy(), expected, atol=0.05)


def test_clip_scores_shape(preprocessed_features):
    model = MultiBranchNet({"mel": True, "cqt": False, "wave": True, "chroma": False},
                           pretrained=False).eval()
    wav = torch.rand(SR * 5) - 0.5
    scores = clip_scores(model, wav, torch.device("cpu"), TEST_FC,
                         _stats(preprocessed_features), ACTIVE,
                         window_len=TEST_FC.clip_len, hop_len=SR, aggregate="mean")
    assert scores.shape == (11,) and (0 <= scores).all() and (scores <= 1).all()


def test_gather_and_evaluate(preprocessed_features, synthetic_irmas):
    model = MultiBranchNet({"mel": True, "cqt": False, "wave": True, "chroma": False},
                           pretrained=False).eval()
    ds = IRMASTestDataset(synthetic_irmas / "IRMAS-TestingData", sample_rate=SR)
    y_true, y_scores = gather_test_scores(
        model, ds, torch.device("cpu"), TEST_FC, _stats(preprocessed_features),
        ACTIVE, TEST_FC.clip_len, SR, "mean", show_progress=False)
    assert y_true.shape == y_scores.shape == (2, 11)
    result = evaluate_scores(y_true, y_scores, threshold=0.5)
    assert {"micro_f1", "macro_f1", "per_class", "threshold"} <= set(result)
    assert len(result["per_class"]) == 11
