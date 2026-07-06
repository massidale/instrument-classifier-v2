import json

import numpy as np
import pytest
import torch

from instrument_classifier.data.dataset import IRMASTestDataset
from instrument_classifier.evaluate import (
    clip_scores, evaluate_from_config, evaluate_scores, gather_test_scores,
    windows_to_inputs,
)
from instrument_classifier.features import extract_all
from instrument_classifier.models.multibranch import MultiBranchNet
from instrument_classifier.windowing import sliding_windows
from conftest import SR, TEST_FC

ACTIVE = ["mel", "wave"]
BRANCHES = {"mel": True, "cqt": False, "wave": True, "chroma": False}


def _stats(features_dir):
    return json.loads((features_dir / "stats.json").read_text())


def _features_cfg():
    return {"sample_rate": TEST_FC.sample_rate, "clip_seconds": TEST_FC.clip_seconds,
            "n_fft": TEST_FC.n_fft, "hop_length": TEST_FC.hop_length,
            "n_mels": TEST_FC.n_mels, "cqt_bins": TEST_FC.cqt_bins}


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


def test_clip_scores_rejects_mismatched_window(preprocessed_features):
    """window_len must equal clip_len, else extract_all would silently truncate."""
    model = MultiBranchNet(BRANCHES, pretrained=False).eval()
    wav = torch.rand(SR * 5) - 0.5
    with pytest.raises(AssertionError, match="window_seconds must equal"):
        clip_scores(model, wav, torch.device("cpu"), TEST_FC,
                    _stats(preprocessed_features), ACTIVE,
                    window_len=SR * 5, hop_len=SR, aggregate="mean")


def test_evaluate_from_config_uses_checkpoint_model_config(
        preprocessed_features, synthetic_irmas, tmp_path):
    """Checkpoint's model_config (non-default head_hidden) drives model build,
    not the current config's — a real round-trip through torch.save/load."""
    model = MultiBranchNet(BRANCHES, num_classes=11, pretrained=False,
                           head_hidden=77, dropout=0.2)
    ckpt_path = tmp_path / "best.pth"
    torch.save({"model": model.state_dict(), "branches": BRANCHES,
                "stats": _stats(preprocessed_features), "threshold": 0.5,
                "val_micro_f1": 0.0,
                "model_config": {"num_classes": 11, "head_hidden": 77,
                                 "dropout": 0.2}}, ckpt_path)
    config = {
        "device": "cpu",
        # Deliberately different from the checkpoint: if used, load_state_dict
        # would raise a size mismatch.
        "model": {"num_classes": 11, "head_hidden": 512, "dropout": 0.3},
        "data": {"test_dir": str(synthetic_irmas / "IRMAS-TestingData")},
        "features": _features_cfg(),
        "eval": {"window_seconds": 3.0, "hop_seconds": 1.0, "aggregate": "mean",
                 "default_threshold": 0.5},
    }
    result = evaluate_from_config(config, ckpt_path)
    assert {"micro_f1", "macro_f1", "per_class", "threshold"} <= set(result)


def test_evaluate_from_config_missing_test_dir(preprocessed_features, tmp_path):
    """A missing/empty test dir raises a clear error, not an opaque np.stack."""
    model = MultiBranchNet(BRANCHES, num_classes=11, pretrained=False)
    ckpt_path = tmp_path / "best.pth"
    torch.save({"model": model.state_dict(), "branches": BRANCHES,
                "stats": _stats(preprocessed_features), "threshold": 0.5,
                "val_micro_f1": 0.0,
                "model_config": {"num_classes": 11, "head_hidden": 512,
                                 "dropout": 0.3}}, ckpt_path)
    empty = tmp_path / "no_test_data"
    empty.mkdir()
    config = {
        "device": "cpu",
        "model": {"num_classes": 11, "head_hidden": 512, "dropout": 0.3},
        "data": {"test_dir": str(empty)},
        "features": _features_cfg(),
        "eval": {"window_seconds": 3.0, "hop_seconds": 1.0, "aggregate": "mean",
                 "default_threshold": 0.5},
    }
    with pytest.raises((SystemExit, ValueError), match="test set"):
        evaluate_from_config(config, ckpt_path)
