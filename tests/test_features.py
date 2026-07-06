import numpy as np
import pytest
from instrument_classifier.features import (
    FEATURE_KEYS, FeatureConfig, extract_all, normalize, pad_or_trim_np,
)

FC = FeatureConfig(sample_rate=22050, clip_seconds=3.0, n_fft=2048,
                   hop_length=512, n_mels=128, cqt_bins=84)

def _sine(seconds=3.0, freq=440.0, sr=22050):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_derived_lengths():
    assert FC.clip_len == 66150
    assert FC.n_frames == 130

def test_extract_all_shapes_and_dtype():
    feats = extract_all(_sine(), FC)
    assert set(feats) == set(FEATURE_KEYS)
    assert feats["mel"].shape == (128, 130)
    assert feats["cqt"].shape == (84, 130)
    assert feats["chroma"].shape == (12, 130)
    assert feats["wave"].shape == (66150,)
    for v in feats.values():
        assert v.dtype == np.float32 and np.isfinite(v).all()

def test_extract_all_subset_keys():
    """Requesting a subset returns only those keys (skips unwanted CQT/chroma)."""
    feats = extract_all(_sine(), FC, keys=["mel"])
    assert set(feats) == {"mel"}
    assert feats["mel"].shape == (128, 130)
    assert feats["mel"].dtype == np.float32

def test_extract_all_subset_wave_is_padded():
    """A shorter clip requested via keys=['wave'] is still padded to clip_len."""
    feats = extract_all(_sine(seconds=1.0), FC, keys=["wave"])
    assert set(feats) == {"wave"}
    assert feats["wave"].shape == (FC.clip_len,)

def test_extract_all_is_deterministic():
    y = _sine()
    a, b = extract_all(y, FC), extract_all(y, FC)
    for k in FEATURE_KEYS:
        np.testing.assert_array_equal(a[k], b[k])

def test_pad_or_trim_np():
    assert pad_or_trim_np(np.ones(10, np.float32), 6).shape == (6,)
    padded = pad_or_trim_np(np.ones(4, np.float32), 6)
    assert padded.shape == (6,) and padded[4:].sum() == 0

def test_normalize_uses_stats():
    feats = extract_all(_sine(), FC)
    stats = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in feats.items()}
    normed = normalize(feats, stats)
    for k in FEATURE_KEYS:
        assert abs(float(normed[k].mean())) < 1e-3
