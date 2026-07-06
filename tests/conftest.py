"""Synthetic mini-IRMAS: sine waves at class-specific frequencies.

Session-scoped because feature extraction (CQT) is the slow part; every
data/model/train test reuses the same tree.
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from instrument_classifier.features import FeatureConfig

SR = 22050
CLASSES = ("cel", "flu", "pia")          # any 3 valid IRMAS codes
FREQS = {"cel": 220.0, "flu": 880.0, "pia": 440.0}

TEST_FC = FeatureConfig(sample_rate=SR, clip_seconds=3.0, n_fft=2048,
                        hop_length=512, n_mels=128, cqt_bins=84)


def _sine(freq: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_irmas(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("irmas")
    train = root / "IRMAS-TrainingData"
    for code in CLASSES:
        d = train / code
        d.mkdir(parents=True)
        for i in range(4):
            sf.write(d / f"{code}_{i}.wav", _sine(FREQS[code] * (1 + 0.01 * i)), SR)
    test = root / "IRMAS-TestingData"
    test.mkdir()
    sf.write(test / "poly1.wav", _sine(220.0, 5.0) + _sine(880.0, 5.0), SR)
    (test / "poly1.txt").write_text("cel\nflu\n")
    sf.write(test / "poly2.wav", _sine(440.0, 4.0), SR)
    (test / "poly2.txt").write_text("pia\n")
    return root


@pytest.fixture(scope="session")
def preprocessed_features(synthetic_irmas) -> Path:
    from scripts_preprocess_shim import preprocess_dataset  # see step 3
    out = synthetic_irmas / "features"
    preprocess_dataset(synthetic_irmas / "IRMAS-TrainingData", out, TEST_FC)
    return out
