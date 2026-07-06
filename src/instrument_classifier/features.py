"""Pure feature extraction shared by preprocessing (offline) and evaluation
(on-the-fly per window). Keeping ONE implementation guarantees train/test
consistency. All functions are numpy-in / numpy-out; torch stays out of here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

FEATURE_KEYS = ("mel", "cqt", "wave", "chroma")


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int
    clip_seconds: float
    n_fft: int
    hop_length: int
    n_mels: int
    cqt_bins: int

    @property
    def clip_len(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))

    @property
    def n_frames(self) -> int:
        return 1 + self.clip_len // self.hop_length  # librosa center=True

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureConfig":
        f = cfg["features"]
        return cls(sample_rate=f["sample_rate"], clip_seconds=f["clip_seconds"],
                   n_fft=f["n_fft"], hop_length=f["hop_length"],
                   n_mels=f["n_mels"], cqt_bins=f["cqt_bins"])


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Mono float32 waveform at ``sample_rate`` (librosa handles resampling)."""
    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32)


def pad_or_trim_np(y: np.ndarray, length: int) -> np.ndarray:
    if y.shape[0] >= length:
        return y[:length]
    return np.pad(y, (0, length - y.shape[0]))


def logmel(y: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    s = librosa.feature.melspectrogram(
        y=y, sr=fc.sample_rate, n_fft=fc.n_fft,
        hop_length=fc.hop_length, n_mels=fc.n_mels)
    return librosa.power_to_db(s, ref=np.max).astype(np.float32)


def cqt(y: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    c = np.abs(librosa.cqt(y=y, sr=fc.sample_rate, hop_length=fc.hop_length,
                           n_bins=fc.cqt_bins, bins_per_octave=12))
    return librosa.amplitude_to_db(c, ref=np.max).astype(np.float32)


def chroma(y: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    return librosa.feature.chroma_stft(
        y=y, sr=fc.sample_rate, n_fft=fc.n_fft,
        hop_length=fc.hop_length).astype(np.float32)


_EXTRACTORS = {"mel": logmel, "cqt": cqt, "chroma": chroma}


def extract_all(y: np.ndarray, fc: FeatureConfig,
                keys: list[str] | tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Requested representations of one fixed-length clip.

    ``keys=None`` returns all four (preprocessing default); pass a subset to
    skip the expensive extractions (e.g. CQT/chroma) for inactive branches.
    The waveform is always padded/trimmed to ``fc.clip_len`` first, and "wave"
    is included in the output only when requested.
    """
    if keys is None:
        keys = FEATURE_KEYS
    unknown = set(keys) - set(FEATURE_KEYS)
    if unknown:
        raise ValueError(f"Unknown feature keys: {sorted(unknown)}")
    y = pad_or_trim_np(y.astype(np.float32), fc.clip_len)
    out: dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = y if k == "wave" else _EXTRACTORS[k](y, fc)
    return out


def normalize(feats: dict[str, np.ndarray], stats: dict) -> dict[str, np.ndarray]:
    """Standardize each feature with train-set scalar mean/std."""
    return {k: ((v - stats[k]["mean"]) / (stats[k]["std"] + 1e-8)).astype(np.float32)
            for k, v in feats.items()}
