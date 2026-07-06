"""IRMAS datasets. Training reads precomputed .npz features (lazy, per item,
only the branches the model actually uses); testing loads raw audio for
sliding-window evaluation with on-the-fly feature extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features import load_audio, normalize
from .labels import IRMAS_CLASSES, encode_labels, label_to_index

_VALID_CODES = set(IRMAS_CLASSES)
_IMAGE_KEYS = ("mel", "cqt", "chroma")  # get a leading channel dim


def parse_test_label_file(path: Path) -> list[str]:
    """IRMAS test annotations: one instrument code per line."""
    tokens = Path(path).read_text().replace("\t", " ").split()
    return [tok for tok in tokens if tok in _VALID_CODES]


class IRMASFeaturesDataset(Dataset):
    """Precomputed-feature training clips (folder-per-class of .npz files).

    Returns ``(features_dict, target)``: normalized float32 tensors, image-like
    features shaped (1, bins, T), waveform shaped (clip_len,), multi-hot target.
    """

    def __init__(
        self,
        features_dir: str | os.PathLike,
        branches: list[str],
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    ):
        self.root = Path(features_dir)
        self.branches = list(branches)
        self.transform = transform
        self.stats = json.loads((self.root / "stats.json").read_text())

        self.samples: list[tuple[Path, int]] = []
        for code in sorted(os.listdir(self.root)):
            class_dir = self.root / code
            if not class_dir.is_dir() or code not in _VALID_CODES:
                continue
            for npz in sorted(class_dir.glob("*.npz")):
                self.samples.append((npz, label_to_index(code)))
        if not self.samples:
            raise FileNotFoundError(
                f"No .npz features under {self.root} — run scripts/preprocess.py first")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        path, class_idx = self.samples[idx]
        with np.load(path) as npz:
            raw = {k: npz[k].astype(np.float32) for k in self.branches}
        normed = normalize(raw, self.stats)
        feats = {
            k: torch.from_numpy(v).unsqueeze(0) if k in _IMAGE_KEYS else torch.from_numpy(v)
            for k, v in normed.items()
        }
        if self.transform is not None:
            feats = self.transform(feats)
        target = torch.zeros(len(IRMAS_CLASSES), dtype=torch.float32)
        target[class_idx] = 1.0
        return feats, target

    def targets(self) -> list[int]:
        """Class index per sample — for the stratified train/val split."""
        return [class_idx for _, class_idx in self.samples]


class IRMASTestDataset(Dataset):
    """Variable-length polyphonic test clips with multi-label .txt annotations.

    Returns ``(waveform, target, name)``; windowing + feature extraction happen
    in evaluate.py so train/test share the exact same feature code path.
    """

    def __init__(self, root: str | os.PathLike, sample_rate: int):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.samples: list[tuple[Path, Path]] = []
        for wav in sorted(self.root.rglob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                self.samples.append((wav, txt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        wav_path, txt_path = self.samples[idx]
        wav = torch.from_numpy(load_audio(wav_path, self.sample_rate))
        target = torch.from_numpy(encode_labels(parse_test_label_file(txt_path)))
        return wav, target, wav_path.stem
