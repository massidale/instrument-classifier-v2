"""IRMAS datasets with lazy, on-demand audio loading.

Nothing is held in RAM up front: each ``__getitem__`` loads exactly one clip
from disk, converts it to mono at the model's sample rate, and (for training)
normalizes it to a fixed length. This scales to the full dataset unlike the
old "load every waveform into a numpy list" approach.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from .labels import IRMAS_CLASSES, encode_labels, label_to_index
from .transforms import pad_or_trim

_VALID_CODES = set(IRMAS_CLASSES)


def _load_mono_resampled(path: Path, sample_rate: int) -> torch.Tensor:
    """Load an audio file as a 1D mono float32 tensor at ``sample_rate``.

    Uses soundfile for decoding (no torchcodec dependency) and torchaudio only
    for high-quality resampling.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # (channels, frames)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0).to(torch.float32)


def parse_test_label_file(path: Path) -> list[str]:
    """Read an IRMAS test annotation file into a list of valid instrument codes.

    Annotations list one instrument per line; we split on any whitespace and
    keep only tokens that are among the 11 IRMAS classes.
    """
    text = Path(path).read_text()
    tokens = text.replace("\t", " ").split()
    return [tok for tok in tokens if tok in _VALID_CODES]


class IRMASTrainDataset(Dataset):
    """Single-label training clips, one instrument per folder.

    Returns ``(waveform[clip_len], target[11])`` where the target is a
    multi-hot vector (single 1 here) so it composes with mixup and BCE loss.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        sample_rate: int,
        clip_seconds: float,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.clip_len = int(round(sample_rate * clip_seconds))
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []
        for code in sorted(os.listdir(self.root)):
            class_dir = self.root / code
            if not class_dir.is_dir() or code not in _VALID_CODES:
                continue
            for wav in sorted(class_dir.glob("*.wav")):
                self.samples.append((wav, label_to_index(code)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, class_idx = self.samples[idx]
        wav = _load_mono_resampled(path, self.sample_rate)
        wav = pad_or_trim(wav, self.clip_len)
        if self.transform is not None:
            wav = self.transform(wav)
        target = torch.zeros(len(IRMAS_CLASSES), dtype=torch.float32)
        target[class_idx] = 1.0
        return wav, target

    def targets(self) -> list[int]:
        """Class index per sample — handy for a stratified train/val split."""
        return [class_idx for _, class_idx in self.samples]


class IRMASTestDataset(Dataset):
    """Variable-length polyphonic test clips with multi-label annotations.

    Returns ``(waveform[variable], target[11], name)``. Clips are NOT cropped;
    the sliding-window evaluation handles their variable length.
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
        wav = _load_mono_resampled(wav_path, self.sample_rate)
        codes = parse_test_label_file(txt_path)
        target = torch.from_numpy(encode_labels(codes))
        return wav, target, wav_path.stem
