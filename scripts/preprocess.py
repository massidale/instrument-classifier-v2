#!/usr/bin/env python
"""Precompute mel/CQT/chroma/waveform features for IRMAS training clips.

Writes one float16 .npz per clip (mirroring the class-folder layout) plus
train-set normalization stats (stats.json). Run once before training:

    python scripts/preprocess.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from instrument_classifier.data.labels import IRMAS_CLASSES
from instrument_classifier.features import (
    FEATURE_KEYS, FeatureConfig, extract_all, load_audio,
)
from instrument_classifier.utils import load_config

_VALID = set(IRMAS_CLASSES)


def preprocess_dataset(train_dir: Path, out_dir: Path, fc: FeatureConfig) -> dict:
    """Extract features for every training wav; return (and save) norm stats."""
    train_dir, out_dir = Path(train_dir), Path(out_dir)
    acc = {k: [0.0, 0.0, 0] for k in FEATURE_KEYS}  # sum, sum_sq, count

    wavs = [w for d in sorted(train_dir.iterdir()) if d.is_dir() and d.name in _VALID
            for w in sorted(d.glob("*.wav"))]
    if not wavs:
        raise SystemExit(f"No IRMAS class folders with .wav found in {train_dir}")

    for wav in tqdm(wavs, desc="preprocess"):
        feats = extract_all(load_audio(wav, fc.sample_rate), fc)
        dest = out_dir / wav.parent.name / (wav.stem + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, **{k: v.astype(np.float16) for k, v in feats.items()})
        for k, v in feats.items():
            v64 = v.astype(np.float64)
            acc[k][0] += float(v64.sum())
            acc[k][1] += float((v64 ** 2).sum())
            acc[k][2] += v64.size

    stats = {}
    for k, (s, sq, n) in acc.items():
        mean = s / n
        stats[k] = {"mean": mean, "std": float(np.sqrt(max(sq / n - mean ** 2, 1e-12)))}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    fc = FeatureConfig.from_config(cfg)
    stats = preprocess_dataset(Path(cfg["data"]["train_dir"]),
                               Path(cfg["data"]["features_dir"]) / "train", fc)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
