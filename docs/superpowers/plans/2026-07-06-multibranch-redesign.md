# MultiBranchNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the PANNs CNN14 finetuning pipeline with a multi-input CNN (mel, CQT, waveform, chroma branches — each toggleable from config) built on ImageNet-pretrained ResNet18 backbones, with explicit on-disk feature preprocessing and a per-branch ablation study, keeping the official IRMAS evaluation protocol unchanged.

**Architecture:** Four independent branches (mel→ResNet18, CQT→ResNet18, waveform→custom Conv1D, chroma→small Conv2D) produce embeddings that are concatenated and fed to an MLP head emitting 11 logits (`BCEWithLogitsLoss`, multi-label). Training features are precomputed to `.npz` by `scripts/preprocess.py`; evaluation computes features on the fly per sliding window using the same functions from `features.py`. Two-phase training: frozen backbones warmup, then full finetuning with discriminative LRs.

**Tech Stack:** PyTorch ≥2.2, torchvision ≥0.17 (ResNet18 IMAGENET1K_V1), librosa ≥0.10 (mel/CQT/chroma/load), soundfile, scikit-learn, pytest. `torchlibrosa`, `torchaudio` and `tensorboard` are dropped.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-04-multibranch-redesign-design.md`. Deviations require user approval.
- Sample rate **22050 Hz mono**; clip length **3.0 s** → **66150 samples**.
- Feature params: log-mel `n_mels=128, n_fft=2048, hop_length=512`; CQT `n_bins=84, bins_per_octave=12, hop_length=512`, in dB; chroma `12 bins, hop 512`. With `center=True` all give **130 frames** for a 3 s clip.
- Feature dict keys are exactly `"mel"`, `"cqt"`, `"wave"`, `"chroma"` — everywhere (npz, dataset, model, config).
- `.npz` stored **float16**; normalization stats (scalar mean/std per feature) computed **on the training set only**, saved to `data/features/stats.json`.
- Multi-label: 11 classes (`IRMAS_CLASSES` order from `labels.py`), `BCEWithLogitsLoss`, sigmoid at inference.
- IRMAS eval protocol unchanged: sliding window 3 s / hop 1 s, `mean` aggregation, threshold tuned on validation. Reuse `windowing.py`, `metrics.py`, `labels.py`, `utils.py` as-is (metrics gains one additive function).
- Reproducibility: global seed 42, seeded `torch.Generator` for all augmentation, ablation runs differ ONLY in `branches`.
- Tests must not download weights or data: model tests use `pretrained=False`, data tests use synthetic sine-wave fixtures.
- Every commit message in English, imperative mood, no attribution footer needed beyond the repo's convention.

## File Structure

```
src/instrument_classifier/
  features.py            NEW    pure feature extraction + FeatureConfig + normalize + load_audio
  data/dataset.py        REWRITE IRMASFeaturesDataset (npz) + IRMASTestDataset (raw audio, kept)
  data/transforms.py     REWRITE SpecAugment + mixup_multi (multi-input)
  data/labels.py         KEEP
  models/multibranch.py  NEW    branches + MultiBranchNet
  models/cnn14.py        DELETE (final task)
  metrics.py             MODIFY  add per_class_f1
  windowing.py           KEEP
  utils.py               KEEP
  train.py               REWRITE two-phase training on feature dicts
  evaluate.py            REWRITE windowed eval with on-the-fly features
scripts/
  preprocess.py          NEW    wav → npz + stats.json
  run_ablation.py        NEW    branch-combination sweep → outputs/ablation.md
  download_pretrained.py DELETE (final task)
  train.py / evaluate.py KEEP   (thin CLI wrappers, unchanged)
tests/
  conftest.py            NEW    synthetic IRMAS tree + preprocessed features fixtures
  test_features.py       NEW
  test_preprocess.py     NEW
  test_dataset.py        REWRITE
  test_transforms.py     REWRITE
  test_multibranch.py    NEW (replaces test_model.py)
  test_metrics.py        EXTEND
  test_evaluate.py       NEW
  test_training_integration.py REWRITE
  test_ablation.py       NEW
  test_labels.py / test_windowing.py KEEP
configs/default.yaml     REWRITE
notebooks/colab_train.ipynb REWRITE
README.md                REWRITE (final task)
pyproject.toml / requirements.txt MODIFY (task 1)
```

Existing interfaces this plan builds on (already in the repo, unchanged):

- `labels.py`: `IRMAS_CLASSES: list[str]` (11 codes), `label_to_index(code) -> int`, `encode_labels(codes) -> np.ndarray[(11,), float32]`
- `metrics.py`: `multilabel_metrics(y_true, y_pred) -> dict` (micro/macro P/R/F1), `tune_threshold(y_true, y_scores, candidates) -> tuple[float, float]`
- `windowing.py`: `sliding_windows(waveform_1d, window_len, hop_len) -> Tensor[(W, window_len)]`, `aggregate_scores(probs_WxC, method) -> Tensor[(C,)]`
- `utils.py`: `load_config(path) -> dict`, `resolve_device(name) -> torch.device`, `set_seed(seed)`, `save_checkpoint(path, model, extra)`, `save_metrics(path, dict)`

---

### Task 1: Dependencies and new config

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Rewrite: `configs/default.yaml`
- Test: `tests/test_config.py` (new)

**Interfaces:**
- Produces: `configs/default.yaml` with top-level keys `seed, device, output_dir, data, features, branches, model, augment, train, eval` — the exact schema every later task reads.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from instrument_classifier.utils import load_config

def test_default_config_schema():
    cfg = load_config("configs/default.yaml")
    assert set(cfg["branches"]) == {"mel", "cqt", "wave", "chroma"}
    assert all(isinstance(v, bool) for v in cfg["branches"].values())
    f = cfg["features"]
    assert f["sample_rate"] == 22050 and f["clip_seconds"] == 3.0
    assert f["n_fft"] == 2048 and f["hop_length"] == 512
    assert f["n_mels"] == 128 and f["cqt_bins"] == 84
    assert cfg["model"]["num_classes"] == 11
    assert cfg["data"]["features_dir"] == "data/features"
    assert cfg["eval"]["window_seconds"] == 3.0 and cfg["eval"]["hop_seconds"] == 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_config.py -v` — Expected: FAIL (KeyError `branches`).

- [ ] **Step 3: Rewrite `configs/default.yaml`**

```yaml
# ---------------------------------------------------------------------------
# MultiBranchNet on IRMAS — every hyperparameter lives here.
# ---------------------------------------------------------------------------
seed: 42
device: auto          # auto | cpu | cuda | mps
output_dir: outputs

data:
  train_dir: data/IRMAS-TrainingData
  test_dir: data/IRMAS-TestingData
  features_dir: data/features        # output of scripts/preprocess.py
  val_fraction: 0.15
  num_workers: 2

features:                            # shared by preprocessing AND eval
  sample_rate: 22050
  clip_seconds: 3.0
  n_fft: 2048
  hop_length: 512
  n_mels: 128
  cqt_bins: 84                       # 7 octaves x 12 bins

branches:                            # the ablation axis: toggle freely
  mel: true
  cqt: true
  wave: true
  chroma: true

model:
  num_classes: 11
  pretrained: true                   # ImageNet weights for mel/CQT ResNet18
  head_hidden: 512
  dropout: 0.3

augment:
  specaugment:
    enabled: true
    time_masks: 2
    time_width: 20                   # frames
    freq_masks: 2
    freq_width: 12                   # bins
  mixup_alpha: 0.4                   # 0 disables

train:
  batch_size: 32
  warmup_epochs: 3                   # backbones frozen
  warmup_lr: 1.0e-3
  finetune_epochs: 25
  head_lr: 5.0e-4
  backbone_lr: 5.0e-5
  weight_decay: 1.0e-4
  early_stopping_patience: 6

eval:
  window_seconds: 3.0
  hop_seconds: 1.0
  aggregate: mean                    # mean | max
  default_threshold: 0.5
```

- [ ] **Step 4: Update dependencies**

In `pyproject.toml` `[project].dependencies` replace the list with:

```toml
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "numpy>=1.24",
    "librosa>=0.10",
    "soundfile>=0.12",
    "scikit-learn>=1.3",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "matplotlib>=3.7",
]
```

(removes `torchaudio`, `torchlibrosa`, `tensorboard`; NB: `numpy>=1.24` not `>=2.0` because librosa/numba lag on numpy 2). Update the `description` to "Musical instrument recognition on IRMAS with a multi-input CNN (mel/CQT/waveform/chroma) and ImageNet-pretrained backbones". Mirror the same list in `requirements.txt`. Keep `[project.scripts]` as-is.

Also change `[tool.pytest.ini_options]` so that `tests/` modules (conftest, shims) are importable as plain modules (`from conftest import TEST_FC`) regardless of pytest import mode:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "tests"]
```

- [ ] **Step 5: Reinstall and run test**

Run: `pip install -e . && pytest tests/test_config.py -v` — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt configs/default.yaml tests/test_config.py
git commit -m "Switch dependencies and config schema to multi-branch design"
```

---

### Task 2: Feature extraction module

**Files:**
- Create: `src/instrument_classifier/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `config["features"]` dict from Task 1.
- Produces (used by preprocess, dataset, evaluate):
  - `FeatureConfig` dataclass: fields `sample_rate:int, clip_seconds:float, n_fft:int, hop_length:int, n_mels:int, cqt_bins:int`; properties `clip_len:int`, `n_frames:int`; classmethod `FeatureConfig.from_config(cfg: dict) -> FeatureConfig` (takes the WHOLE config dict, reads `cfg["features"]`).
  - `FEATURE_KEYS = ("mel", "cqt", "wave", "chroma")`
  - `load_audio(path, sample_rate) -> np.ndarray` 1-D float32 mono
  - `pad_or_trim_np(y, length) -> np.ndarray`
  - `extract_all(y, fc) -> dict[str, np.ndarray]` with shapes mel `(128,130)`, cqt `(84,130)`, chroma `(12,130)`, wave `(66150,)`, all float32
  - `normalize(feats: dict, stats: dict) -> dict[str, np.ndarray]` — `(x - mean) / (std + 1e-8)` per key; `stats[key] = {"mean": float, "std": float}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
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
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_features.py -v` — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `features.py`**

```python
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


def extract_all(y: np.ndarray, fc: FeatureConfig) -> dict[str, np.ndarray]:
    """All four representations of one fixed-length clip."""
    y = pad_or_trim_np(y.astype(np.float32), fc.clip_len)
    return {"mel": logmel(y, fc), "cqt": cqt(y, fc),
            "chroma": chroma(y, fc), "wave": y}


def normalize(feats: dict[str, np.ndarray], stats: dict) -> dict[str, np.ndarray]:
    """Standardize each feature with train-set scalar mean/std."""
    return {k: ((v - stats[k]["mean"]) / (stats[k]["std"] + 1e-8)).astype(np.float32)
            for k, v in feats.items()}
```

- [ ] **Step 4: Run tests** — `pytest tests/test_features.py -v` — Expected: PASS (first run is slow: numba JIT for CQT).

- [ ] **Step 5: Commit**

```bash
git add src/instrument_classifier/features.py tests/test_features.py
git commit -m "Add shared feature extraction module (mel, CQT, chroma, wave)"
```

---

### Task 3: Preprocessing script + synthetic-data fixtures

**Files:**
- Create: `scripts/preprocess.py`
- Create: `tests/conftest.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `extract_all`, `load_audio`, `FeatureConfig`, `FEATURE_KEYS` (Task 2); `IRMAS_CLASSES` from `labels.py`.
- Produces:
  - `preprocess_dataset(train_dir: Path, out_dir: Path, fc: FeatureConfig) -> dict` — writes `out_dir/<class>/<stem>.npz` (float16, keys = FEATURE_KEYS) mirroring the class-folder layout, writes and returns `stats` dict `{key: {"mean", "std"}}`, saved to `out_dir/stats.json`.
  - Pytest fixtures `synthetic_irmas(tmp_path_factory)` (session-scoped: 3 classes × 4 wavs of distinct frequencies + a 2-clip fake test set with `.txt` labels) and `preprocessed_features(synthetic_irmas)` — session-scoped so the expensive CQT runs once for the whole suite.

- [ ] **Step 1: Write fixtures**

```python
# tests/conftest.py
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
```

NOTE for step 3: `scripts/` is not a package. Make `preprocess_dataset` importable by placing the logic in `scripts/preprocess.py` AND adding `scripts` to test path via a tiny shim `tests/scripts_preprocess_shim.py`:

```python
# tests/scripts_preprocess_shim.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preprocess import preprocess_dataset  # noqa: E402,F401
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_preprocess.py
import json

import numpy as np

from instrument_classifier.features import FEATURE_KEYS
from conftest import CLASSES, TEST_FC


def test_preprocess_writes_npz_and_stats(preprocessed_features):
    stats = json.loads((preprocessed_features / "stats.json").read_text())
    assert set(stats) == set(FEATURE_KEYS)
    for k in FEATURE_KEYS:
        assert set(stats[k]) == {"mean", "std"} and stats[k]["std"] > 0

    npzs = sorted(preprocessed_features.rglob("*.npz"))
    assert len(npzs) == len(CLASSES) * 4
    sample = np.load(npzs[0])
    assert set(sample.files) == set(FEATURE_KEYS)
    assert sample["mel"].dtype == np.float16
    assert sample["mel"].shape == (128, TEST_FC.n_frames)
    assert sample["wave"].shape == (TEST_FC.clip_len,)
```

Run: `pytest tests/test_preprocess.py -v` — Expected: FAIL (no `scripts/preprocess.py`).

- [ ] **Step 3: Implement `scripts/preprocess.py`**

```python
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
```

NOTE: the features dir layout is `data/features/train/<class>/<stem>.npz` with `stats.json` at `data/features/train/stats.json` (the `preprocess_dataset` out_dir IS the train features dir — config `features_dir` + `/train` is composed in `main` and later in `train.py`).

- [ ] **Step 4: Run tests** — `pytest tests/test_preprocess.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/preprocess.py tests/conftest.py tests/scripts_preprocess_shim.py tests/test_preprocess.py
git commit -m "Add offline preprocessing to npz with train-set norm stats"
```

---

### Task 4: Features dataset (rewrite `data/dataset.py`)

**Files:**
- Rewrite: `src/instrument_classifier/data/dataset.py`
- Rewrite: `tests/test_dataset.py`

**Interfaces:**
- Consumes: features dir layout + `stats.json` (Task 3), `normalize`/`FEATURE_KEYS` (Task 2), `labels.py`.
- Produces:
  - `IRMASFeaturesDataset(features_dir, branches: list[str], transform=None)` — `__getitem__ -> tuple[dict[str, Tensor], Tensor]`. Feature tensors: `"mel"` `(1,128,T)`, `"cqt"` `(1,84,T)`, `"chroma"` `(1,12,T)` (channel dim added), `"wave"` `(clip_len,)`; all float32, normalized with stats. Target `(11,)` multi-hot float32. Loads ONLY the keys in `branches`. Method `targets() -> list[int]`. Attribute `stats: dict`.
  - `IRMASTestDataset(root, sample_rate)` — kept behavior: `__getitem__ -> (waveform_1d, target_11, name)`, but audio loading now via `features.load_audio` (librosa) for train/eval consistency.
  - `parse_test_label_file(path) -> list[str]` — kept as today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
import torch

from instrument_classifier.data.dataset import IRMASFeaturesDataset, IRMASTestDataset
from conftest import CLASSES, SR, TEST_FC


def test_features_dataset_shapes(preprocessed_features):
    ds = IRMASFeaturesDataset(preprocessed_features, branches=["mel", "cqt", "wave", "chroma"])
    assert len(ds) == len(CLASSES) * 4
    feats, target = ds[0]
    assert set(feats) == {"mel", "cqt", "wave", "chroma"}
    assert feats["mel"].shape == (1, 128, TEST_FC.n_frames)
    assert feats["cqt"].shape == (1, 84, TEST_FC.n_frames)
    assert feats["chroma"].shape == (1, 12, TEST_FC.n_frames)
    assert feats["wave"].shape == (TEST_FC.clip_len,)
    assert all(v.dtype == torch.float32 for v in feats.values())
    assert target.shape == (11,) and target.sum() == 1.0


def test_features_dataset_loads_only_active_branches(preprocessed_features):
    ds = IRMASFeaturesDataset(preprocessed_features, branches=["mel"])
    feats, _ = ds[0]
    assert set(feats) == {"mel"}


def test_features_dataset_targets_for_stratify(preprocessed_features):
    ds = IRMASFeaturesDataset(preprocessed_features, branches=["mel"])
    assert len(ds.targets()) == len(ds) and len(set(ds.targets())) == len(CLASSES)


def test_test_dataset_multilabel(synthetic_irmas):
    ds = IRMASTestDataset(synthetic_irmas / "IRMAS-TestingData", sample_rate=SR)
    wav, target, name = ds[0]
    assert wav.ndim == 1 and target.shape == (11,)
    assert target.sum() == 2.0 and name == "poly1"
```

Run: `pytest tests/test_dataset.py -v` — Expected: FAIL (no `IRMASFeaturesDataset`).

- [ ] **Step 2: Rewrite `data/dataset.py`**

```python
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
```

- [ ] **Step 3: Run tests** — `pytest tests/test_dataset.py -v` — Expected: PASS. Also run `pytest tests/test_labels.py tests/test_windowing.py -v` to confirm nothing broke.

- [ ] **Step 4: Commit**

```bash
git add src/instrument_classifier/data/dataset.py tests/test_dataset.py
git commit -m "Rewrite datasets: npz feature loading + librosa test audio"
```

---

### Task 5: Transforms rewrite — SpecAugment + multi-input mixup

**Files:**
- Rewrite: `src/instrument_classifier/data/transforms.py`
- Rewrite: `tests/test_transforms.py`

**Interfaces:**
- Consumes: feature dict convention (Task 4).
- Produces (used by train.py):
  - `SpecAugment(time_masks:int, time_width:int, freq_masks:int, freq_width:int, seed:int=0)` — callable `dict -> dict`; masks (sets to 0.0) random time/freq stripes on `"mel"` and `"cqt"` ONLY; other keys pass through untouched; picklable (top-level class) for DataLoader workers.
  - `mixup_batch(feats: dict[str, Tensor], targets: Tensor, alpha: float, generator=None) -> tuple[dict, Tensor]` — ONE `lam ~ Beta(alpha,alpha)` and ONE permutation applied to every tensor in the dict and to targets; `alpha <= 0` returns inputs unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transforms.py
import torch

from instrument_classifier.data.transforms import SpecAugment, mixup_batch


def _feats(b=0):  # deterministic fake batchless features
    g = torch.Generator().manual_seed(7 + b)
    return {"mel": torch.rand(1, 128, 130, generator=g) + 1.0,
            "cqt": torch.rand(1, 84, 130, generator=g) + 1.0,
            "wave": torch.rand(66150, generator=g) + 1.0}


def test_specaugment_masks_only_spectrograms():
    aug = SpecAugment(time_masks=2, time_width=20, freq_masks=2, freq_width=12, seed=0)
    feats = _feats()
    out = aug({k: v.clone() for k, v in feats.items()})
    assert (out["mel"] == 0).any() and (out["cqt"] == 0).any()      # masked
    assert torch.equal(out["wave"], feats["wave"])                   # untouched
    assert out["mel"].shape == feats["mel"].shape


def test_specaugment_reproducible_with_seed():
    a = SpecAugment(2, 20, 2, 12, seed=5)({k: v.clone() for k, v in _feats().items()})
    b = SpecAugment(2, 20, 2, 12, seed=5)({k: v.clone() for k, v in _feats().items()})
    assert torch.equal(a["mel"], b["mel"])


def test_mixup_consistent_lambda_across_inputs():
    g = torch.Generator().manual_seed(0)
    feats = {"mel": torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4)]),
             "wave": torch.stack([torch.zeros(8), torch.ones(8)])}
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mixed, my = mixup_batch(feats, targets, alpha=0.4, generator=g)
    lam_mel = float(mixed["mel"][0].mean())   # 0*lam + 1*(1-lam) or unchanged
    lam_wave = float(mixed["wave"][0].mean())
    assert abs(lam_mel - lam_wave) < 1e-6                     # same lam everywhere
    assert torch.allclose(my.sum(dim=1), torch.ones(2))       # targets stay convex


def test_mixup_alpha_zero_is_noop():
    feats = {"mel": torch.rand(2, 1, 4, 4)}
    targets = torch.rand(2, 11)
    mixed, my = mixup_batch(feats, targets, alpha=0.0)
    assert torch.equal(mixed["mel"], feats["mel"]) and torch.equal(my, targets)
```

Run: `pytest tests/test_transforms.py -v` — Expected: FAIL.

- [ ] **Step 2: Rewrite `transforms.py`**

```python
"""Augmentation for precomputed multi-input features.

SpecAugment operates on the stored spectrograms at training time; mixup mixes
every active input with the SAME lambda so the fused sample stays coherent.
Waveform gain/noise augmentation from the old pipeline is gone: it cannot be
applied meaningfully to precomputed log-spectrograms."""

from __future__ import annotations

import torch

_SPEC_KEYS = ("mel", "cqt")  # SpecAugment targets


class SpecAugment:
    """Zero out random time/frequency stripes on mel and CQT tensors (1, F, T).

    Picklable and top-level so it survives DataLoader worker processes."""

    def __init__(self, time_masks: int, time_width: int,
                 freq_masks: int, freq_width: int, seed: int = 0):
        self.time_masks, self.time_width = time_masks, time_width
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.generator = torch.Generator().manual_seed(seed)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        _, n_freq, n_time = x.shape
        for _ in range(self.freq_masks):
            w = min(self.freq_width, n_freq)
            f0 = int(torch.randint(0, n_freq - w + 1, (1,), generator=self.generator))
            x[:, f0:f0 + w, :] = 0.0
        for _ in range(self.time_masks):
            w = min(self.time_width, n_time)
            t0 = int(torch.randint(0, n_time - w + 1, (1,), generator=self.generator))
            x[:, :, t0:t0 + w] = 0.0
        return x

    def __call__(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: self._mask(v.clone()) if k in _SPEC_KEYS else v
                for k, v in feats.items()}


def mixup_batch(
    feats: dict[str, torch.Tensor],
    targets: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Batch mixup with one lambda and one permutation shared by all inputs.

    Multi-hot targets become convex combinations — still valid for BCE, and
    they expose the network to two-instrument mixtures during training."""
    if alpha <= 0.0:
        return feats, targets
    g1 = torch._standard_gamma(torch.full((1,), alpha), generator=generator)
    g2 = torch._standard_gamma(torch.full((1,), alpha), generator=generator)
    lam = float((g1 / (g1 + g2)).item())
    batch = targets.shape[0]
    perm = torch.randperm(batch, generator=generator)
    mixed = {k: lam * v + (1.0 - lam) * v[perm] for k, v in feats.items()}
    return mixed, lam * targets + (1.0 - lam) * targets[perm]
```

- [ ] **Step 3: Run tests** — `pytest tests/test_transforms.py -v` — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/instrument_classifier/data/transforms.py tests/test_transforms.py
git commit -m "Replace waveform augmentation with SpecAugment + multi-input mixup"
```

---

### Task 6: MultiBranchNet model

**Files:**
- Create: `src/instrument_classifier/models/multibranch.py`
- Create: `tests/test_multibranch.py`
- Delete: `tests/test_model.py` (CNN14 tests, superseded)

**Interfaces:**
- Consumes: feature dict convention; `config["branches"]`, `config["model"]`.
- Produces (used by train/evaluate/ablation):
  - `BRANCH_DIMS = {"mel": 512, "cqt": 512, "wave": 256, "chroma": 128}`
  - `MultiBranchNet(branches: dict[str, bool], num_classes: int = 11, pretrained: bool = True, head_hidden: int = 512, dropout: float = 0.3)`; raises `ValueError` if no branch is active. Attribute `active: list[str]` (sorted, deterministic order).
  - `forward(inputs: dict[str, Tensor]) -> dict` with `"logits"` `(B, num_classes)` and `"embedding"` `(B, sum(active dims))`.
  - `freeze_backbones()` / `unfreeze_backbones()` — only the ResNet18 `.backbone` params of mel/cqt branches (input BNs and custom branches stay trainable).
  - `param_groups(backbone_lr, rest_lr) -> list[dict]` — group 1 = trainable ResNet backbone params, group 2 = every other trainable param.
  - `build_model(config: dict) -> MultiBranchNet` — convenience from full config.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multibranch.py
import itertools

import pytest
import torch

from instrument_classifier.models.multibranch import BRANCH_DIMS, MultiBranchNet

ALL = ("mel", "cqt", "wave", "chroma")


def _inputs(batch=2, t=130):
    return {"mel": torch.rand(batch, 1, 128, t), "cqt": torch.rand(batch, 1, 84, t),
            "chroma": torch.rand(batch, 1, 12, t), "wave": torch.rand(batch, 66150)}


def _combos():
    for r in range(1, len(ALL) + 1):
        yield from itertools.combinations(ALL, r)


@pytest.mark.parametrize("combo", list(_combos()), ids="+".join)
def test_forward_every_branch_combination(combo):
    model = MultiBranchNet({k: k in combo for k in ALL}, pretrained=False).eval()
    out = model({k: v for k, v in _inputs().items() if k in combo})
    assert out["logits"].shape == (2, 11)
    assert out["embedding"].shape == (2, sum(BRANCH_DIMS[k] for k in combo))


def test_no_active_branch_raises():
    with pytest.raises(ValueError):
        MultiBranchNet({k: False for k in ALL}, pretrained=False)


def test_freeze_unfreeze_backbones():
    model = MultiBranchNet({"mel": True, "cqt": False, "wave": True, "chroma": False},
                           pretrained=False)
    model.freeze_backbones()
    backbone = [p for p in model.branches["mel"].backbone.parameters()]
    assert not any(p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in model.head.parameters())
    assert all(p.requires_grad for p in model.branches["wave"].parameters())
    model.unfreeze_backbones()
    assert all(p.requires_grad for p in backbone)


def test_param_groups_cover_all_trainables():
    model = MultiBranchNet({k: True for k in ALL}, pretrained=False)
    groups = model.param_groups(backbone_lr=1e-5, rest_lr=1e-3)
    ids_in_groups = {id(p) for g in groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert ids_in_groups == trainable
    assert groups[0]["lr"] == 1e-5 and groups[1]["lr"] == 1e-3
```

Run: `pytest tests/test_multibranch.py -v` — Expected: FAIL (module missing).

- [ ] **Step 2: Implement `models/multibranch.py`**

```python
"""MultiBranchNet: a multi-input CNN for instrument recognition.

Each audio representation gets its own branch; embeddings are concatenated
(late fusion) and classified by an MLP head with 11 independent sigmoid
outputs (multi-label). Branches are toggleable from config, which is what
makes the per-branch ablation study possible.

  mel    (1,128,T) -> ResNet18 (ImageNet) -> 512
  cqt    (1, 84,T) -> ResNet18 (ImageNet) -> 512
  wave   (66150,)  -> Conv1D stack (scratch) -> 256
  chroma (1, 12,T) -> small Conv2D (scratch) -> 128
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

BRANCH_DIMS = {"mel": 512, "cqt": 512, "wave": 256, "chroma": 128}


class ImageBranch(nn.Module):
    """Spectrogram-as-image branch: input BN + 1->3 channel repeat + ResNet18.

    Repeating the mono channel keeps the pretrained first conv intact; the
    input BatchNorm adapts spectrogram statistics to what the backbone expects."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(1)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()  # expose the 512-d pooled embedding
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, F, T)
        return self.backbone(self.bn_in(x).repeat(1, 3, 1, 1))


class WaveBranch(nn.Module):
    """Raw-waveform branch trained from scratch: 5 conv blocks, ~4x downsample each."""

    def __init__(self):
        super().__init__()
        chans = [1, 32, 64, 128, 256, 256]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv1d(cin, cout, kernel_size=9, padding=4, bias=False),
                       nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.MaxPool1d(4)]
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, samples)
        return self.pool(self.net(x.unsqueeze(1))).squeeze(-1)


class ChromaBranch(nn.Module):
    """Small 2D CNN for the 12-bin chroma map."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, 12, T)
        return self.net(x).flatten(1)


_BRANCH_FACTORIES = {
    "mel": lambda pretrained: ImageBranch(pretrained),
    "cqt": lambda pretrained: ImageBranch(pretrained),
    "wave": lambda pretrained: WaveBranch(),
    "chroma": lambda pretrained: ChromaBranch(),
}


class MultiBranchNet(nn.Module):
    def __init__(self, branches: dict[str, bool], num_classes: int = 11,
                 pretrained: bool = True, head_hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.active = sorted(k for k, on in branches.items() if on)
        if not self.active:
            raise ValueError("MultiBranchNet needs at least one active branch")
        unknown = set(self.active) - set(_BRANCH_FACTORIES)
        if unknown:
            raise ValueError(f"Unknown branches: {sorted(unknown)}")
        self.branches = nn.ModuleDict(
            {k: _BRANCH_FACTORIES[k](pretrained) for k in self.active})
        in_dim = sum(BRANCH_DIMS[k] for k in self.active)
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(head_hidden, num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = torch.cat([self.branches[k](inputs[k]) for k in self.active], dim=1)
        return {"logits": self.head(embedding), "embedding": embedding}

    # -- finetuning helpers --------------------------------------------------
    def _backbone_params(self):
        for k in self.active:
            branch = self.branches[k]
            if isinstance(branch, ImageBranch):
                yield from branch.backbone.parameters()

    def freeze_backbones(self) -> None:
        for p in self._backbone_params():
            p.requires_grad = False

    def unfreeze_backbones(self) -> None:
        for p in self._backbone_params():
            p.requires_grad = True

    def param_groups(self, backbone_lr: float, rest_lr: float) -> list[dict]:
        backbone = [p for p in self._backbone_params() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters()
                if p.requires_grad and id(p) not in backbone_ids]
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr})
        if rest:
            groups.append({"params": rest, "lr": rest_lr})
        return groups


def build_model(config: dict) -> MultiBranchNet:
    m = config["model"]
    return MultiBranchNet(branches=config["branches"], num_classes=m["num_classes"],
                          pretrained=m["pretrained"], head_hidden=m["head_hidden"],
                          dropout=m["dropout"])
```

- [ ] **Step 3: Run tests, delete old model test**

Run: `pytest tests/test_multibranch.py -v` — Expected: PASS (15 combos).
Then: `git rm tests/test_model.py`

- [ ] **Step 4: Commit**

```bash
git add src/instrument_classifier/models/multibranch.py tests/test_multibranch.py
git commit -m "Add MultiBranchNet with toggleable branches and finetuning helpers"
```

---

### Task 7: Per-class F1 in metrics

**Files:**
- Modify: `src/instrument_classifier/metrics.py`
- Modify: `tests/test_metrics.py` (append test)

**Interfaces:**
- Consumes: existing `multilabel_metrics`; `IRMAS_CLASSES` from labels.
- Produces: `per_class_f1(y_true, y_pred) -> dict[str, float]` keyed by IRMAS class code, same `(n_samples, 11)` binary array convention as `multilabel_metrics`.

- [ ] **Step 1: Append the failing test**

```python
# append to tests/test_metrics.py
import numpy as np

from instrument_classifier.data.labels import IRMAS_CLASSES
from instrument_classifier.metrics import per_class_f1


def test_per_class_f1_keys_and_values():
    n = len(IRMAS_CLASSES)
    y_true = np.zeros((4, n)); y_pred = np.zeros((4, n))
    y_true[:, 0] = 1.0; y_pred[:, 0] = 1.0          # class 0 perfect
    y_true[:2, 1] = 1.0                              # class 1 never predicted
    scores = per_class_f1(y_true, y_pred)
    assert list(scores) == IRMAS_CLASSES
    assert scores[IRMAS_CLASSES[0]] == 1.0
    assert scores[IRMAS_CLASSES[1]] == 0.0
```

Run: `pytest tests/test_metrics.py -v` — Expected: new test FAILS (ImportError), old ones PASS.

- [ ] **Step 2: Implement — append to `metrics.py`**

```python
from .data.labels import IRMAS_CLASSES  # add to imports at top


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """F1 per IRMAS class — the 'which instruments are hard' table."""
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {code: float(s) for code, s in zip(IRMAS_CLASSES, scores)}
```

(`f1_score` from sklearn is already imported in `metrics.py`; if the module computes F1 manually instead, add `from sklearn.metrics import f1_score` — check the file when editing.)

- [ ] **Step 3: Run** — `pytest tests/test_metrics.py -v` — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/instrument_classifier/metrics.py tests/test_metrics.py
git commit -m "Add per-class F1 metric"
```

---

### Task 8: Evaluation rewrite (windowed, on-the-fly features)

**Files:**
- Rewrite: `src/instrument_classifier/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `sliding_windows`/`aggregate_scores` (windowing.py), `extract_all`/`normalize`/`FeatureConfig` (Task 2), `IRMASTestDataset` (Task 4), `MultiBranchNet.forward` dict convention (Task 6), `multilabel_metrics`/`per_class_f1`/`tune_threshold` (Task 7), `utils.py`.
- Produces (used by train.py, ablation, CLI):
  - `windows_to_inputs(windows: Tensor[(W, L)], fc, stats, active: list[str]) -> dict[str, Tensor]` — per-window feature extraction + normalization + stacking, only active branches.
  - `clip_scores(model, waveform, device, fc, stats, active, window_len, hop_len, aggregate, batch_size=16) -> Tensor[(11,)]`
  - `gather_test_scores(model, dataset, device, fc, stats, active, window_len, hop_len, aggregate, show_progress=True) -> tuple[np.ndarray, np.ndarray]`
  - `evaluate_scores(y_true, y_scores, threshold) -> dict` — micro/macro metrics + `"per_class"` dict + `"threshold"`.
  - `evaluate_from_config(config, checkpoint_path) -> dict` and CLI `main()` (checkpoint stores `branches` and `stats`; see Task 9).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
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
```

NOTE: the exact micro/macro key names must match what `multilabel_metrics` already returns (check `metrics.py` — the existing train.py logs `val_microF1`, so keys are likely `micro_f1`-style; adjust the assertion to the real names when implementing, but do NOT rename existing metric keys).

Run: `pytest tests/test_evaluate.py -v` — Expected: FAIL.

- [ ] **Step 2: Rewrite `evaluate.py`**

```python
"""IRMAS multi-label evaluation: sliding windows over each polyphonic test
clip, per-window feature extraction (same code as preprocessing), sigmoid
scores aggregated per clip, micro/macro + per-class F1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data.dataset import IRMASTestDataset
from .features import FeatureConfig, extract_all, normalize
from .metrics import multilabel_metrics, per_class_f1
from .models.multibranch import MultiBranchNet
from .utils import load_config, resolve_device, save_metrics
from .windowing import aggregate_scores, sliding_windows

_IMAGE_KEYS = ("mel", "cqt", "chroma")


def windows_to_inputs(
    windows: torch.Tensor, fc: FeatureConfig, stats: dict, active: list[str],
) -> dict[str, torch.Tensor]:
    """(W, window_len) raw windows -> model input dict, only active branches."""
    per_key: dict[str, list[np.ndarray]] = {k: [] for k in active}
    for w in windows.numpy():
        feats = normalize(
            {k: v for k, v in extract_all(w, fc).items() if k in active},
            stats)
        for k in active:
            per_key[k].append(feats[k])
    out = {}
    for k, arrs in per_key.items():
        t = torch.from_numpy(np.stack(arrs))
        out[k] = t.unsqueeze(1) if k in _IMAGE_KEYS else t
    return out


@torch.no_grad()
def clip_scores(model, waveform, device, fc, stats, active,
                window_len, hop_len, aggregate="mean", batch_size=16) -> torch.Tensor:
    windows = sliding_windows(waveform, window_len, hop_len)
    probs = []
    for start in range(0, windows.shape[0], batch_size):
        inputs = windows_to_inputs(windows[start:start + batch_size], fc, stats, active)
        logits = model({k: v.to(device) for k, v in inputs.items()})["logits"]
        probs.append(torch.sigmoid(logits).cpu())
    return aggregate_scores(torch.cat(probs, dim=0), method=aggregate)


@torch.no_grad()
def gather_test_scores(model, dataset, device, fc, stats, active,
                       window_len, hop_len, aggregate="mean",
                       show_progress=True) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_scores = [], []
    for wav, target, _name in tqdm(dataset, desc="eval", disable=not show_progress):
        y_scores.append(clip_scores(model, wav, device, fc, stats, active,
                                    window_len, hop_len, aggregate).numpy())
        y_true.append(target.numpy())
    return np.stack(y_true), np.stack(y_scores)


def evaluate_scores(y_true, y_scores, threshold: float) -> dict:
    y_pred = (y_scores >= threshold).astype(np.float32)
    metrics = multilabel_metrics(y_true, y_pred)
    metrics["per_class"] = per_class_f1(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    return metrics


def evaluate_from_config(config: dict, checkpoint_path: str | Path) -> dict:
    device = resolve_device(config["device"])
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = MultiBranchNet(branches=ckpt["branches"],
                           num_classes=config["model"]["num_classes"],
                           pretrained=False,
                           head_hidden=config["model"]["head_hidden"],
                           dropout=config["model"]["dropout"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    fc = FeatureConfig.from_config(config)
    ev = config["eval"]
    y_true, y_scores = gather_test_scores(
        model, IRMASTestDataset(config["data"]["test_dir"], fc.sample_rate),
        device, fc, ckpt["stats"], model.active,
        int(round(ev["window_seconds"] * fc.sample_rate)),
        int(round(ev["hop_seconds"] * fc.sample_rate)), ev["aggregate"])
    return evaluate_scores(y_true, y_scores,
                           ckpt.get("threshold", ev["default_threshold"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the IRMAS test set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = evaluate_from_config(config, args.checkpoint)
    print(metrics)
    save_metrics(args.out or str(Path(config["output_dir"]) / "test_metrics.json"), metrics)


if __name__ == "__main__":
    main()
```

Adjust the metric key names in the test to match `multilabel_metrics`'s real output before finalizing (read `metrics.py` first — the keys must not be renamed).

- [ ] **Step 3: Run** — `pytest tests/test_evaluate.py -v` — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/instrument_classifier/evaluate.py tests/test_evaluate.py
git commit -m "Rewrite evaluation: windowed multi-input scoring with shared features"
```

---

### Task 9: Training rewrite

**Files:**
- Rewrite: `src/instrument_classifier/train.py`
- Rewrite: `tests/test_training_integration.py`

**Interfaces:**
- Consumes: everything above. DataLoader batching relies on PyTorch's `default_collate`, which batches dicts of tensors natively.
- Produces:
  - `run_training(config: dict) -> dict` — same contract as today: trains, tunes threshold, evaluates on test dir if present, saves `outputs/metrics.json` + `outputs/best.pth`.
  - Checkpoint format (`best.pth`, via `save_checkpoint(path, model, extra=...)`): extra keys `threshold: float`, `val_micro_f1: float`, `branches: dict[str, bool]`, `stats: dict` — exactly what `evaluate_from_config` (Task 8) reads.
  - CLI `main()` for `scripts/train.py` / `irmas-train`.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_training_integration.py
"""End-to-end smoke test on the synthetic dataset: 1 warmup + 2 finetune
epochs on CPU with pretrained=False must run, save a checkpoint with the new
format, and improve training loss."""
import torch

from instrument_classifier.train import run_training
from conftest import TEST_FC


def _config(features_dir, irmas_root, out_dir):
    return {
        "seed": 42, "device": "cpu", "output_dir": str(out_dir),
        "data": {"train_dir": str(irmas_root / "IRMAS-TrainingData"),
                 "test_dir": str(irmas_root / "IRMAS-TestingData"),
                 "features_dir": str(features_dir.parent),  # dataset uses <features_dir>/train
                 "val_fraction": 0.25, "num_workers": 0},
        "features": {"sample_rate": 22050, "clip_seconds": 3.0, "n_fft": 2048,
                     "hop_length": 512, "n_mels": 128, "cqt_bins": 84},
        "branches": {"mel": True, "cqt": False, "wave": True, "chroma": False},
        "model": {"num_classes": 11, "pretrained": False,
                  "head_hidden": 64, "dropout": 0.1},
        "augment": {"specaugment": {"enabled": False, "time_masks": 0, "time_width": 0,
                                    "freq_masks": 0, "freq_width": 0},
                    "mixup_alpha": 0.0},
        "train": {"batch_size": 4, "warmup_epochs": 1, "warmup_lr": 1e-3,
                  "finetune_epochs": 2, "head_lr": 1e-3, "backbone_lr": 1e-4,
                  "weight_decay": 0.0, "early_stopping_patience": 10},
        "eval": {"window_seconds": 3.0, "hop_seconds": 1.0, "aggregate": "mean",
                 "default_threshold": 0.5},
    }


def test_training_runs_and_checkpoints(preprocessed_features, synthetic_irmas, tmp_path):
    # conftest preprocesses into <root>/features; training expects <features_dir>/train,
    # so link it to match the layout produced by scripts/preprocess.py.
    feat_root = tmp_path / "features"
    feat_root.mkdir()
    (feat_root / "train").symlink_to(preprocessed_features)

    config = _config(feat_root / "train", synthetic_irmas, tmp_path / "out")
    config["data"]["features_dir"] = str(feat_root)
    results = run_training(config)

    ckpt = torch.load(tmp_path / "out" / "best.pth", map_location="cpu",
                      weights_only=False)
    assert {"model", "threshold", "val_micro_f1", "branches", "stats"} <= set(ckpt)
    assert ckpt["branches"] == {"mel": True, "cqt": False, "wave": True, "chroma": False}
    assert "test" in results and "per_class" in results["test"]
    assert (tmp_path / "out" / "metrics.json").exists()
```

Run: `pytest tests/test_training_integration.py -v` — Expected: FAIL.

- [ ] **Step 2: Rewrite `train.py`**

```python
"""Two-phase training of MultiBranchNet on precomputed IRMAS features:
(1) warmup with frozen ResNet backbones, (2) full finetuning with
discriminative LRs + cosine decay + early stopping. Threshold tuned on the
validation split; final evaluation on the official IRMAS test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .data.dataset import IRMASFeaturesDataset, IRMASTestDataset
from .data.transforms import SpecAugment, mixup_batch
from .evaluate import evaluate_scores, gather_test_scores
from .features import FeatureConfig
from .metrics import tune_threshold
from .models.multibranch import build_model
from .utils import load_config, resolve_device, save_checkpoint, save_metrics, set_seed


def train_one_epoch(model, loader, optimizer, device, criterion,
                    mixup_alpha=0.0, generator=None) -> float:
    model.train()
    total, n = 0.0, 0
    for feats, target in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        target = target.to(device)
        if mixup_alpha > 0:
            feats, target = mixup_batch(feats, target, mixup_alpha, generator=generator)
        optimizer.zero_grad()
        loss = criterion(model(feats)["logits"], target)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def score_fixed_clips(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Sigmoid scores for fixed-length validation clips -> (y_true, y_scores)."""
    model.eval()
    y_true, y_scores = [], []
    for feats, target in loader:
        logits = model({k: v.to(device) for k, v in feats.items()})["logits"]
        y_scores.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(target.numpy())
    return np.concatenate(y_true), np.concatenate(y_scores)


def _stratified_split(dataset: IRMASFeaturesDataset, val_fraction: float, seed: int):
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_fraction, random_state=seed, stratify=dataset.targets())
    return train_idx.tolist(), val_idx.tolist()


def run_training(config: dict) -> dict:
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    data_cfg, train_cfg, aug_cfg = config["data"], config["train"], config["augment"]
    active = [k for k, on in config["branches"].items() if on]
    features_dir = Path(data_cfg["features_dir"]) / "train"

    sa_cfg = aug_cfg["specaugment"]
    transform = (SpecAugment(sa_cfg["time_masks"], sa_cfg["time_width"],
                             sa_cfg["freq_masks"], sa_cfg["freq_width"],
                             seed=config["seed"])
                 if sa_cfg["enabled"] else None)
    full_aug = IRMASFeaturesDataset(features_dir, active, transform=transform)
    full_plain = IRMASFeaturesDataset(features_dir, active, transform=None)
    train_idx, val_idx = _stratified_split(full_plain, data_cfg["val_fraction"], config["seed"])
    train_loader = DataLoader(Subset(full_aug, train_idx),
                              batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(Subset(full_plain, val_idx),
                            batch_size=train_cfg["batch_size"],
                            num_workers=data_cfg["num_workers"])

    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(config["seed"])

    out_dir = Path(config["output_dir"])
    ckpt_path = out_dir / "best.pth"
    best_f1, best_threshold, patience = -1.0, config["eval"]["default_threshold"], 0
    candidates = np.linspace(0.05, 0.95, 19)

    def validate() -> tuple[float, float]:
        y_true, y_scores = score_fixed_clips(model, val_loader, device)
        return tune_threshold(y_true, y_scores, candidates)

    def checkpoint_if_best(f1: float, t: float) -> None:
        nonlocal best_f1, best_threshold, patience
        if f1 > best_f1:
            best_f1, best_threshold, patience = f1, t, 0
            save_checkpoint(ckpt_path, model, extra={
                "threshold": best_threshold, "val_micro_f1": best_f1,
                "branches": config["branches"], "stats": full_plain.stats})
        else:
            patience += 1

    # Phase 1: frozen ResNet backbones — train head + scratch branches.
    model.freeze_backbones()
    opt = torch.optim.Adam(model.param_groups(train_cfg["backbone_lr"],
                                              train_cfg["warmup_lr"]),
                           weight_decay=train_cfg["weight_decay"])
    for epoch in range(train_cfg["warmup_epochs"]):
        loss = train_one_epoch(model, train_loader, opt, device, criterion,
                               aug_cfg["mixup_alpha"], gen)
        t, f1 = validate()
        print(f"[warmup {epoch+1}/{train_cfg['warmup_epochs']}] "
              f"loss={loss:.4f} val_microF1={f1:.4f}")

    # Phase 2: everything trainable, discriminative LRs + cosine decay.
    model.unfreeze_backbones()
    opt = torch.optim.Adam(model.param_groups(train_cfg["backbone_lr"],
                                              train_cfg["head_lr"]),
                           weight_decay=train_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=train_cfg["finetune_epochs"])
    for epoch in range(train_cfg["finetune_epochs"]):
        loss = train_one_epoch(model, train_loader, opt, device, criterion,
                               aug_cfg["mixup_alpha"], gen)
        sched.step()
        t, f1 = validate()
        print(f"[finetune {epoch+1}/{train_cfg['finetune_epochs']}] "
              f"loss={loss:.4f} val_microF1={f1:.4f}")
        checkpoint_if_best(f1, t)
        if patience >= train_cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_f1 < 0:  # warmup-only runs (e.g. finetune_epochs=0) still checkpoint
        t, f1 = validate()
        checkpoint_if_best(f1, t)

    results = {"val_micro_f1": best_f1, "threshold": best_threshold,
               "branches": config["branches"]}

    test_dir = Path(data_cfg["test_dir"])
    if test_dir.exists() and any(test_dir.rglob("*.wav")):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        fc = FeatureConfig.from_config(config)
        ev = config["eval"]
        y_true, y_scores = gather_test_scores(
            model, IRMASTestDataset(test_dir, fc.sample_rate), device, fc,
            full_plain.stats, model.active,
            int(round(ev["window_seconds"] * fc.sample_rate)),
            int(round(ev["hop_seconds"] * fc.sample_rate)), ev["aggregate"])
        results["test"] = evaluate_scores(y_true, y_scores, best_threshold)
        print("TEST:", results["test"])

    save_metrics(out_dir / "metrics.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MultiBranchNet on IRMAS")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
```

NOTE: `save_checkpoint` in `utils.py` must store the extras alongside `"model"` — check its signature when implementing; today `train.py` already calls it with `extra={...}`, so the format matches.

- [ ] **Step 3: Run the integration test** — `pytest tests/test_training_integration.py -v` — Expected: PASS (takes ~1-2 min on CPU: ResNet18 on 8 clips × 3 epochs).

- [ ] **Step 4: Run the whole suite** — `pytest -x -q` — Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/instrument_classifier/train.py tests/test_training_integration.py
git commit -m "Rewrite training for multi-input features with branch-aware checkpoints"
```

---

### Task 10: Ablation runner

**Files:**
- Create: `scripts/run_ablation.py`
- Test: `tests/test_ablation.py`
- Create: `tests/scripts_ablation_shim.py` (same pattern as Task 3's shim)

**Interfaces:**
- Consumes: `run_training(config)` (Task 9).
- Produces:
  - `ABLATION_COMBOS: list[dict[str, bool]]` — exactly, in order: mel only; mel+cqt; mel+cqt+wave; mel+cqt+wave+chroma.
  - `format_ablation_table(rows: list[dict]) -> str` — markdown table; each row is `{"name": str, "val_micro_f1": float, "test": {…}}` (test optional).
  - CLI: `python scripts/run_ablation.py --config configs/default.yaml [--epochs N]` — for each combo, deep-copies the config, sets `branches`, redirects `output_dir` to `outputs/ablation/<name>`, optionally overrides `finetune_epochs`, calls `run_training`, then writes `outputs/ablation.md`. Runs differ ONLY in `branches` (same seed/split/procedure) — this is the spec's hard requirement.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ablation.py
from scripts_ablation_shim import ABLATION_COMBOS, combo_name, format_ablation_table


def test_combos_grow_monotonically():
    actives = [sorted(k for k, on in c.items() if on) for c in ABLATION_COMBOS]
    assert actives[0] == ["mel"]
    assert actives[-1] == ["chroma", "cqt", "mel", "wave"]
    for smaller, larger in zip(actives, actives[1:]):
        assert set(smaller) < set(larger)


def test_format_ablation_table():
    rows = [
        {"name": "mel", "val_micro_f1": 0.61,
         "test": {"micro_f1": 0.55, "macro_f1": 0.48}},
        {"name": "mel+cqt", "val_micro_f1": 0.64, "test": None},
    ]
    table = format_ablation_table(rows)
    assert "| mel " in table and "| mel+cqt " in table
    assert "0.5500" in table and "n/a" in table
```

```python
# tests/scripts_ablation_shim.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ablation import ABLATION_COMBOS, combo_name, format_ablation_table  # noqa: E402,F401
```

Run: `pytest tests/test_ablation.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement `scripts/run_ablation.py`**

```python
#!/usr/bin/env python
"""Per-branch ablation study: train N configurations that differ ONLY in the
active branches (same seed, split, epochs, threshold procedure) and write a
comparison table to outputs/ablation.md.

    python scripts/run_ablation.py --config configs/default.yaml
    python scripts/run_ablation.py --config configs/default.yaml --epochs 2  # smoke run
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from instrument_classifier.train import run_training
from instrument_classifier.utils import load_config

ABLATION_COMBOS = [
    {"mel": True, "cqt": False, "wave": False, "chroma": False},
    {"mel": True, "cqt": True, "wave": False, "chroma": False},
    {"mel": True, "cqt": True, "wave": True, "chroma": False},
    {"mel": True, "cqt": True, "wave": True, "chroma": True},
]


def combo_name(combo: dict[str, bool]) -> str:
    order = ("mel", "cqt", "wave", "chroma")  # report order, not alphabetical
    return "+".join(k for k in order if combo[k])


def format_ablation_table(rows: list[dict]) -> str:
    lines = ["| Branches | val micro-F1 | test micro-F1 | test macro-F1 |",
             "|---|---|---|---|"]
    for r in rows:
        test = r.get("test") or {}
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "n/a"
        lines.append(f"| {r['name']} | {fmt(r['val_micro_f1'])} "
                     f"| {fmt(test.get('micro_f1'))} | {fmt(test.get('macro_f1'))} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override finetune_epochs (smoke runs)")
    args = parser.parse_args()
    base = load_config(args.config)

    rows = []
    for combo in ABLATION_COMBOS:
        name = combo_name(combo)
        cfg = copy.deepcopy(base)          # identical except for what follows
        cfg["branches"] = dict(combo)
        cfg["output_dir"] = str(Path(base["output_dir"]) / "ablation" / name)
        if args.epochs is not None:
            cfg["train"]["finetune_epochs"] = args.epochs
        print(f"=== ablation: {name} ===")
        results = run_training(cfg)
        rows.append({"name": name, "val_micro_f1": results["val_micro_f1"],
                     "test": results.get("test")})

    out = Path(base["output_dir"]) / "ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Ablation study — branch contributions\n\n"
                   + format_ablation_table(rows))
    print(f"Wrote {out}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
```

NOTE: match the exact micro/macro key names used by `multilabel_metrics` (as in Task 8) in `format_ablation_table`.

- [ ] **Step 3: Run** — `pytest tests/test_ablation.py -v` — Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_ablation.py tests/test_ablation.py tests/scripts_ablation_shim.py
git commit -m "Add branch-ablation runner with markdown report"
```

---

### Task 11: Cleanup, Colab notebook, README

**Files:**
- Delete: `src/instrument_classifier/models/cnn14.py`, `scripts/download_pretrained.py`
- Rewrite: `notebooks/colab_train.ipynb`
- Rewrite: `README.md`
- Modify: `.gitignore` (ensure `data/`, `outputs/`, `checkpoints/` are ignored — check, likely already there)

**Interfaces:** none new — this is the final consistency pass.

- [ ] **Step 1: Delete dead code and confirm nothing references it**

```bash
git rm src/instrument_classifier/models/cnn14.py scripts/download_pretrained.py
grep -rn "cnn14\|torchlibrosa\|download_pretrained\|Cnn14" src scripts tests configs README.md notebooks --include="*.py" --include="*.yaml" --include="*.md" || echo CLEAN
```

Expected: `CLEAN` (fix any stragglers before proceeding).

- [ ] **Step 2: Run the full suite** — `pytest -q` — Expected: all green.

- [ ] **Step 3: Rewrite the Colab notebook**

Recreate `notebooks/colab_train.ipynb` with exactly these cells (markdown headers + code):

1. **[md]** Title: "MultiBranchNet on IRMAS — Colab training". Note: GPU runtime required (T4), features live on Drive so preprocessing runs once.
2. **[code]** Mount Drive + clone/pull the repo:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   %cd /content
   !git clone https://github.com/USER/instrument-classifier-v2.git || (cd instrument-classifier-v2 && git pull)
   %cd instrument-classifier-v2
   ```
3. **[code]** `!pip install -q -e .`
4. **[code]** Download IRMAS into Drive if missing, symlink into `data/`:
   ```python
   import os
   DRIVE = "/content/drive/MyDrive/irmas"
   os.makedirs(DRIVE, exist_ok=True)
   !ln -sfn {DRIVE} data
   !python scripts/download_data.py --data-dir data
   ```
5. **[code]** Preprocess once (skips if `stats.json` exists):
   ```python
   import pathlib
   if not pathlib.Path("data/features/train/stats.json").exists():
       !python scripts/preprocess.py --config configs/default.yaml
   ```
6. **[code]** `!python scripts/train.py --config configs/default.yaml`
7. **[code]** `!python scripts/run_ablation.py --config configs/default.yaml`
8. **[code]** Show results: `print(open("outputs/ablation.md").read())`

(Write the .ipynb JSON directly; nbformat 4, one kernel spec `python3`.)

- [ ] **Step 4: Rewrite README.md**

Cover, mirroring the current README's structure and tone: project description (multi-input CNN, 4 branches, ImageNet transfer, IRMAS protocol); the "why this design" table now contrasting v1 (Keras multi-feature small CNN) / v2-CNN14 (single-input finetuning) / v2-MultiBranch (this design: original architecture + per-branch ablation); setup (`pip install -e .`); data download (`scripts/download_data.py` — note: no pretrained checkpoint download needed, torchvision fetches ResNet18 automatically); **pipeline** section with the three commands in order (`preprocess.py` → `train.py` → `run_ablation.py` / `evaluate.py`); project layout tree; tests (`pytest`); results section with the ablation table placeholder (val/test micro/macro F1 per combo, "_TBD after full training run_"); credits (IRMAS — Bosch et al.; ResNet — He et al. 2015; SpecAugment — Park et al. 2019; mixup — Zhang et al. 2017).

- [ ] **Step 5: Final check and commit**

```bash
pytest -q && git add -A && git commit -m "Remove CNN14 pipeline; update notebook and README for MultiBranchNet"
```

---

## Verification (after all tasks)

1. `pytest -q` — full suite green, no network access needed.
2. `python -c "from instrument_classifier.models.multibranch import MultiBranchNet"` — imports clean.
3. Smoke ablation locally (optional, slow): point `configs/default.yaml` at a 20-clip subset, run `python scripts/run_ablation.py --epochs 1`.
4. Real run happens on Colab via the notebook (out of scope for local execution).
