# Instrument Classifier v2 — IRMAS via a Multi-Input CNN

Musical instrument recognition on the **IRMAS** benchmark with a purpose-built
**multi-input CNN**. Each audio representation gets its own branch — log-mel and
CQT spectrograms ride **ImageNet-pretrained ResNet18** backbones, while raw
waveform and chroma branches are trained from scratch — and their embeddings are
fused (late fusion) into a shared multi-label head. Because the branches are
toggleable, the same code doubles as a **per-branch ablation study**.
Multi-label evaluation follows the official IRMAS protocol.

This is a clean PyTorch rewrite of an older Keras notebook project.

## Why this design

| Concern | v1 (Keras) | v2-CNN14 | v2-MultiBranch (this repo) |
|---|---|---|---|
| Model | small multi-feature CNN from scratch | single-input CNN14 finetuned | 4-branch multi-input CNN, late fusion |
| Transfer learning | none | PANNs weights (AudioSet) | ImageNet ResNet18 on mel & CQT branches |
| Features | CQT + mel + chroma + waveform (into RAM) | log-mel computed in-graph | mel + CQT + waveform + chroma, precomputed to `.npz` |
| Ablation | none | none | per-branch toggles → contribution table |
| Data loading | everything into RAM as numpy lists | lazy per-item `Dataset` | lazy `Dataset` over cached features |
| Evaluation | K-Fold on training clips only | official IRMAS test set, micro/macro-F1 | official IRMAS test set, windowed micro/macro-F1 |
| Augmentation | none | waveform noise/gain | SpecAugment + multi-input mixup |
| Reproducibility | none | YAML config + seed | single YAML config, global seed, saved checkpoints + metrics |
| Framework | TensorFlow/Keras | PyTorch | PyTorch, installable package, tested |

## Task

- **Train** on 6,705 single-instrument 3s clips (11 classes: `cel cla flu gac gel org pia sax tru vio voi`).
- **Test** on the polyphonic, multi-label IRMAS test set using a sliding 3s window aggregated per clip.
- **Report** micro/macro precision, recall and F1, with the decision threshold tuned on a held-out validation split.

## Setup

```bash
conda create -n instrument-classifier-v2 python=3.11 -y
conda activate instrument-classifier-v2
pip install -e .
```

No pretrained-checkpoint download is needed: torchvision fetches the ResNet18
ImageNet weights automatically on first use.

## Get the data

```bash
# IRMAS training + testing data (~3 GB). Reuse existing training data if you have it:
python scripts/download_data.py --link-train /path/to/IRMAS-TrainingData
# ...or download everything:
python scripts/download_data.py
```

## Pipeline

Three steps, in order. All hyperparameters live in
[`configs/default.yaml`](configs/default.yaml).

```bash
# 1. Precompute mel/CQT/chroma/waveform features once (writes .npz + norm stats)
python scripts/preprocess.py --config configs/default.yaml

# 2. Train: warm up the new head with backbones frozen, then finetune with a
#    discriminative learning rate; tunes the threshold and evaluates on test
python scripts/train.py --config configs/default.yaml

# 3. Ablation: retrain across branch combinations and write a comparison table
python scripts/run_ablation.py --config configs/default.yaml
```

To re-score a saved checkpoint without retraining:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best.pth
```

Or run the whole thing on a free Colab GPU with
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb).

## Project layout

```
src/instrument_classifier/
  data/{labels,transforms,dataset}.py   # vocab, SpecAugment/mixup, lazy feature datasets
  models/multibranch.py                  # MultiBranchNet: 4 toggleable branches + fusion head
  features.py                            # shared mel/CQT/chroma/waveform extraction
  windowing.py                           # sliding-window split + score aggregation
  metrics.py                             # micro/macro/per-class F1, threshold tuning
  train.py / evaluate.py                 # orchestration + CLI
scripts/                                 # data download, preprocess, train, evaluate, ablation
configs/default.yaml                     # every hyperparameter
notebooks/colab_train.ipynb              # run the pipeline on a free Colab GPU
tests/                                   # pytest suite (run: pytest)
```

## Tests

```bash
pytest
```

Covers label encoding, feature extraction, dataset shapes, SpecAugment/mixup,
sliding-window aggregation, metric correctness, model wiring (branch toggles),
and an end-to-end training integration test.

## Results

Filled in after a full training run (`outputs/ablation.md`):

| Branches | val micro-F1 | test micro-F1 | test macro-F1 |
|---|---|---|---|
| mel | _TBD_ | _TBD_ | _TBD_ |
| mel+cqt | _TBD_ | _TBD_ | _TBD_ |
| mel+cqt+wave | _TBD_ | _TBD_ | _TBD_ |
| mel+cqt+wave+chroma | 0.8454 | 0.5328 | 0.4394 |

Full-model run: 3 warmup + 25 finetune epochs, threshold 0.40 tuned on validation.
Test micro P/R = 0.89/0.38 — precision-heavy, reflecting the known val(mono)/test(polyphonic)
calibration gap discussed in the evaluation protocol. Per-class F1 ranges from 0.81 (voi)
and 0.80 (sax) down to 0.00 (cla). Remaining rows pending the ablation study.

## Credits

- IRMAS dataset — Bosch et al., MTG, Universitat Pompeu Fabra.
- ResNet — He et al., "Deep Residual Learning for Image Recognition", 2015.
- SpecAugment — Park et al., "SpecAugment", 2019.
- mixup — Zhang et al., "mixup: Beyond Empirical Risk Minimization", 2017.
