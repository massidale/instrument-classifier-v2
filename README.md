# Instrument Classifier v2 — IRMAS via Transfer Learning

Musical instrument recognition on the **IRMAS** benchmark, done the modern way:
finetune a **PANNs CNN14** backbone (pretrained on AudioSet) instead of training a
tiny CNN from scratch. Multi-label evaluation follows the official IRMAS protocol.

This is a clean PyTorch rewrite of an older Keras notebook project.

## Why this design

| Concern | Old project | This project |
|---|---|---|
| Model | 2-layer CNN trained from scratch | CNN14 pretrained on AudioSet, finetuned |
| Features | hand-crafted CQT + mel + chroma + waveform | log-mel computed in-graph (as the backbone expects) |
| Data loading | everything into RAM as numpy lists | lazy per-item `Dataset` / `DataLoader` |
| Evaluation | K-Fold on training clips only | official IRMAS multi-label test set, micro/macro-F1 |
| Reproducibility | none | single YAML config, global seed, saved checkpoints + metrics |
| Framework | TensorFlow/Keras | PyTorch, installable package, tested |

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

## Get the data & pretrained weights

```bash
# CNN14 AudioSet checkpoint (~300 MB)
python scripts/download_pretrained.py

# IRMAS training + testing data (~3 GB). Reuse existing training data if you have it:
python scripts/download_data.py --link-train /path/to/IRMAS-TrainingData
# ...or download everything:
python scripts/download_data.py
```

## Train & evaluate

```bash
python scripts/train.py --config configs/default.yaml      # finetunes, tunes threshold, evaluates on test
python scripts/evaluate.py --checkpoint outputs/best.pth   # re-evaluate a saved checkpoint
```

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml).
Training runs in two phases: (1) freeze the backbone and warm up the new head,
then (2) unfreeze everything with a discriminative learning rate and cosine decay.

## Project layout

```
src/instrument_classifier/
  data/{labels,transforms,dataset}.py   # vocab, augmentation, lazy IRMAS datasets
  models/cnn14.py                        # PANNs CNN14 + new head + freeze/LR helpers
  windowing.py                           # sliding-window split + score aggregation
  metrics.py                             # micro/macro F1, threshold tuning
  train.py / evaluate.py                 # orchestration + CLI
scripts/                                 # data/weights download + CLI wrappers
notebooks/colab_train.ipynb              # run training on a free Colab GPU
tests/                                   # pytest suite (run: pytest)
```

## Tests

```bash
pytest
```

Covers label encoding, dataset shapes, augmentation, sliding-window aggregation,
metric correctness, model wiring, and an end-to-end overfit sanity check.

## Results

Filled in after a full training run (`outputs/metrics.json`):

| Metric | Value |
|---|---|
| Test micro-F1 | _TBD_ |
| Test macro-F1 | _TBD_ |

## Credits

- IRMAS dataset — Bosch et al., MTG, Universitat Pompeu Fabra.
- PANNs / CNN14 — Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks", 2020.
