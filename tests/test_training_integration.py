"""Behavioral checks: the training loop actually learns, and eval runs end-to-end.

These are slow (they build and train the real CNN14) but small: short clips,
few samples, a handful of steps. They guard the wiring, not accuracy.
"""

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from instrument_classifier.data.dataset import IRMASTrainDataset, IRMASTestDataset
from instrument_classifier.evaluate import gather_test_scores, evaluate_scores
from instrument_classifier.models.cnn14 import build_cnn14_finetune
from instrument_classifier.train import train_one_epoch

SR = 16000


def _tone(path, freq, seconds=1.0, sr=SR):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)


def test_training_loss_decreases_on_separable_data(tmp_path):
    # Two instruments <-> two clearly separable pure tones. Clips are 2s so the
    # model's internal SpecAugment (time_drop_width=64 frames) has enough frames.
    for code, freq in (("pia", 220.0), ("vio", 3000.0)):
        d = tmp_path / code
        d.mkdir()
        for i in range(3):
            _tone(d / f"{i}.wav", freq, seconds=2.0)

    ds = IRMASTrainDataset(tmp_path, sample_rate=SR, clip_seconds=2.0)
    loader = DataLoader(ds, batch_size=6, shuffle=True, num_workers=0)

    torch.manual_seed(0)
    model = build_cnn14_finetune(num_classes=11, pretrained_path=None)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    first = train_one_epoch(model, loader, opt, torch.device("cpu"), criterion)
    for _ in range(7):
        last = train_one_epoch(model, loader, opt, torch.device("cpu"), criterion)

    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_eval_pipeline_runs_end_to_end(tmp_path):
    _tone(tmp_path / "clip.wav", 440.0, seconds=4.0)
    (tmp_path / "clip.txt").write_text("pia\nvoi\n")

    ds = IRMASTestDataset(tmp_path, sample_rate=SR)
    model = build_cnn14_finetune(num_classes=11, pretrained_path=None)

    y_true, y_scores = gather_test_scores(
        model, ds, torch.device("cpu"),
        window_len=SR, hop_len=SR // 2, aggregate="mean", show_progress=False,
    )
    assert y_true.shape == (1, 11)
    assert y_scores.shape == (1, 11)

    metrics = evaluate_scores(y_true, y_scores, threshold=0.5)
    assert "micro_f1" in metrics and "macro_f1" in metrics
    assert 0.0 <= metrics["micro_f1"] <= 1.0
