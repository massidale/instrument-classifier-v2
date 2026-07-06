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
