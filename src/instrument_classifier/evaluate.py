"""IRMAS multi-label evaluation: sliding-window inference + micro/macro F1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data.dataset import IRMASTestDataset
from .metrics import multilabel_metrics, tune_threshold
from .models.cnn14 import build_cnn14_finetune
from .utils import load_config, resolve_device, save_metrics
from .windowing import aggregate_scores, sliding_windows


@torch.no_grad()
def clip_scores(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    device: torch.device,
    window_len: int,
    hop_len: int,
    aggregate: str = "mean",
    batch_size: int = 32,
) -> torch.Tensor:
    """Score one variable-length clip: window -> sigmoid -> pool to (n_classes,)."""
    windows = sliding_windows(waveform, window_len, hop_len)  # (W, window_len)
    probs = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device)
        logits = model(batch)["logits"]
        probs.append(torch.sigmoid(logits).cpu())
    probs = torch.cat(probs, dim=0)  # (W, n_classes)
    return aggregate_scores(probs, method=aggregate)


@torch.no_grad()
def gather_test_scores(
    model: torch.nn.Module,
    dataset: IRMASTestDataset,
    device: torch.device,
    window_len: int,
    hop_len: int,
    aggregate: str = "mean",
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_scores)`` arrays of shape (n_clips, n_classes)."""
    model.eval()
    y_true, y_scores = [], []
    iterator = tqdm(dataset, desc="eval", disable=not show_progress)
    for wav, target, _name in iterator:
        scores = clip_scores(model, wav, device, window_len, hop_len, aggregate)
        y_scores.append(scores.numpy())
        y_true.append(target.numpy())
    return np.stack(y_true), np.stack(y_scores)


def evaluate_scores(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Binarize scores at a threshold and compute multi-label metrics."""
    y_pred = (y_scores >= threshold).astype(np.float32)
    metrics = multilabel_metrics(y_true, y_pred)
    metrics["threshold"] = float(threshold)
    return metrics


def evaluate_from_config(config: dict, checkpoint_path: str | Path) -> dict:
    """Load a finetuned checkpoint and evaluate on the IRMAS test set."""
    device = resolve_device(config["device"])
    model = build_cnn14_finetune(config["model"]["num_classes"], pretrained_path=None)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    sr = config["data"]["sample_rate"]
    ev = config["eval"]
    window_len = int(round(ev["window_seconds"] * sr))
    hop_len = int(round(ev["hop_seconds"] * sr))

    test_ds = IRMASTestDataset(config["data"]["test_dir"], sample_rate=sr)
    y_true, y_scores = gather_test_scores(
        model, test_ds, device, window_len, hop_len, ev["aggregate"]
    )
    threshold = ckpt.get("threshold", ev["default_threshold"])
    return evaluate_scores(y_true, y_scores, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the IRMAS test set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None, help="Where to write metrics JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = evaluate_from_config(config, args.checkpoint)
    print(metrics)
    out = args.out or str(Path(config["output_dir"]) / "test_metrics.json")
    save_metrics(out, metrics)


if __name__ == "__main__":
    main()
