"""Two-phase finetuning of CNN14 on IRMAS, with threshold tuning + test eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data.dataset import IRMASTestDataset, IRMASTrainDataset
from .data.transforms import AugmentTransform, mixup_batch
from .evaluate import evaluate_scores, gather_test_scores
from .metrics import multilabel_metrics, tune_threshold
from .models.cnn14 import build_cnn14_finetune
from .utils import load_config, resolve_device, save_checkpoint, save_metrics, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    mixup_alpha: float = 0.0,
    generator: torch.Generator | None = None,
) -> float:
    """Run one training epoch; return the mean batch loss."""
    model.train()
    total, n = 0.0, 0
    for wav, target in loader:
        wav, target = wav.to(device), target.to(device)
        if mixup_alpha > 0:
            wav, target = mixup_batch(wav, target, mixup_alpha, generator=generator)
        optimizer.zero_grad()
        logits = model(wav)["logits"]
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def score_fixed_clips(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Sigmoid scores for fixed-length clips (one forward each). -> (y_true, y_scores)."""
    model.eval()
    y_true, y_scores = [], []
    for wav, target in loader:
        logits = model(wav.to(device))["logits"]
        y_scores.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(target.numpy())
    return np.concatenate(y_true), np.concatenate(y_scores)


def _stratified_split(dataset: IRMASTrainDataset, val_fraction: float, seed: int):
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_fraction, random_state=seed, stratify=dataset.targets()
    )
    return train_idx.tolist(), val_idx.tolist()


def run_training(config: dict) -> dict:
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    data_cfg, train_cfg, aug_cfg = config["data"], config["train"], config["augment"]

    aug = (
        AugmentTransform(aug_cfg["gain_db"], aug_cfg["noise_snr_db"], seed=config["seed"])
        if aug_cfg["enabled"]
        else None
    )
    full_aug = IRMASTrainDataset(
        data_cfg["train_dir"], data_cfg["sample_rate"], data_cfg["clip_seconds"], transform=aug
    )
    full_plain = IRMASTrainDataset(
        data_cfg["train_dir"], data_cfg["sample_rate"], data_cfg["clip_seconds"], transform=None
    )
    train_idx, val_idx = _stratified_split(full_plain, data_cfg["val_fraction"], config["seed"])
    train_ds, val_ds = Subset(full_aug, train_idx), Subset(full_plain, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], num_workers=data_cfg["num_workers"])

    model = build_cnn14_finetune(config["model"]["num_classes"], config["model"]["pretrained_path"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(config["seed"])

    out_dir = Path(config["output_dir"])
    ckpt_path = out_dir / "best.pth"
    best_f1, best_threshold, patience = -1.0, config["eval"]["default_threshold"], 0
    candidates = np.linspace(0.05, 0.95, 19)

    def validate() -> tuple[float, float]:
        y_true, y_scores = score_fixed_clips(model, val_loader, device)
        t, f1 = tune_threshold(y_true, y_scores, candidates)
        return t, f1

    # Phase 1: freeze backbone, train only the new head.
    model.freeze_backbone()
    opt = torch.optim.Adam(model.param_groups(train_cfg["backbone_lr"], train_cfg["warmup_lr"]),
                           weight_decay=train_cfg["weight_decay"])
    for epoch in range(train_cfg["warmup_epochs"]):
        loss = train_one_epoch(model, train_loader, opt, device, criterion, aug_cfg["mixup_alpha"], gen)
        t, f1 = validate()
        print(f"[warmup {epoch+1}/{train_cfg['warmup_epochs']}] loss={loss:.4f} val_microF1={f1:.4f}")

    # Phase 2: unfreeze everything, discriminative LR + cosine decay, early stopping.
    model.unfreeze_backbone()
    opt = torch.optim.Adam(model.param_groups(train_cfg["backbone_lr"], train_cfg["head_lr"]),
                           weight_decay=train_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg["finetune_epochs"])
    for epoch in range(train_cfg["finetune_epochs"]):
        loss = train_one_epoch(model, train_loader, opt, device, criterion, aug_cfg["mixup_alpha"], gen)
        sched.step()
        t, f1 = validate()
        print(f"[finetune {epoch+1}/{train_cfg['finetune_epochs']}] loss={loss:.4f} val_microF1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_threshold, patience = f1, t, 0
            save_checkpoint(ckpt_path, model, extra={"threshold": best_threshold, "val_micro_f1": best_f1})
        else:
            patience += 1
            if patience >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation on the official IRMAS test set using the tuned threshold.
    results = {"val_micro_f1": best_f1, "threshold": best_threshold}
    test_dir = Path(data_cfg["test_dir"])
    if test_dir.exists() and any(test_dir.rglob("*.wav")):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
        model.to(device)
        sr, ev = data_cfg["sample_rate"], config["eval"]
        y_true, y_scores = gather_test_scores(
            model, IRMASTestDataset(test_dir, sr), device,
            int(round(ev["window_seconds"] * sr)), int(round(ev["hop_seconds"] * sr)), ev["aggregate"],
        )
        results["test"] = evaluate_scores(y_true, y_scores, best_threshold)
        print("TEST:", results["test"])

    save_metrics(out_dir / "metrics.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune CNN14 on IRMAS")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
