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
                "branches": config["branches"], "stats": full_plain.stats,
                "model_config": {"num_classes": config["model"]["num_classes"],
                                 "head_hidden": config["model"]["head_hidden"],
                                 "dropout": config["model"]["dropout"]}})
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
        # Checkpoint warmup weights too, so a degrading finetune can't discard a
        # superior frozen-backbone model.
        checkpoint_if_best(f1, t)

    # Reset patience so early stopping counts only finetune epochs.
    patience = 0

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

    if best_f1 < 0:  # no epoch ever checkpointed (warmup=0 and finetune=0)
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
