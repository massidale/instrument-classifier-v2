"""End-to-end smoke test on the synthetic dataset: 1 warmup + 2 finetune
epochs on CPU with pretrained=False must run to completion, save a best.pth
checkpoint in the expected format (model/threshold/val_micro_f1/branches/stats),
write metrics.json, and return test metrics including per-class F1."""
import torch

from instrument_classifier.train import run_training


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
