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
