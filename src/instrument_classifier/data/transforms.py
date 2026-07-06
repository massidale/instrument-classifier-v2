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
