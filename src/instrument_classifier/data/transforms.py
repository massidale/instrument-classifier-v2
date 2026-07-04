"""Waveform-level transforms: length normalization and augmentation.

All functions operate on torch tensors and accept an optional
``torch.Generator`` so augmentation is reproducible under a fixed seed.
SpecAugment is applied inside the CNN14 model itself, so it is not here.
"""

from __future__ import annotations

import math

import torch


def pad_or_trim(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Force a 1D waveform to exactly ``length`` samples.

    Shorter signals are zero-padded at the end; longer signals are cropped
    from the start. Deterministic — random cropping is a separate concern.
    """
    n = waveform.shape[-1]
    if n == length:
        return waveform
    if n > length:
        return waveform[..., :length]
    pad = length - n
    return torch.nn.functional.pad(waveform, (0, pad))


def random_gain(
    waveform: torch.Tensor,
    max_db: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Multiply the signal by a random gain drawn uniformly in ``[-max_db, +max_db]``."""
    gain_db = (torch.rand(1, generator=generator).item() * 2 - 1) * max_db
    factor = 10.0 ** (gain_db / 20.0)
    return waveform * factor


def add_gaussian_noise(
    waveform: torch.Tensor,
    snr_db: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add white gaussian noise at approximately the requested SNR (in dB).

    For a silent input we fall back to a small fixed noise floor so the
    augmentation still perturbs the signal.
    """
    signal_power = waveform.pow(2).mean()
    if signal_power <= 0:
        noise_std = 1e-3
    else:
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = math.sqrt(float(noise_power))
    noise = torch.randn(waveform.shape, generator=generator) * noise_std
    return waveform + noise


class AugmentTransform:
    """Composable waveform augmentation for a DataLoader ``transform``.

    Applies a random gain and (optionally) gaussian noise. Picklable and
    top-level so it survives multiprocessing workers. ``gain_db == 0`` and
    ``noise_snr_db is None`` make it a no-op.
    """

    def __init__(self, gain_db: float, noise_snr_db: float | None, seed: int = 0):
        self.gain_db = gain_db
        self.noise_snr_db = noise_snr_db
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.gain_db and self.gain_db > 0:
            waveform = random_gain(waveform, self.gain_db, generator=self.generator)
        if self.noise_snr_db is not None:
            waveform = add_gaussian_noise(
                waveform, self.noise_snr_db, generator=self.generator
            )
        return waveform


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation for a batch.

    Mixes each sample with another (permuted) sample using a single
    ``lam ~ Beta(alpha, alpha)`` coefficient. With multi-hot targets the
    mixed target is a convex combination, which stays a valid multi-label
    signal for ``BCEWithLogitsLoss``. ``alpha == 0`` disables mixup.
    """
    if alpha <= 0.0:
        return x, y

    # Beta(alpha, alpha) via two Gamma draws, using the provided generator.
    g1 = torch._standard_gamma(torch.full((1,), alpha), generator=generator)
    g2 = torch._standard_gamma(torch.full((1,), alpha), generator=generator)
    lam = float((g1 / (g1 + g2)).item())

    perm = torch.randperm(x.shape[0], generator=generator)
    x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_mixed = lam * y + (1.0 - lam) * y[perm]
    return x_mixed, y_mixed
