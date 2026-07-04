import torch

from instrument_classifier.data.transforms import (
    pad_or_trim,
    random_gain,
    add_gaussian_noise,
    mixup_batch,
    AugmentTransform,
)


def test_augment_transform_preserves_shape_and_perturbs():
    wav = torch.ones(2000)
    aug = AugmentTransform(gain_db=6.0, noise_snr_db=20.0, seed=0)
    out = aug(wav)
    assert out.shape == wav.shape
    assert not torch.equal(out, wav)


def test_augment_transform_disabled_is_identity():
    wav = torch.randn(2000)
    aug = AugmentTransform(gain_db=0.0, noise_snr_db=None, seed=0)
    out = aug(wav)
    assert torch.equal(out, wav)


def test_pad_or_trim_pads_short_signal_with_zeros():
    wav = torch.ones(100)
    out = pad_or_trim(wav, 160)
    assert out.shape == (160,)
    assert torch.all(out[:100] == 1.0)
    assert torch.all(out[100:] == 0.0)


def test_pad_or_trim_crops_long_signal():
    wav = torch.arange(500, dtype=torch.float32)
    out = pad_or_trim(wav, 160)
    assert out.shape == (160,)


def test_pad_or_trim_exact_length_is_identity():
    wav = torch.randn(160)
    out = pad_or_trim(wav, 160)
    assert torch.equal(out, wav)


def test_random_gain_preserves_shape_and_scales():
    wav = torch.ones(1000)
    gen = torch.Generator().manual_seed(0)
    out = random_gain(wav, max_db=6.0, generator=gen)
    assert out.shape == wav.shape
    # A pure gain multiplies every sample by the same positive constant.
    ratios = out / wav
    assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-6)
    assert ratios[0] > 0


def test_add_gaussian_noise_changes_signal_but_keeps_shape():
    wav = torch.zeros(1000)
    gen = torch.Generator().manual_seed(0)
    out = add_gaussian_noise(wav, snr_db=20.0, generator=gen)
    assert out.shape == wav.shape
    assert out.abs().sum() > 0  # noise was actually added


def test_mixup_with_zero_alpha_is_identity():
    x = torch.randn(4, 100)
    y = torch.eye(4)
    x_out, y_out = mixup_batch(x, y, alpha=0.0)
    assert torch.equal(x_out, x)
    assert torch.equal(y_out, y)


def test_mixup_produces_convex_combination():
    x = torch.randn(8, 50)
    y = torch.zeros(8, 11)
    y[:, 0] = 1.0
    gen = torch.Generator().manual_seed(42)
    x_out, y_out = mixup_batch(x, y, alpha=0.4, generator=gen)
    assert x_out.shape == x.shape
    assert y_out.shape == y.shape
    # Targets stay valid multi-label probabilities.
    assert torch.all(y_out >= 0.0)
    assert torch.all(y_out <= 1.0)
