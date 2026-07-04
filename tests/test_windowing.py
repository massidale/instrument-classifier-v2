import torch

from instrument_classifier.windowing import sliding_windows, aggregate_scores


def test_sliding_windows_exact_multiple():
    wav = torch.arange(30, dtype=torch.float32)
    w = sliding_windows(wav, window_len=10, hop_len=10)
    assert w.shape == (3, 10)
    assert torch.equal(w[0], torch.arange(0, 10, dtype=torch.float32))
    assert torch.equal(w[2], torch.arange(20, 30, dtype=torch.float32))


def test_sliding_windows_overlap():
    wav = torch.arange(20, dtype=torch.float32)
    w = sliding_windows(wav, window_len=10, hop_len=5)
    # windows start at 0, 5, 10 -> 3 windows
    assert w.shape == (3, 10)
    assert torch.equal(w[1], torch.arange(5, 15, dtype=torch.float32))


def test_sliding_windows_pads_last_partial_window():
    wav = torch.ones(12)
    w = sliding_windows(wav, window_len=10, hop_len=10)
    # 0..10 full, 10..20 partial -> padded with zeros
    assert w.shape == (2, 10)
    assert torch.all(w[1][:2] == 1.0)
    assert torch.all(w[1][2:] == 0.0)


def test_sliding_windows_short_signal_single_padded_window():
    wav = torch.ones(4)
    w = sliding_windows(wav, window_len=10, hop_len=5)
    assert w.shape == (1, 10)
    assert torch.all(w[0][:4] == 1.0)


def test_aggregate_mean():
    scores = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    out = aggregate_scores(scores, method="mean")
    assert torch.allclose(out, torch.tensor([0.3, 0.7]))


def test_aggregate_max():
    scores = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    out = aggregate_scores(scores, method="max")
    assert torch.allclose(out, torch.tensor([0.4, 0.8]))
