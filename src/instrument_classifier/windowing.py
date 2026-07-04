"""Sliding-window helpers for evaluating variable-length IRMAS test clips.

The model is trained on fixed 3s windows, but IRMAS test clips are 5-20s.
We cut each clip into overlapping windows, score them, and pool the scores
into a single clip-level prediction.
"""

from __future__ import annotations

import torch


def sliding_windows(
    waveform: torch.Tensor, window_len: int, hop_len: int
) -> torch.Tensor:
    """Split a 1D waveform into ``(n_windows, window_len)`` overlapping windows.

    The final partial window is zero-padded so every clip yields at least one
    window, even when shorter than ``window_len``.
    """
    n = waveform.shape[-1]
    if n <= window_len:
        return _pad_to(waveform, window_len).unsqueeze(0)

    starts = list(range(0, n - window_len + 1, hop_len))
    # Ensure the tail of the signal is covered by a final (padded) window.
    if starts[-1] + window_len < n:
        starts.append(starts[-1] + hop_len)

    windows = []
    for s in starts:
        chunk = waveform[..., s : s + window_len]
        windows.append(_pad_to(chunk, window_len))
    return torch.stack(windows, dim=0)


def _pad_to(chunk: torch.Tensor, window_len: int) -> torch.Tensor:
    pad = window_len - chunk.shape[-1]
    if pad > 0:
        chunk = torch.nn.functional.pad(chunk, (0, pad))
    return chunk


def aggregate_scores(window_scores: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Pool per-window class scores ``(n_windows, n_classes)`` into ``(n_classes,)``."""
    if method == "mean":
        return window_scores.mean(dim=0)
    if method == "max":
        return window_scores.amax(dim=0)
    raise ValueError(f"unknown aggregation method: {method!r}")
