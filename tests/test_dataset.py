import numpy as np
import soundfile as sf
import torch

from instrument_classifier.data.dataset import (
    IRMASTrainDataset,
    IRMASTestDataset,
    parse_test_label_file,
)


def _write_wav(path, seconds=3.0, sr=44100, freq=440.0):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, audio, sr)


def _make_train_tree(tmp_path):
    for code in ("pia", "voi"):
        d = tmp_path / code
        d.mkdir()
        for i in range(3):
            _write_wav(d / f"{code}_{i}.wav")
    return tmp_path


def test_train_dataset_length_counts_all_files(tmp_path):
    root = _make_train_tree(tmp_path)
    ds = IRMASTrainDataset(root, sample_rate=32000, clip_seconds=3.0)
    assert len(ds) == 6


def test_train_item_shapes_and_target(tmp_path):
    root = _make_train_tree(tmp_path)
    ds = IRMASTrainDataset(root, sample_rate=32000, clip_seconds=3.0)
    wav, target = ds[0]
    assert wav.shape == (96000,)  # 32000 * 3.0, mono, resampled
    assert wav.dtype == torch.float32
    assert target.shape == (11,)
    assert target.sum() == 1.0  # single-label training clip


def test_train_dataset_resamples_and_fixes_length(tmp_path):
    d = tmp_path / "sax"
    d.mkdir()
    _write_wav(d / "long.wav", seconds=5.0, sr=44100)  # too long, wrong sr
    ds = IRMASTrainDataset(tmp_path, sample_rate=16000, clip_seconds=2.0)
    wav, _ = ds[0]
    assert wav.shape == (32000,)  # 16000 * 2.0


def test_parse_test_label_file_returns_valid_codes(tmp_path):
    p = tmp_path / "clip.txt"
    p.write_text("sax\tvio\n")
    codes = parse_test_label_file(p)
    assert sorted(codes) == ["sax", "vio"]


def test_parse_test_label_file_ignores_unknown_tokens(tmp_path):
    p = tmp_path / "clip.txt"
    p.write_text("pia\ndru\nvoi\n")  # 'dru' is not one of the 11 IRMAS classes
    codes = parse_test_label_file(p)
    assert sorted(codes) == ["pia", "voi"]


def test_test_dataset_returns_full_clip_and_multihot(tmp_path):
    _write_wav(tmp_path / "song.wav", seconds=7.0, sr=44100)
    (tmp_path / "song.txt").write_text("pia\nvoi\n")
    ds = IRMASTestDataset(tmp_path, sample_rate=32000)
    wav, target, name = ds[0]
    assert wav.ndim == 1
    # 7s resampled to 32kHz, NOT cropped (windowing happens at eval time)
    assert abs(wav.shape[0] - 7 * 32000) < 32000
    assert target.shape == (11,)
    assert target.sum() == 2.0
    assert "song" in name
