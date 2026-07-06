from instrument_classifier.utils import load_config


def test_default_config_schema():
    cfg = load_config("configs/default.yaml")
    assert set(cfg["branches"]) == {"mel", "cqt", "wave", "chroma"}
    assert all(isinstance(v, bool) for v in cfg["branches"].values())
    f = cfg["features"]
    assert f["sample_rate"] == 22050 and f["clip_seconds"] == 3.0
    assert f["n_fft"] == 2048 and f["hop_length"] == 512
    assert f["n_mels"] == 128 and f["cqt_bins"] == 84
    assert cfg["model"]["num_classes"] == 11
    assert cfg["data"]["features_dir"] == "data/features"
    assert cfg["eval"]["window_seconds"] == 3.0 and cfg["eval"]["hop_seconds"] == 1.0
