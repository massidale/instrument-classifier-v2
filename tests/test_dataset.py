import torch

from instrument_classifier.data.dataset import IRMASFeaturesDataset, IRMASTestDataset
from conftest import CLASSES, SR, TEST_FC


def test_features_dataset_shapes(preprocessed_features):
    ds = IRMASFeaturesDataset(preprocessed_features, branches=["mel", "cqt", "wave", "chroma"])
    assert len(ds) == len(CLASSES) * 4
    feats, target = ds[0]
    assert set(feats) == {"mel", "cqt", "wave", "chroma"}
    assert feats["mel"].shape == (1, 128, TEST_FC.n_frames)
    assert feats["cqt"].shape == (1, 84, TEST_FC.n_frames)
    assert feats["chroma"].shape == (1, 12, TEST_FC.n_frames)
    assert feats["wave"].shape == (TEST_FC.clip_len,)
    assert all(v.dtype == torch.float32 for v in feats.values())
    assert target.shape == (11,) and target.sum() == 1.0


def test_features_dataset_loads_only_active_branches(preprocessed_features):
    ds = IRMASFeaturesDataset(preprocessed_features, branches=["mel"])
    feats, _ = ds[0]
    assert set(feats) == {"mel"}


def test_features_dataset_targets_for_stratify(preprocessed_features):
    ds = IRMASFeaturesDataset(preprocessed_features, branches=["mel"])
    assert len(ds.targets()) == len(ds) and len(set(ds.targets())) == len(CLASSES)


def test_test_dataset_multilabel(synthetic_irmas):
    ds = IRMASTestDataset(synthetic_irmas / "IRMAS-TestingData", sample_rate=SR)
    wav, target, name = ds[0]
    assert wav.ndim == 1 and target.shape == (11,)
    assert target.sum() == 2.0 and name == "poly1"
