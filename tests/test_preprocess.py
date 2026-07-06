import json

import numpy as np

from instrument_classifier.features import FEATURE_KEYS
from conftest import CLASSES, TEST_FC


def test_preprocess_writes_npz_and_stats(preprocessed_features):
    stats = json.loads((preprocessed_features / "stats.json").read_text())
    assert set(stats) == set(FEATURE_KEYS)
    for k in FEATURE_KEYS:
        assert set(stats[k]) == {"mean", "std"} and stats[k]["std"] > 0

    npzs = sorted(preprocessed_features.rglob("*.npz"))
    assert len(npzs) == len(CLASSES) * 4
    sample = np.load(npzs[0])
    assert set(sample.files) == set(FEATURE_KEYS)
    assert sample["mel"].dtype == np.float16
    assert sample["mel"].shape == (128, TEST_FC.n_frames)
    assert sample["wave"].shape == (TEST_FC.clip_len,)
