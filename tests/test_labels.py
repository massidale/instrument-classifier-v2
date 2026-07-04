import numpy as np
import pytest

from instrument_classifier.data.labels import (
    IRMAS_CLASSES,
    NUM_CLASSES,
    label_to_index,
    index_to_label,
    encode_labels,
    decode_prediction,
)


def test_eleven_irmas_classes_in_canonical_order():
    assert NUM_CLASSES == 11
    assert IRMAS_CLASSES == (
        "cel", "cla", "flu", "gac", "gel",
        "org", "pia", "sax", "tru", "vio", "voi",
    )


def test_label_index_roundtrip():
    for i, code in enumerate(IRMAS_CLASSES):
        assert label_to_index(code) == i
        assert index_to_label(i) == code


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        label_to_index("drums")


def test_encode_single_label_is_multi_hot():
    vec = encode_labels(["pia"])
    assert vec.shape == (11,)
    assert vec.dtype == np.float32
    assert vec[label_to_index("pia")] == 1.0
    assert vec.sum() == 1.0


def test_encode_multi_label():
    vec = encode_labels(["pia", "voi", "gel"])
    assert vec.sum() == 3.0
    for code in ("pia", "voi", "gel"):
        assert vec[label_to_index(code)] == 1.0


def test_encode_empty_is_all_zeros():
    vec = encode_labels([])
    assert vec.shape == (11,)
    assert vec.sum() == 0.0


def test_decode_prediction_applies_threshold():
    scores = np.zeros(11, dtype=np.float32)
    scores[label_to_index("pia")] = 0.9
    scores[label_to_index("voi")] = 0.4
    predicted = decode_prediction(scores, threshold=0.5)
    assert predicted == ["pia"]
