"""IRMAS instrument label vocabulary and multi-label (de)coding.

IRMAS uses 11 instrument codes. We fix a canonical order so that model
outputs, targets and saved metrics all agree on the index of each class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Canonical order of the 11 IRMAS instrument codes.
IRMAS_CLASSES: tuple[str, ...] = (
    "cel",  # cello
    "cla",  # clarinet
    "flu",  # flute
    "gac",  # acoustic guitar
    "gel",  # electric guitar
    "org",  # organ
    "pia",  # piano
    "sax",  # saxophone
    "tru",  # trumpet
    "vio",  # violin
    "voi",  # human singing voice
)

# Human-readable names, aligned with IRMAS_CLASSES.
CLASS_NAMES: dict[str, str] = {
    "cel": "cello",
    "cla": "clarinet",
    "flu": "flute",
    "gac": "acoustic guitar",
    "gel": "electric guitar",
    "org": "organ",
    "pia": "piano",
    "sax": "saxophone",
    "tru": "trumpet",
    "vio": "violin",
    "voi": "voice",
}

NUM_CLASSES: int = len(IRMAS_CLASSES)

_LABEL_TO_INDEX: dict[str, int] = {code: i for i, code in enumerate(IRMAS_CLASSES)}


def label_to_index(code: str) -> int:
    """Map an IRMAS code (e.g. ``"pia"``) to its class index."""
    return _LABEL_TO_INDEX[code]


def index_to_label(index: int) -> str:
    """Map a class index back to its IRMAS code."""
    return IRMAS_CLASSES[index]


def encode_labels(codes: Sequence[str]) -> np.ndarray:
    """Encode a set of instrument codes as a multi-hot float32 vector."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for code in codes:
        vec[label_to_index(code)] = 1.0
    return vec


def decode_prediction(scores: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Turn per-class scores into the list of predicted instrument codes."""
    return [IRMAS_CLASSES[i] for i in range(NUM_CLASSES) if scores[i] >= threshold]
