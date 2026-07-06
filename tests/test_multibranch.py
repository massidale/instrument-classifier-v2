import itertools

import pytest
import torch

from instrument_classifier.models.multibranch import BRANCH_DIMS, MultiBranchNet

ALL = ("mel", "cqt", "wave", "chroma")


def _inputs(batch=2, t=130):
    return {"mel": torch.rand(batch, 1, 128, t), "cqt": torch.rand(batch, 1, 84, t),
            "chroma": torch.rand(batch, 1, 12, t), "wave": torch.rand(batch, 66150)}


def _combos():
    for r in range(1, len(ALL) + 1):
        yield from itertools.combinations(ALL, r)


@pytest.mark.parametrize("combo", list(_combos()), ids="+".join)
def test_forward_every_branch_combination(combo):
    model = MultiBranchNet({k: k in combo for k in ALL}, pretrained=False).eval()
    out = model({k: v for k, v in _inputs().items() if k in combo})
    assert out["logits"].shape == (2, 11)
    assert out["embedding"].shape == (2, sum(BRANCH_DIMS[k] for k in combo))


def test_no_active_branch_raises():
    with pytest.raises(ValueError):
        MultiBranchNet({k: False for k in ALL}, pretrained=False)


def test_freeze_unfreeze_backbones():
    model = MultiBranchNet({"mel": True, "cqt": False, "wave": True, "chroma": False},
                           pretrained=False)
    model.freeze_backbones()
    backbone = [p for p in model.branches["mel"].backbone.parameters()]
    assert not any(p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in model.head.parameters())
    assert all(p.requires_grad for p in model.branches["wave"].parameters())
    model.unfreeze_backbones()
    assert all(p.requires_grad for p in backbone)


def test_param_groups_cover_all_trainables():
    model = MultiBranchNet({k: True for k in ALL}, pretrained=False)
    groups = model.param_groups(backbone_lr=1e-5, rest_lr=1e-3)
    ids_in_groups = {id(p) for g in groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert ids_in_groups == trainable
    assert groups[0]["lr"] == 1e-5 and groups[1]["lr"] == 1e-3
