import torch

from instrument_classifier.models.cnn14 import build_cnn14_finetune


def test_forward_returns_logits_and_embedding():
    model = build_cnn14_finetune(num_classes=11, pretrained_path=None)
    model.eval()
    wav = torch.randn(2, 32000)
    with torch.no_grad():
        out = model(wav)
    assert out["logits"].shape == (2, 11)
    assert out["embedding"].shape == (2, 2048)
    assert torch.isfinite(out["logits"]).all()


def test_head_has_requested_number_of_classes():
    model = build_cnn14_finetune(num_classes=7, pretrained_path=None)
    assert model.fc_audioset.out_features == 7


def test_freeze_backbone_only_trains_head():
    model = build_cnn14_finetune(num_classes=11, pretrained_path=None)
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.conv_block1.parameters())
    assert all(p.requires_grad for p in model.fc_audioset.parameters())


def test_unfreeze_backbone_trains_everything():
    model = build_cnn14_finetune(num_classes=11, pretrained_path=None)
    model.freeze_backbone()
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.conv_block1.parameters())


def test_param_groups_separate_backbone_and_head():
    model = build_cnn14_finetune(num_classes=11, pretrained_path=None)
    groups = model.param_groups(backbone_lr=1e-5, head_lr=1e-3)
    lrs = {g["lr"] for g in groups}
    assert lrs == {1e-5, 1e-3}
    # every trainable parameter appears in exactly one group
    n_grouped = sum(len(g["params"]) for g in groups)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grouped == n_total
