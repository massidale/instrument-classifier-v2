"""MultiBranchNet: a multi-input CNN for instrument recognition.

Each audio representation gets its own branch; embeddings are concatenated
(late fusion) and classified by an MLP head with 11 independent sigmoid
outputs (multi-label). Branches are toggleable from config, which is what
makes the per-branch ablation study possible.

  mel    (1,128,T) -> ResNet18 (ImageNet) -> 512
  cqt    (1, 84,T) -> ResNet18 (ImageNet) -> 512
  wave   (66150,)  -> Conv1D stack (scratch) -> 256
  chroma (1, 12,T) -> small Conv2D (scratch) -> 128
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

BRANCH_DIMS = {"mel": 512, "cqt": 512, "wave": 256, "chroma": 128}


class ImageBranch(nn.Module):
    """Spectrogram-as-image branch: input BN + 1->3 channel repeat + ResNet18.

    Repeating the mono channel keeps the pretrained first conv intact; the
    input BatchNorm adapts spectrogram statistics to what the backbone expects."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(1)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()  # expose the 512-d pooled embedding
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, F, T)
        return self.backbone(self.bn_in(x).repeat(1, 3, 1, 1))


class WaveBranch(nn.Module):
    """Raw-waveform branch trained from scratch: 5 conv blocks, ~4x downsample each."""

    def __init__(self):
        super().__init__()
        chans = [1, 32, 64, 128, 256, 256]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv1d(cin, cout, kernel_size=9, padding=4, bias=False),
                       nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.MaxPool1d(4)]
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, samples)
        return self.pool(self.net(x.unsqueeze(1))).squeeze(-1)


class ChromaBranch(nn.Module):
    """Small 2D CNN for the 12-bin chroma map."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, 12, T)
        return self.net(x).flatten(1)


_BRANCH_FACTORIES = {
    "mel": lambda pretrained: ImageBranch(pretrained),
    "cqt": lambda pretrained: ImageBranch(pretrained),
    "wave": lambda pretrained: WaveBranch(),
    "chroma": lambda pretrained: ChromaBranch(),
}


class MultiBranchNet(nn.Module):
    def __init__(self, branches: dict[str, bool], num_classes: int = 11,
                 pretrained: bool = True, head_hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.active = sorted(k for k, on in branches.items() if on)
        if not self.active:
            raise ValueError("MultiBranchNet needs at least one active branch")
        unknown = set(self.active) - set(_BRANCH_FACTORIES)
        if unknown:
            raise ValueError(f"Unknown branches: {sorted(unknown)}")
        self.branches = nn.ModuleDict(
            {k: _BRANCH_FACTORIES[k](pretrained) for k in self.active})
        in_dim = sum(BRANCH_DIMS[k] for k in self.active)
        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(head_hidden, num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = torch.cat([self.branches[k](inputs[k]) for k in self.active], dim=1)
        return {"logits": self.head(embedding), "embedding": embedding}

    # -- finetuning helpers --------------------------------------------------
    def _backbone_params(self):
        for k in self.active:
            branch = self.branches[k]
            if isinstance(branch, ImageBranch):
                yield from branch.backbone.parameters()

    def freeze_backbones(self) -> None:
        for p in self._backbone_params():
            p.requires_grad = False

    def unfreeze_backbones(self) -> None:
        for p in self._backbone_params():
            p.requires_grad = True

    def param_groups(self, backbone_lr: float, rest_lr: float) -> list[dict]:
        backbone = [p for p in self._backbone_params() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters()
                if p.requires_grad and id(p) not in backbone_ids]
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr})
        if rest:
            groups.append({"params": rest, "lr": rest_lr})
        return groups


def build_model(config: dict) -> MultiBranchNet:
    m = config["model"]
    return MultiBranchNet(branches=config["branches"], num_classes=m["num_classes"],
                          pretrained=m["pretrained"], head_hidden=m["head_hidden"],
                          dropout=m["dropout"])
