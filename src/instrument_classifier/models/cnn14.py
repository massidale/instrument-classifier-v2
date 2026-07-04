"""PANNs CNN14 backbone with a fresh classification head for finetuning.

The architecture faithfully reproduces Kong et al., "PANNs: Large-Scale
Pretrained Audio Neural Networks for Audio Pattern Recognition" (2020), so the
public AudioSet checkpoint ``Cnn14_mAP=0.431.pth`` loads directly. We then swap
the 527-way AudioSet head for an ``n_classes`` linear layer and return logits
(sigmoid is applied at inference), which pairs with ``BCEWithLogitsLoss``.

Reference: https://github.com/qiuqiangkong/audioset_tagging_cnn
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

# Front-end config baked into the released Cnn14_mAP=0.431 checkpoint.
_SAMPLE_RATE = 32000
_WINDOW_SIZE = 1024
_HOP_SIZE = 320
_MEL_BINS = 64
_FMIN = 50
_FMAX = 14000
_AUDIOSET_CLASSES = 527


def _init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(bn: nn.Module) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type="avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        return x


class Cnn14(nn.Module):
    """CNN14 taking a raw waveform and producing logits + a 2048-d embedding."""

    def __init__(self, classes_num: int = _AUDIOSET_CLASSES):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=_WINDOW_SIZE, hop_length=_HOP_SIZE, win_length=_WINDOW_SIZE,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=_SAMPLE_RATE, n_fft=_WINDOW_SIZE, n_mels=_MEL_BINS, fmin=_FMIN,
            fmax=_FMAX, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2,
        )
        self.bn0 = nn.BatchNorm2d(_MEL_BINS)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)

    # -- convenience groupings for finetuning ------------------------------
    def _backbone_modules(self):
        return [
            self.bn0, self.conv_block1, self.conv_block2, self.conv_block3,
            self.conv_block4, self.conv_block5, self.conv_block6, self.fc1,
        ]

    def freeze_backbone(self) -> None:
        for m in self._backbone_modules():
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for m in self._backbone_modules():
            for p in m.parameters():
                p.requires_grad = True

    def param_groups(self, backbone_lr: float, head_lr: float):
        head_params = list(self.fc_audioset.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in head_ids
        ]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        trainable_head = [p for p in head_params if p.requires_grad]
        if trainable_head:
            groups.append({"params": trainable_head, "lr": head_lr})
        return groups

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.spectrogram_extractor(waveform)  # (B, 1, T, freq)
        x = self.logmel_extractor(x)              # (B, 1, T, mel)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)              # pool over frequency
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2                            # pool over time
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        logits = self.fc_audioset(embedding)
        return {"logits": logits, "embedding": embedding}


def build_cnn14_finetune(
    num_classes: int,
    pretrained_path: str | Path | None,
) -> Cnn14:
    """Build CNN14, optionally load the AudioSet checkpoint, swap in a new head."""
    model = Cnn14(classes_num=_AUDIOSET_CLASSES)

    if pretrained_path is not None:
        checkpoint = torch.load(str(pretrained_path), map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)

    # Replace the 527-way AudioSet head with a fresh task head.
    model.fc_audioset = nn.Linear(2048, num_classes, bias=True)
    _init_layer(model.fc_audioset)
    return model
