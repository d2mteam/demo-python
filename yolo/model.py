from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn


def _make_conv_block(block_cfg: Dict[str, Any]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = block_cfg["in_channels"]
    out_channels = block_cfg["out_channels"]
    kernel_size = block_cfg.get("kernel_size", 3)
    stride = block_cfg.get("stride", 1)
    padding = block_cfg.get("padding", 1)
    bias = not block_cfg.get("batchnorm", True)
    layers.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    )
    if block_cfg.get("batchnorm", True):
        layers.append(nn.BatchNorm2d(out_channels))
    activation = block_cfg.get("activation", "leaky_relu")
    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "identity":
        layers.append(nn.Identity())
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return nn.Sequential(*layers)


def _build_layers(blocks: List[Dict[str, Any]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for block in blocks:
        block_type = block["type"]
        if block_type == "conv":
            layers.append(_make_conv_block(block))
        elif block_type == "maxpool":
            layers.append(
                nn.MaxPool2d(
                    kernel_size=block.get("kernel_size", 2),
                    stride=block.get("stride", 2),
                )
            )
        elif block_type == "repeat":
            repeats = block["repeats"]
            for _ in range(repeats):
                layers.append(_build_layers(block["blocks"]))
        else:
            raise ValueError(f"Unsupported block type: {block_type}")
    return nn.Sequential(*layers)


class YoloV1(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config["model"]
        self.S = config["data"]["S"]
        self.B = config["data"]["B"]
        self.C = config["data"]["C"]
        self.image_size = config["data"]["image_size"]

        backbone_blocks = model_cfg["architecture"]
        self.backbone = _build_layers(backbone_blocks)

        with torch.no_grad():
            dummy = torch.zeros(1, model_cfg["input_channels"], self.image_size, self.image_size)
            features = self.backbone(dummy)
            flattened = features.view(1, -1)
            feature_dim = flattened.shape[1]

        head_cfg = model_cfg.get("head", {})
        hidden = head_cfg.get("fc_hidden", 4096)
        dropout = head_cfg.get("dropout", 0.5)
        out_dim = self.S * self.S * (self.C + self.B * 5)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x.view(x.size(0), self.S, self.S, self.C + self.B * 5)
