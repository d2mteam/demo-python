from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from yolo.utils import compute_iou


class YoloV1Loss(nn.Module):
    def __init__(self, config: Dict[str, float]):
        super().__init__()
        self.S = config["S"]
        self.B = config["B"]
        self.C = config["C"]
        weights = config["loss_weights"]
        self.lambda_coord = weights.get("coord", 5.0)
        self.lambda_noobj = weights.get("noobj", 0.5)
        self.lambda_obj = weights.get("obj", 1.0)
        self.lambda_class = weights.get("class", 1.0)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds/targets shape: (N, S, S, C + B*5)
        N = preds.size(0)
        pred_boxes = preds[..., self.C : self.C + self.B * 5].view(N, self.S, self.S, self.B, 5)
        pred_classes = preds[..., : self.C]

        target_boxes = targets[..., self.C : self.C + self.B * 5].view(
            N, self.S, self.S, self.B, 5
        )
        target_classes = targets[..., : self.C]

        obj_mask = target_boxes[..., 4] > 0

        ious = compute_iou(
            pred_boxes[..., 0:4].reshape(-1, 4),
            target_boxes[..., 0:4].reshape(-1, 4),
        ).view(N, self.S, self.S, self.B)

        best_iou, best_idx = ious.max(dim=-1, keepdim=True)
        best_idx = best_idx.expand_as(obj_mask)

        responsible_mask = obj_mask & (torch.arange(self.B, device=preds.device)[None, None, None, :] == best_idx)

        coord_mask = responsible_mask.unsqueeze(-1)
        pred_xy = pred_boxes[..., 0:2]
        pred_wh = pred_boxes[..., 2:4]
        target_xy = target_boxes[..., 0:2]
        target_wh = target_boxes[..., 2:4]

        pred_wh_sqrt = torch.sign(pred_wh) * torch.sqrt(pred_wh.clamp(min=1e-6))
        target_wh_sqrt = torch.sqrt(target_wh.clamp(min=1e-6))

        coord_loss = (
            (pred_xy - target_xy) ** 2 + (pred_wh_sqrt - target_wh_sqrt) ** 2
        )
        coord_loss = coord_loss * coord_mask[..., 0:1]
        coord_loss = coord_loss.sum()

        pred_conf = pred_boxes[..., 4]
        obj_loss = ((pred_conf - best_iou.squeeze(-1)) ** 2) * responsible_mask
        obj_loss = obj_loss.sum()

        noobj_loss = (pred_conf ** 2) * (~responsible_mask)
        noobj_loss = noobj_loss.sum()

        class_loss = ((pred_classes - target_classes) ** 2) * (target_classes.sum(dim=-1, keepdim=True) > 0)
        class_loss = class_loss.sum()

        total = (
            self.lambda_coord * coord_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
        )
        return total / N
