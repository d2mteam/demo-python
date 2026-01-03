from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x_center, y_center, w, h (normalized)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes in xywh format (normalized)."""
    boxes1_xyxy = xywh_to_xyxy(boxes1)
    boxes2_xyxy = xywh_to_xyxy(boxes2)

    x1 = torch.max(boxes1_xyxy[..., 0:1], boxes2_xyxy[..., 0:1])
    y1 = torch.max(boxes1_xyxy[..., 1:2], boxes2_xyxy[..., 1:2])
    x2 = torch.min(boxes1_xyxy[..., 2:3], boxes2_xyxy[..., 2:3])
    y2 = torch.min(boxes1_xyxy[..., 3:4], boxes2_xyxy[..., 3:4])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1_xyxy[..., 2:3] - boxes1_xyxy[..., 0:1]).clamp(min=0) * (
        boxes1_xyxy[..., 3:4] - boxes1_xyxy[..., 1:2]
    ).clamp(min=0)
    area2 = (boxes2_xyxy[..., 2:3] - boxes2_xyxy[..., 0:1]).clamp(min=0) * (
        boxes2_xyxy[..., 3:4] - boxes2_xyxy[..., 1:2]
    ).clamp(min=0)

    union = area1 + area2 - inter_area + 1e-6
    return inter_area / union


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def non_max_suppression(
    detections: List[Detection],
    iou_threshold: float = 0.5,
) -> List[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda det: det.confidence, reverse=True)
    kept: List[Detection] = []
    while detections:
        current = detections.pop(0)
        kept.append(current)
        remaining = []
        for det in detections:
            if det.class_id != current.class_id:
                remaining.append(det)
                continue
            iou = compute_iou(
                torch.tensor([current.bbox]),
                torch.tensor([det.bbox]),
            ).item()
            if iou < iou_threshold:
                remaining.append(det)
        detections = remaining
    return kept
