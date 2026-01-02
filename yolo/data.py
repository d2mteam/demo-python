from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection, VOCDetection


@dataclass
class Sample:
    image: torch.Tensor
    target: torch.Tensor


def encode_targets(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    B: int,
    C: int,
) -> torch.Tensor:
    """Encode boxes (xywh normalized) and labels into YOLOv1 target tensor."""
    target = torch.zeros((S, S, C + B * 5), dtype=torch.float32)
    for box, label in zip(boxes, labels):
        x, y, w, h = box.tolist()
        i, j = int(y * S), int(x * S)
        i = min(max(i, 0), S - 1)
        j = min(max(j, 0), S - 1)
        x_cell = x * S - j
        y_cell = y * S - i
        target[i, j, label] = 1.0
        for b in range(B):
            offset = C + b * 5
            target[i, j, offset : offset + 5] = torch.tensor(
                [x_cell, y_cell, w, h, 1.0]
            )
    return target


def decode_predictions(
    preds: torch.Tensor,
    S: int,
    B: int,
    C: int,
    conf_threshold: float,
) -> List[Tuple[int, float, List[float]]]:
    """Decode a single image prediction into list of (class_id, confidence, bbox)."""
    results = []
    for i in range(S):
        for j in range(S):
            cell = preds[i, j]
            class_probs = torch.softmax(cell[:C], dim=0)
            box_data = cell[C:].view(B, 5)
            for b in range(B):
                x_cell, y_cell, w, h, conf = box_data[b]
                x = (torch.sigmoid(x_cell) + j) / S
                y = (torch.sigmoid(y_cell) + i) / S
                w = torch.relu(w) ** 2
                h = torch.relu(h) ** 2
                conf = torch.sigmoid(conf)
                class_id = int(torch.argmax(class_probs))
                score = conf * class_probs[class_id]
                if score >= conf_threshold:
                    results.append(
                        (class_id, float(score), [float(x), float(y), float(w), float(h)])
                    )
    return results


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        year: str,
        image_set: str,
        image_size: int,
        class_map: Dict[str, int],
        S: int,
        B: int,
        C: int,
    ) -> None:
        self.voc = VOCDetection(root=root, year=year, image_set=image_set, download=False)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.class_map = class_map
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, idx: int) -> Sample:
        image, annotation = self.voc[idx]
        image = self.transform(image)
        boxes = []
        labels = []
        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        for obj in objects:
            name = obj["name"]
            if name not in self.class_map:
                continue
            bbox = obj["bndbox"]
            x_min = float(bbox["xmin"])
            y_min = float(bbox["ymin"])
            x_max = float(bbox["xmax"])
            y_max = float(bbox["ymax"])
            x = ((x_min + x_max) / 2) / image.shape[2]
            y = ((y_min + y_max) / 2) / image.shape[1]
            w = (x_max - x_min) / image.shape[2]
            h = (y_max - y_min) / image.shape[1]
            boxes.append([x, y, w, h])
            labels.append(self.class_map[name])
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        target = encode_targets(boxes_tensor, labels_tensor, self.S, self.B, self.C)
        return Sample(image=image, target=target)


class CocoDataset(Dataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        image_size: int,
        class_map: Dict[int, int],
        S: int,
        B: int,
        C: int,
    ) -> None:
        self.coco = CocoDetection(root=root, annFile=ann_file)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.class_map = class_map
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.coco)

    def __getitem__(self, idx: int) -> Sample:
        image, annotations = self.coco[idx]
        image = self.transform(image)
        boxes = []
        labels = []
        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id not in self.class_map:
                continue
            x_min, y_min, w, h = ann["bbox"]
            x = (x_min + w / 2) / image.shape[2]
            y = (y_min + h / 2) / image.shape[1]
            w = w / image.shape[2]
            h = h / image.shape[1]
            boxes.append([x, y, w, h])
            labels.append(self.class_map[cat_id])
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        target = encode_targets(boxes_tensor, labels_tensor, self.S, self.B, self.C)
        return Sample(image=image, target=target)


def load_image(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return transform(image)
