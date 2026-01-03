import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from yolo.config import load_config
from yolo.data import CocoDataset, VOCDataset
from yolo.loss import YoloV1Loss
from yolo.model import YoloV1


def build_class_map(dataset_cfg: Dict) -> Dict:
    if dataset_cfg["type"] == "voc":
        class_names = dataset_cfg["classes"]
        return {name: idx for idx, name in enumerate(class_names)}
    if dataset_cfg["type"] == "coco":
        category_ids = dataset_cfg["category_ids"]
        return {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    raise ValueError("Unsupported dataset type")


def collate_fn(batch):
    images = torch.stack([sample.image for sample in batch])
    targets = torch.stack([sample.target for sample in batch])
    return images, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv1 demo")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--S", type=int)
    parser.add_argument("--B", type=int)
    parser.add_argument("--C", type=int)
    parser.add_argument("--coord", type=float)
    parser.add_argument("--noobj", type=float)
    parser.add_argument("--obj", type=float)
    parser.add_argument("--class-weight", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    for key, cli_value in (("S", args.S), ("B", args.B), ("C", args.C)):
        if cli_value is not None:
            config["data"][key] = cli_value
    loss_weights = config["data"].setdefault("loss_weights", {})
    if args.coord is not None:
        loss_weights["coord"] = args.coord
    if args.noobj is not None:
        loss_weights["noobj"] = args.noobj
    if args.obj is not None:
        loss_weights["obj"] = args.obj
    if args.class_weight is not None:
        loss_weights["class"] = args.class_weight

    dataset_cfg = config["data"]["dataset"]
    class_map = build_class_map(dataset_cfg)

    if dataset_cfg["type"] == "voc":
        dataset = VOCDataset(
            root=dataset_cfg["root"],
            year=dataset_cfg.get("year", "2012"),
            image_set=dataset_cfg.get("image_set", "train"),
            image_size=config["data"]["image_size"],
            class_map=class_map,
            S=config["data"]["S"],
            B=config["data"]["B"],
            C=config["data"]["C"],
        )
    else:
        dataset = CocoDataset(
            root=dataset_cfg["root"],
            ann_file=dataset_cfg["ann_file"],
            image_size=config["data"]["image_size"],
            class_map=class_map,
            S=config["data"]["S"],
            B=config["data"]["B"],
            C=config["data"]["C"],
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = YoloV1(config).to(args.device)
    criterion = YoloV1Loss(config["data"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for images, targets in loader:
            images = images.to(args.device)
            targets = targets.to(args.device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")
        checkpoint = output_dir / f"yolov1_epoch_{epoch + 1}.pt"
        torch.save({"model": model.state_dict(), "config": config}, checkpoint)


if __name__ == "__main__":
    main()
