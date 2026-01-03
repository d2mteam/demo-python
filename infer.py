import argparse
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont

from yolo.config import load_config
from yolo.data import decode_predictions, load_image
from yolo.model import YoloV1
from yolo.utils import Detection, non_max_suppression


def build_index_to_name(dataset_cfg: Dict) -> Dict[int, str]:
    if dataset_cfg["type"] == "voc":
        class_names = dataset_cfg["classes"]
        return {idx: name for idx, name in enumerate(class_names)}
    if dataset_cfg["type"] == "coco":
        category_ids = dataset_cfg["category_ids"]
        return {idx: str(cat_id) for idx, cat_id in enumerate(category_ids)}
    raise ValueError("Unsupported dataset type")


def draw_boxes(image: Image.Image, detections: List[Detection], class_map: Dict[int, str]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        x, y, w, h = det.bbox
        x1 = (x - w / 2) * image.width
        y1 = (y - h / 2) * image.height
        x2 = (x + w / 2) * image.width
        y2 = (y + h / 2) * image.height
        label = f"{class_map.get(det.class_id, det.class_id)}: {det.confidence:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference YOLOv1 demo")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = YoloV1(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_tensor = load_image(args.image, config["data"]["image_size"]).unsqueeze(0).to(args.device)
    with torch.no_grad():
        preds = model(image_tensor).squeeze(0)
    preds = preds.sigmoid()

    decoded = decode_predictions(
        preds,
        config["data"]["S"],
        config["data"]["B"],
        config["data"]["C"],
        conf_threshold=args.conf,
    )
    detections = [Detection(class_id=c, confidence=score, bbox=box) for c, score, box in decoded]
    detections = non_max_suppression(detections, iou_threshold=args.nms)

    image = Image.open(args.image).convert("RGB")
    class_names = build_index_to_name(config["data"]["dataset"])
    image = draw_boxes(image, detections, class_names)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
