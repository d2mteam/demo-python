# YOLOv1 Demo (PyTorch, cấu hình động)

Demo YOLOv1 phục vụ học tập với kiến trúc **khai báo bằng config (YAML/JSON)**. Model được build **động từ config**, không hardcode layer. Các tham số `S`, `B`, `C` và loss weights lấy từ config/CLI.

## Cấu trúc
```
.
├── configs/yolov1.yaml
├── train.py
├── infer.py
└── yolo/
    ├── config.py
    ├── data.py
    ├── loss.py
    ├── model.py
    └── utils.py
```

## Cài đặt
```
pip install torch torchvision pyyaml pillow
```

## Config kiến trúc
Ví dụ trong `configs/yolov1.yaml`:
- `model.architecture`: danh sách block `conv`, `maxpool`, `repeat`.
- `model.head`: cấu hình fully-connected head.
- `data`: `image_size`, `S`, `B`, `C`, `loss_weights`.

Bạn có thể thay đổi kiến trúc chỉ bằng sửa config (thêm/bớt block) mà **không cần sửa code model**.

## Shape/Tensor
Với mỗi ảnh:
- Input: `N x 3 x H x W` (mặc định 448x448).
- Output: `N x S x S x (C + 5B)`.
- Mỗi cell có `C` logits phân lớp và `B` boxes: `(x, y, w, h, conf)`.

## Encode/Decode
- **Encode** (`yolo.data.encode_targets`):
  - Box `x,y,w,h` chuẩn hoá [0,1] theo ảnh, `x,y` được đổi sang **tọa độ theo cell**:  
    `x_cell = x * S - j`, `y_cell = y * S - i`.
  - Với mỗi object gán vào cell `(i,j)` bằng `i = floor(y*S)`, `j = floor(x*S)`.
  - Target tensor có shape `S x S x (C + 5B)`.
- **Decode** (`yolo.data.decode_predictions`):
  - Từ output của model lấy `class_probs` (softmax) và `B` boxes.
  - Confidence = `conf * class_prob[class_id]`.
  - Dùng `NMS` trong `yolo.utils.non_max_suppression`.

## Loss YOLOv1
Implement trong `yolo.loss.YoloV1Loss`:
- **Coordinate loss**: `lambda_coord * [(x - x̂)^2 + (y - ŷ)^2 + (sqrt(w) - sqrt(ŵ))^2 + (sqrt(h) - sqrt(ĥ))^2]`.
- **Objectness loss**: chỉ box chịu trách nhiệm (IoU cao nhất) nhận loss object.
- **No-object loss**: phạt các box không chịu trách nhiệm.
- **Class loss**: MSE giữa `p(class)` và target class.

Các trọng số `coord/noobj/obj/class` lấy từ `data.loss_weights` trong config hoặc override bằng CLI.

## Train
```
python train.py --config configs/yolov1.yaml --epochs 10 --batch-size 8 --lr 1e-4
```
Override tham số:
```
python train.py --config configs/yolov1.yaml --S 7 --B 2 --C 20 --coord 5 --noobj 0.5
```

## Infer
```
python infer.py \
  --config configs/yolov1.yaml \
  --checkpoint checkpoints/yolov1_epoch_1.pt \
  --image path/to/image.jpg \
  --output outputs/result.jpg
```

## Dataset
- VOC: `torchvision.datasets.VOCDetection`.
- COCO: `torchvision.datasets.CocoDetection`.

Ví dụ cập nhật config cho COCO:
```yaml
  dataset:
    type: coco
    root: ./data/coco/images/train2017
    ann_file: ./data/coco/annotations/instances_train2017.json
    category_ids: [1, 2, 3]  # subset
```

## Ghi chú
- Demo này ưu tiên dễ đọc và phù hợp học tập.
- Mọi thay đổi kiến trúc chỉ cần sửa file YAML/JSON (không sửa code model).
