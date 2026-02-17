# check_model.py
# Usage examples:
#   python check_model.py --ckpt ./outputs_hb_finetune/checkpoint-700
#   python check_model.py --ckpt ./outputs_hb_finetune/checkpoint-700 --threshold 0.15 --num-images 25
#   python check_model.py --ckpt ./outputs_hb_finetune/checkpoint-700 --single-image ./some.jpg
#
# What it does:
#  1) Evaluates mAP on your COCO val split (hb_valid_split.json)
#  2) Saves visualizations with predicted bounding boxes to ./pred_viz/

import os
import json
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Optional (nice viz). If not installed, we fall back to saving raw boxes only.
try:
    import supervision as sv

    _HAS_SV = True
except Exception:
    _HAS_SV = False


def load_coco(annotation_path: str):
    with open(annotation_path, "r") as f:
        return json.load(f)


def build_gt_index(coco: dict):
    """Return:
    - id->file_name
    - id->(w,h)
    - anns_per_image: image_id -> list[ann]
    """
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    id2size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    anns_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_per_image[ann["image_id"]].append(ann)

    return id2file, id2size, anns_per_image


def to_xyxy_from_coco_bbox(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def evaluate_map(
    model, processor, coco, image_dir: str, device: torch.device, threshold: float
):
    """
    Computes mAP using supervision (if available). If supervision isn't available,
    prints a basic detection-rate sanity metric instead.
    """
    id2file, id2size, anns_per_image = build_gt_index(coco)

    predictions = []
    targets = []

    for image_id, file_name in id2file.items():
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = id2size[image_id]

        # GT
        gt_boxes = []
        gt_labels = []
        for ann in anns_per_image[image_id]:
            gt_boxes.append(to_xyxy_from_coco_bbox(ann["bbox"]))
            gt_labels.append(int(ann["category_id"]))
        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if gt_boxes
            else np.empty((0, 4), dtype=np.float32)
        )
        gt_labels = (
            np.array(gt_labels, dtype=np.int64)
            if gt_labels
            else np.empty((0,), dtype=np.int64)
        )

        # Pred
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=threshold
        )[0]

        pred_boxes = (
            results["boxes"].detach().cpu().numpy()
            if len(results["boxes"])
            else np.empty((0, 4), dtype=np.float32)
        )
        pred_scores = (
            results["scores"].detach().cpu().numpy()
            if len(results["scores"])
            else np.empty((0,), dtype=np.float32)
        )
        pred_labels = (
            results["labels"].detach().cpu().numpy()
            if len(results["labels"])
            else np.empty((0,), dtype=np.int64)
        )

        if _HAS_SV:
            predictions.append(
                sv.Detections(
                    xyxy=pred_boxes, confidence=pred_scores, class_id=pred_labels
                )
            )
            targets.append(sv.Detections(xyxy=gt_boxes, class_id=gt_labels))
        else:
            predictions.append((pred_boxes, pred_scores, pred_labels))
            targets.append((gt_boxes, gt_labels))

    if _HAS_SV:
        try:
            m = sv.MeanAveragePrecision.from_detections(
                predictions=predictions, targets=targets
            )
            print("\n=== mAP (supervision) ===")
            print(f"mAP@[.5:.95]: {m.map50_95:.4f}")
            print(f"mAP@.5     : {m.map50:.4f}")
            print(f"mAP@.75    : {m.map75:.4f}")
            print("========================\n")
        except Exception as e:
            print(f"mAP computation failed: {e}")
    else:
        # Basic fallback: detection rate (not true accuracy)
        det_any = sum(1 for (pb, _, _), _ in zip(predictions, targets) if len(pb) > 0)
        total = len(predictions)
        print("\n(supervision not available) Basic sanity metric:")
        print(
            f"Images with >=1 detection @ threshold={threshold}: {det_any}/{total} ({(det_any/total*100 if total else 0):.1f}%)\n"
        )


def visualize_predictions(
    model,
    processor,
    coco,
    image_dir: str,
    device: torch.device,
    out_dir: str,
    num_images: int,
    threshold: float,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    id2file, id2size, _ = build_gt_index(coco)
    image_ids = list(id2file.keys())
    random.Random(seed).shuffle(image_ids)
    image_ids = image_ids[:num_images]

    if not _HAS_SV:
        print(
            "Note: 'supervision' not installed; will save raw prediction txt files instead of annotated images."
        )

    for image_id in image_ids:
        file_name = id2file[image_id]
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = id2size[image_id]

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=threshold
        )[0]

        pred_boxes = (
            results["boxes"].detach().cpu().numpy()
            if len(results["boxes"])
            else np.empty((0, 4), dtype=np.float32)
        )
        pred_scores = (
            results["scores"].detach().cpu().numpy()
            if len(results["scores"])
            else np.empty((0,), dtype=np.float32)
        )
        pred_labels = (
            results["labels"].detach().cpu().numpy()
            if len(results["labels"])
            else np.empty((0,), dtype=np.int64)
        )

        base = os.path.splitext(os.path.basename(file_name))[0]
        if _HAS_SV:
            det = sv.Detections(
                xyxy=pred_boxes, confidence=pred_scores, class_id=pred_labels
            )
            annotator = sv.BoxAnnotator()
            labels = [f"{int(c)}:{s:.2f}" for c, s in zip(det.class_id, det.confidence)]
            img_np = np.array(image)
            annotated = annotator.annotate(scene=img_np, detections=det, labels=labels)
            out_path = os.path.join(out_dir, f"{base}_pred.png")
            Image.fromarray(annotated).save(out_path)
        else:
            out_path = os.path.join(out_dir, f"{base}_pred.txt")
            with open(out_path, "w") as f:
                for b, s, c in zip(pred_boxes, pred_scores, pred_labels):
                    f.write(f"class={int(c)} score={float(s):.4f} xyxy={b.tolist()}\n")

    print(f"Saved predictions to: {out_dir}")


def run_single_image(model, processor, device, img_path: str, threshold: float):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=threshold
    )[0]

    boxes = (
        results["boxes"].detach().cpu().numpy()
        if len(results["boxes"])
        else np.empty((0, 4), dtype=np.float32)
    )
    scores = (
        results["scores"].detach().cpu().numpy()
        if len(results["scores"])
        else np.empty((0,), dtype=np.float32)
    )
    labels = (
        results["labels"].detach().cpu().numpy()
        if len(results["labels"])
        else np.empty((0,), dtype=np.int64)
    )

    print(f"\nPredictions for: {img_path}")
    if len(boxes) == 0:
        print(f"  (no detections @ threshold={threshold})")
        return
    for b, s, c in zip(boxes, scores, labels):
        print(f"  class={int(c)} score={float(s):.3f} box_xyxy={b.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint dir, e.g. ./outputs_hb_finetune/checkpoint-700",
    )
    parser.add_argument(
        "--image-dir",
        default="./Label-Birdfeeder-Camera-Observations-3/train",
        help="Directory containing images",
    )
    parser.add_argument(
        "--ann",
        default="./Label-Birdfeeder-Camera-Observations-3/hb_valid_split.json",
        help="COCO annotation json (validation split)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Score threshold for post-processing",
    )
    parser.add_argument(
        "--num-images", type=int, default=20, help="How many val images to visualize"
    )
    parser.add_argument(
        "--out-dir", default="./pred_viz", help="Output directory for visualizations"
    )
    parser.add_argument(
        "--single-image",
        default=None,
        help="If set, only run inference on this image path",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip mAP evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    model = RTDetrForObjectDetection.from_pretrained(args.ckpt).to(device)
    model.eval()

    if args.single_image:
        run_single_image(model, processor, device, args.single_image, args.threshold)
        return

    coco = load_coco(args.ann)

    if not args.no_eval:
        evaluate_map(model, processor, coco, args.image_dir, device, args.threshold)

    visualize_predictions(
        model=model,
        processor=processor,
        coco=coco,
        image_dir=args.image_dir,
        device=device,
        out_dir=args.out_dir,
        num_images=args.num_images,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
