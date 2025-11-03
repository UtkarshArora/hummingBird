import json
import albumentations as A
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import accuracy_score
import supervision as sv

from transformers import TrainerCallback
import torch
import math
from roboflow import Roboflow
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from datasets import Dataset
import os
import psutil
import subprocess
from PIL import Image
import random

# def compute_metrics(eval_pred):
#     predictions, references = eval_pred
#     metric = MeanAveragePrecision()
#     metric.update(predictions, references)
#     return metric.compute()

def compute_metrics(eval_pred):
    """
    Calculate accuracy for object detection.
    For RT-DETR, this computes detection accuracy.
    """
    predictions, labels = eval_pred
    
    # For object detection, you typically compare:
    # - Predicted boxes vs ground truth boxes
    # - Using IoU (Intersection over Union) threshold
    
    # Simple approach: count correct detections
    # (Adjust based on your specific needs)
    
    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions
    
    # Get predicted class (argmax)
    preds = np.argmax(logits, axis=-1)
    
    # Flatten and compare
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(labels_flat, preds_flat)
    
    return {"accuracy": accuracy}


# class for detecting if loss is NaN, because it should be a number
class NanLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and torch.isnan(torch.tensor(logs["loss"])):
            print("NaN loss detected. Stopping training.")
            control.should_training_stop = True


# Filter annotations to keep only category_id == 2 (hummingbird)
"""
def filter_hummingbird_annotations(annotation_path):
    with open(annotation_path, "r") as f:
        data = json.load(f)

    filtered_annotations = [
        ann for ann in data["annotations"] if ann["category_id"] == 2
    ]
    used_image_ids = {ann["image_id"] for ann in filtered_annotations}

    filtered_images = [img for img in data["images"] if img["id"] in used_image_ids]
    filtered_categories = [cat for cat in data["categories"] if cat["id"] == 2]

    cleaned_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories,
    }

    with open(annotation_path, "w") as f:
        json.dump(cleaned_data, f)


# we only need to show annotations which have a category_id of 2, because RT-DETR works on that

filter_hummingbird_annotations(
    "./Label-Birdfeeder-Camera-Observations-3/train/_annotations.coco.json"
)
"""
# getting data from Roboflow


# should api key be hidden somehow? Security issue.
rf = Roboflow(api_key="emHcbgLhITmU2KHvC6I7")
project = rf.workspace("humming-bird-detection").project(
    "label-birdfeeder-camera-observations"
)
version = project.version(3)
dataset = version.download("coco")

from transformers import (
    RTDetrForObjectDetection,
    RTDetrConfig,
    RTDetrImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_IMAGE_DIR = "./Label-Birdfeeder-Camera-Observations-3/train"


class RTDetrDataCollatorWithAugmentation:

    @staticmethod
    def xyxy_to_xywh(box):
        x1, y1, x2, y2 = box
        return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

    # Transforming the images while training so that the model learns better
    def __init__(self, processor, is_training=True):
        self.processor = processor
        self.is_training = is_training

        if self.is_training:
            self.transform = A.Compose(
                [
                    A.VerticalFlip(p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.3),
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7
                    ),
                    A.RandomBrightnessContrast(p=0.5),
                    A.OneOf(
                        [
                            A.Blur(blur_limit=5, p=1.0),
                            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                            A.MotionBlur(blur_limit=5, p=1.0),
                        ],
                        p=0.3,
                    ),
                    #A.GaussNoise(var_limit=(0.01, 0.05), p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                    A.Blur(blur_limit=3, p=0.1),
                    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.5),
                    A.RandomShadow(p=0.2),
                    A.RandomFog(p=0.1),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["category_ids"],
                    min_visibility=0.3,
                    min_area=25,
                ),
            )
        else:
            self.transform = None

    def __call__(self, batch):
        pixel_values = []
        labels = []

        for example in batch:
            try:
                image = Image.open(example["image_path"]).convert("RGB")
                image_np = np.array(image)
                orig_image_np = image_np.copy()  # Save original image

                bboxes = example["objects"]["bbox"]
                category_ids = example["objects"]["category_id"]
                orig_bboxes = bboxes.copy()  # Save original bboxes
                orig_category_ids = category_ids.copy()  # Save original categories

                # Apply augmentation only during training
                apply_aug = self.is_training and torch.is_grad_enabled()
                if apply_aug and self.transform:
                    transformed = self.transform(
                        image=image_np, bboxes=bboxes, category_ids=category_ids
                    )

                    # Only use augmented result if there's at least one valid box
                    if len(transformed["bboxes"]) > 0:
                        image_np = transformed["image"]
                        bboxes = transformed["bboxes"]
                        category_ids = transformed["category_ids"]
                    else:
                        # Restore original image and boxes
                        image_np = orig_image_np
                        bboxes = orig_bboxes
                        category_ids = orig_category_ids

                        # Reduce log spam - only show 1% of warnings
                        if random.random() < 0.01:
                            print(
                                f"Augmentation removed all boxes for image {example['image_id']}. Using original."
                            )

                # Prepare target annotations
                target = {
                    "image_id": example["image_id"],
                    "annotations": [
                        {
                            "bbox":self.xyxy_to_xywh(box),
                            "category_id": label,
                            "area": max(0.0, (box[2] - box[0]) * (box[3] - box[1])),
                            "iscrowd": 0,
                        }
                        for box, label in zip(bboxes, category_ids)
                    ],
                }

                # Process the image and annotations
                encoding = self.processor(
                    images=image_np, annotations=target, return_tensors="pt"
                )

                pixel_values.append(encoding["pixel_values"].squeeze())
                labels.append(encoding["labels"][0])

            except Exception as e:
                print(
                    f"Skipping corrupted sample {example.get('image_id', 'unknown')}: {e}"
                )
                continue

        if not pixel_values:
            return {}

        return {"pixel_values": torch.stack(pixel_values), "labels": labels}


ANNOTATION_PATH = (
    "./Label-Birdfeeder-Camera-Observations-3/train/_annotations.coco.json"
)
OUTPUT_DIR = "./Label-Birdfeeder-Camera-Observations-3"

with open(ANNOTATION_PATH, "r") as f:
    coco = json.load(f)

random.seed(42)
images = coco["images"]
random.shuffle(images)

split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
valid_images = images[split_idx:]

train_image_ids = {img["id"] for img in train_images}
valid_image_ids = {img["id"] for img in valid_images}
train_annotations = [
    ann for ann in coco["annotations"] if ann["image_id"] in train_image_ids
]
valid_annotations = [
    ann for ann in coco["annotations"] if ann["image_id"] in valid_image_ids
]

split_coco = {
    "train": {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco["categories"],
    },
    "valid": {
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": coco["categories"],
    },
}

with open(os.path.join(OUTPUT_DIR, "train_split.json"), "w") as f:
    json.dump(split_coco["train"], f)
with open(os.path.join(OUTPUT_DIR, "valid_split.json"), "w") as f:
    json.dump(split_coco["valid"], f)

print("✅ Dataset split complete: train_split.json and valid_split.json created.")


def remap_category_ids(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    for ann in data["annotations"]:
        if ann["category_id"] == 2:
            ann["category_id"] = 0

    # Overwrite categories to match the config
    data["categories"] = [{"id": 0, "name": "hummingbird"}]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


# Run after split
remap_category_ids(os.path.join(OUTPUT_DIR, "train_split.json"))
remap_category_ids(os.path.join(OUTPUT_DIR, "valid_split.json"))
print("✅ category_id=2 changed to 0 in both split files.")


def load_coco_dataset(image_dir, annotation_file):
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)

    dataset_entries = []
    for image_id, image_info in images.items():
        image_path = os.path.join(image_dir, image_info["file_name"])
        if not os.path.exists(image_path):
            continue

        image_annotations = annotations_by_image.get(image_id, [])
        objects = {"bbox": [], "category_id": [], "area": [], "iscrowd": []}
        for ann in image_annotations:
            x, y, w, h = ann["bbox"]
            objects["bbox"].append([x, y, x + w, y + h])
            objects["category_id"].append(0)
            objects["area"].append(ann.get("area", w * h))
            objects["iscrowd"].append(ann.get("iscrowd", 0))

        dataset_entries.append(
            {"image_id": image_id, "image_path": image_path, "objects": objects}
        )

    return dataset_entries


TRAIN_ANNOTATIONS = "./Label-Birdfeeder-Camera-Observations-3/train_split.json"
TEST_ANNOTATIONS = "./Label-Birdfeeder-Camera-Observations-3/valid_split.json"

print("Loading training dataset...")
train_data = load_coco_dataset(TRAIN_IMAGE_DIR, TRAIN_ANNOTATIONS)
train_dataset = Dataset.from_list(train_data)

print("Loading test dataset...")
test_data = load_coco_dataset(TRAIN_IMAGE_DIR, TEST_ANNOTATIONS)
test_dataset = Dataset.from_list(test_data)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
base_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
new_config = RTDetrConfig.from_pretrained(
    "PekingU/rtdetr_r50vd",
    num_labels=1,
    id2label={0: "hummingbird"},
    label2id={"hummingbird": 0},
)
model = RTDetrForObjectDetection(new_config)
model.model.backbone.load_state_dict(base_model.model.backbone.state_dict())

PRIOR_PROB = 0.01
bias_value = -math.log((1 - PRIOR_PROB) / PRIOR_PROB)

# Initialize encoder score head
model.enc_score_head = torch.nn.Linear(base_model.model.enc_score_head.in_features, 1)
torch.nn.init.normal_(model.model.enc_score_head.weight, std=0.01)
torch.nn.init.constant_(model.model.enc_score_head.bias, bias_value)

# Initialize decoder class embeddings
for i in range(len(model.model.decoder.class_embed)):
    model.model.decoder.class_embed[i] = torch.nn.Linear(
        base_model.model.decoder.class_embed[i].in_features, 1
    )
    torch.nn.init.normal_(model.model.decoder.class_embed[i].weight, std=0.01)
    torch.nn.init.constant_(model.model.decoder.class_embed[i].bias, bias_value)

del base_model
model.to(device)
for param in model.model.backbone.parameters():
    param.requires_grad = False

print("Model architecture verification:")
print(
    f"Encoder head: in_features={model.model.enc_score_head.in_features}, out_features={model.model.enc_score_head.out_features}"
)
for i, layer in enumerate(model.model.decoder.class_embed):
    print(
        f"Decoder class_embed[{i}]: in_features={layer.in_features}, out_features={layer.out_features}"
    )
# processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
# new_config = RTDetrConfig.from_pretrained(
#     "PekingU/rtdetr_r50vd",
#     num_labels=1,
#     id2label={0: "hummingbird"},
#     label2id={"hummingbird": 0}
# )

# model = RTDetrForObjectDetection.from_pretrained(
#     "PekingU/rtdetr_r50vd",
#     config=new_config,
#     ignore_mismatched_sizes=True
# )
# model.to(device)

# IMPROVED TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=64,  # Reduced for better gradient updates
    per_device_eval_batch_size=64,
    num_train_epochs=50,  # Reduced from 70
    learning_rate=5e-4,
    max_grad_norm=1.0,
    weight_decay=0.01,  # Increased regularization
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,  # More frequent saves
    remove_unused_columns=False,
    dataloader_num_workers=8,
    save_total_limit=5,
    load_best_model_at_end=True,
    eval_strategy="steps",
    eval_steps=50,  # More frequent evaluation
    warmup_steps=500,  # Reduced warmup
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",  # Better LR scheduling
    report_to="none",
    gradient_accumulation_steps=2,  # Effective batch size = 4
    dataloader_drop_last=False,
    seed=42,
    fp16=torch.cuda.is_available(),  # Mixed precision for faster training
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    tf32=False,
    torch_compile=False,
    dataloader_pin_memory=True,
   # gradient_checkpointing=True,
    # compute_metrics=compute_metrics,
)

# Initialize improved data collators
train_data_collator = RTDetrDataCollatorWithAugmentation(
    processor=processor, is_training=True
)
eval_data_collator = RTDetrDataCollatorWithAugmentation(
    processor=processor, is_training=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=train_data_collator,
   # compute_metrics=compute_metrics,  # Add this
)

def train_with_checkpoint_resume(trainer, training_args):
    """Train and automatically resume from checkpoint if it exists."""
    checkpoint_dir = Path(training_args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Find latest checkpoint
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
    )

    if checkpoints:
        last_ckpt = str(checkpoints[-1])
        print(f"✅ Resuming training from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("🆕 No checkpoint found. Starting fresh training...")
        trainer.train()
"""
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
"""
# Create trainer with early stopping
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=train_data_collator,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=10, early_stopping_threshold=0.001
        ),
        NanLossCallback(),
    ],
)
"""
print("Starting training...")
#trainer.train()
train_with_checkpoint_resume(trainer, training_args)

# from pathlib import Path

# checkpoint_dir = Path(training_args.output_dir)
# checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
# if checkpoints:
#     last_ckpt = str(checkpoints[-1])
#     print(f"Resuming training from checkpoint: {last_ckpt}")
#     trainer.train(resume_from_checkpoint=last_ckpt)
# else:
#     print("No checkpoint found. Starting fresh training...")
#     trainer.train()

# IMPROVED EVALUATION
import supervision as sv


def evaluate_model_improved():
    # Load test annotations
    with open(TEST_ANNOTATIONS, "r") as f:
        coco = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    image_id_to_size = {
        img["id"]: (img["width"], img["height"]) for img in coco["images"]
    }

    anns_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_per_image[ann["image_id"]].append(ann)

    model.eval()
    targets = []
    predictions = []

    # Use multiple thresholds for better detection
    thresholds = [0.05, 0.1, 0.15, 0.2]  # Much lower thresholds

    for image_id, file_name in image_id_to_file.items():
        img_path = os.path.join(TRAIN_IMAGE_DIR, file_name)
        image = Image.open(img_path).convert("RGB")
        w, h = image_id_to_size[image_id]

        # Ground truth processing
        gt_boxes = []
        gt_labels = []
        for ann in anns_per_image[image_id]:
            x, y, box_w, box_h = ann["bbox"]
            gt_boxes.append([x, y, x + box_w, y + box_h])
            gt_labels.append(0)

        if gt_boxes:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_boxes = np.empty((0, 4), dtype=np.float32)
            gt_labels = np.empty((0,), dtype=np.int64)

        targets.append(sv.Detections(xyxy=gt_boxes, class_id=gt_labels))

        # Prediction with multiple thresholds
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Try multiple thresholds and take the best
        best_results = None
        best_score = 0

        for threshold in thresholds:
            results = processor.post_process_object_detection(
                outputs, target_sizes=[(h, w)], threshold=threshold
            )[0]

            if len(results["boxes"]) > 0 and results["scores"].max() > best_score:
                best_results = results
                best_score = results["scores"].max()

        if best_results is not None:
            pred_boxes = best_results["boxes"].cpu().numpy()
            pred_scores = best_results["scores"].cpu().numpy()
            pred_labels = best_results["labels"].cpu().numpy()
        else:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_scores = np.empty((0,), dtype=np.float32)
            pred_labels = np.empty((0,), dtype=np.int64)

        predictions.append(
            sv.Detections(xyxy=pred_boxes, class_id=pred_labels, confidence=pred_scores)
        )

    # Calculate mAP
    if targets and predictions:
        try:
            mean_average_precision = sv.MeanAveragePrecision.from_detections(
                predictions=predictions, targets=targets
            )
            print(f"mAP@[.5:.95]: {mean_average_precision.map50_95:.3f}")
            print(f"mAP@.5: {mean_average_precision.map50:.3f}")
            print(f"mAP@.75: {mean_average_precision.map75:.3f}")
            return mean_average_precision
        except Exception as e:
            print(f"Error calculating mAP: {e}")
            return None
    else:
        print("No valid predictions or targets found for evaluation")
        return None


print("Evaluating model...")
evaluate_model_improved()

