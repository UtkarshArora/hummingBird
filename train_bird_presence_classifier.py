import os
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


ANNOTATION_PATH = (
    "./Label-Birdfeeder-Camera-Observations-3/train/_annotations.coco.json"
)
TRAIN_IMAGE_DIR = "./Label-Birdfeeder-Camera-Observations-3/train"
OUTPUT_DIR = "./Label-Birdfeeder-Camera-Observations-3"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


class BirdPresenceDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # float label for BCE
        y = torch.tensor([rec["label"]], dtype=torch.float32)
        return img, y


def build_items_from_coco(coco_json_path, image_dir):
    coco = json.load(open(coco_json_path, "r"))
    ann_count = defaultdict(int)
    for ann in coco["annotations"]:
        ann_count[ann["image_id"]] += 1

    items = []
    missing = 0
    for img in coco["images"]:
        image_id = img["id"]
        file_name = img["file_name"]
        path = os.path.join(image_dir, file_name)
        if not os.path.exists(path):
            missing += 1
            continue
        label = 1 if ann_count[image_id] > 0 else 0
        items.append({"image_id": image_id, "image_path": path, "label": label})

    print(f"✅ Built {len(items)} items (missing files skipped: {missing})")
    return items


def split_items(items, train_frac=0.8):
    items = items.copy()
    random.shuffle(items)
    n_train = int(train_frac * len(items))
    return items[:n_train], items[n_train:]


def compute_pos_weight(train_items):
    # pos_weight = (#neg / #pos) for BCEWithLogitsLoss
    pos = sum(x["label"] == 1 for x in train_items)
    neg = sum(x["label"] == 0 for x in train_items)
    if pos == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([neg / max(1, pos)], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    tp = fp = tn = fn = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()

        total += y.size(0)
        correct += (pred == y).sum().item()

        tp += ((pred == 1) & (y == 1)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    items = build_items_from_coco(ANNOTATION_PATH, TRAIN_IMAGE_DIR)
    train_items, val_items = split_items(items, train_frac=0.8)

    tf_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = BirdPresenceDataset(train_items, transform=tf_train)
    val_ds = BirdPresenceDataset(val_items, transform=tf_val)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Linear(backbone.fc.in_features, 1)
    model = backbone.to(device)

    pos_weight = compute_pos_weight(train_items).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    best_f1 = -1
    best_path = os.path.join(OUTPUT_DIR, "bird_presence_resnet18.pt")
    metrics_path = os.path.join(OUTPUT_DIR, "bird_presence_metrics.json")

    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)

        train_loss = running / max(1, len(train_ds))
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} val_f1={val_metrics['f1']:.3f} "
            f"val_recall={val_metrics['recall']:.3f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            json.dump(
                {"best_epoch": epoch, "best_val": val_metrics},
                open(metrics_path, "w"),
                indent=2,
            )
            print(f"✅ Saved best model to: {best_path}")

    print("Done.")
    print("Best model:", best_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
