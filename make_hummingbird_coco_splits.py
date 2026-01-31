import os
import json
import random
from collections import defaultdict

ANNOTATION_PATH = (
    "./Label-Birdfeeder-Camera-Observations-3/train/_annotations.coco.json"
)
OUTPUT_DIR = "./Label-Birdfeeder-Camera-Observations-3"

SEED = 42
random.seed(SEED)

HUMMINGBIRD_ORIGINAL_ID = 2


def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


def save_coco(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_anns_by_img(coco):
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    return anns_by_img


def filter_bird_present_images(coco):
    anns_by_img = build_anns_by_img(coco)
    kept_images = [img for img in coco["images"] if len(anns_by_img[img["id"]]) > 0]
    kept_ids = {img["id"] for img in kept_images}
    return kept_images, kept_ids


def keep_only_hummingbird_annotations(coco, kept_image_ids, hb_cat_id=2):
    new_anns = []
    new_id = 1
    for ann in coco["annotations"]:
        if ann["image_id"] not in kept_image_ids:
            continue
        if ann["category_id"] == hb_cat_id:
            a = dict(ann)
            a["id"] = new_id
            new_id += 1
            a["category_id"] = 0
            new_anns.append(a)
    return new_anns


def split_images(images, train_frac=0.8):
    images = images.copy()
    random.shuffle(images)
    n_train = int(train_frac * len(images))
    return images[:n_train], images[n_train:]


def subset_annotations(annotations, image_id_set):
    return [a for a in annotations if a["image_id"] in image_id_set]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    coco = load_coco(ANNOTATION_PATH)

    # 1) remove no-bird (no-annotation) images
    kept_images, kept_ids = filter_bird_present_images(coco)
    print(f"✅ Bird-present images kept: {len(kept_images)} / {len(coco['images'])}")

    # 2) keep only hummingbird anns (category_id==2) and remap -> 0
    hb_annotations = keep_only_hummingbird_annotations(
        coco, kept_ids, hb_cat_id=HUMMINGBIRD_ORIGINAL_ID
    )
    print(
        f"✅ Hummingbird annotations kept: {len(hb_annotations)} / {len(coco['annotations'])}"
    )

    # 3) split images
    train_images, valid_images = split_images(kept_images, train_frac=0.8)
    train_ids = {img["id"] for img in train_images}
    valid_ids = {img["id"] for img in valid_images}

    train_anns = subset_annotations(hb_annotations, train_ids)
    valid_anns = subset_annotations(hb_annotations, valid_ids)

    out_train = {
        "images": train_images,
        "annotations": train_anns,
        "categories": [{"id": 0, "name": "hummingbird"}],
    }
    out_valid = {
        "images": valid_images,
        "annotations": valid_anns,
        "categories": [{"id": 0, "name": "hummingbird"}],
    }

    train_out_path = os.path.join(OUTPUT_DIR, "hb_train_split.json")
    valid_out_path = os.path.join(OUTPUT_DIR, "hb_valid_split.json")
    save_coco(out_train, train_out_path)
    save_coco(out_valid, valid_out_path)

    print("✅ Wrote:")
    print("  ", train_out_path)
    print("  ", valid_out_path)

    # Optional: show how many train images are positive vs negative (hb boxes)
    anns_by_train = defaultdict(int)
    for a in train_anns:
        anns_by_train[a["image_id"]] += 1
    pos = sum(1 for img in train_images if anns_by_train[img["id"]] > 0)
    neg = len(train_images) - pos
    print(
        f"Train: {pos} images with hb boxes, {neg} hard-negative images (birds but not hb)."
    )


if __name__ == "__main__":
    main()
