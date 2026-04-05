"""
Label me creates a json with respective annotations for all captchas
These JSONs are used to convert into a format suitable for training YOLO model
"""

import json
import os
import shutil
from collections import defaultdict

# ===== CONFIG =====
# images + jsons live here
DATA_DIR = "under-annotation"
# This is where conerted images are moved to
TRAINABLE_DIR = "trainable"
CLASSES = "abcdefghijklmnopqrstuvwxyz"
# ==================

# Ensure directory to move annotations to actually exists
# If not, create one!
os.makedirs(TRAINABLE_DIR, exist_ok=True)

def label_to_id(label):
    # This function ensures labels aren't invalid.
    # If valid, returns index
    label = label.lower()
    if label not in CLASSES:
        raise ValueError(f"Unknown label: {label}")
    # pretty cool function to return index of a char from string!
    return CLASSES.index(label)

# For my case it is almost always .png but just incase!
IMAGE_EXTS = [".png", ".jpg", ".jpeg"]

# Move images! Iterates through all files to find images and their respective
# Json files to convert to a yolo accepted format.
for file in os.listdir(DATA_DIR):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(DATA_DIR, file)

    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        label = shape["label"].lower()

        (x1, y1), (x2, y2) = shape["points"]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        class_id = label_to_id(label)

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    if not yolo_lines:
        continue

    base_name = os.path.splitext(file)[0]

    # find corresponding image first
    image_path = None
    for ext in IMAGE_EXTS:
        candidate = os.path.join(DATA_DIR, base_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    # if no image exists, skip this JSON
    if image_path is None:
        print(f"⚠️ Skipping {file}: no corresponding image found")
        continue

    # write label file into trainable/ directory
    txt_path = os.path.join(TRAINABLE_DIR, base_name + ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # move image
    shutil.move(
        image_path,
        os.path.join(TRAINABLE_DIR, os.path.basename(image_path))
    )

    # Move Labelme Json data.
    shutil.move(
        json_path,
        os.path.join(TRAINABLE_DIR, os.path.basename(json_path))
    )


# Instances of Images
counts = defaultdict(int)

for file in os.listdir(TRAINABLE_DIR):
    if not file.endswith(".txt"):
        continue

    with open(os.path.join(TRAINABLE_DIR, file)) as f:
        for line in f:
            class_id = int(line.split()[0])
            counts[class_id] += 1

print("\nLabel distribution in trainable/")
print("-" * 35)

total = 0
for i, letter in enumerate(CLASSES):
    n = counts[i]
    total += n
    print(f"{letter}: {n}")

print("-" * 35)
print(f"Total labeled characters: {total}")

