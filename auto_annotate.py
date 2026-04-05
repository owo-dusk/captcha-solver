"""
Automatically annotate images based on a previously trained images
helped to reduce time that usually takes from 30 mins to around 10~ mins
Pretty useful when your model gains around 70%~ + accuracy!
"""

from ultralytics import YOLO
import cv2
import os
import random
#import shutil
import json

from rich.console import Console

# --- Configuration ---
MODEL_PATH = "/mnt/mintuser-home/mint-profile/Desktop/dusk-dev/runs/detect/runs/train/captcha_stabilise4/weights/best.pt"
IMAGE_DIR = "." # checking current folder.
CONF_THRES = 0.4
IMG_SIZE = 384

console = Console()
model = YOLO(MODEL_PATH)

files = os.listdir(IMAGE_DIR)
random.shuffle(files)

for img_name in files:
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    if img is None:
        continue

    orig = img.copy()

    results = model.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)

    detections = []

    for r in results:
        # Extract tensors as numpy arrays for float precision
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            # Format suitable for Labelme - [[x1, y1], [x2, y2]]
            points = [[float(x1), float(y1)], [float(x2), float(y2)]]

            detections.append(
                {
                    "label": model.names[int(classes[i])],
                    "conf": float(confs[i]),
                    "bbox_norm": points,  # Labelme style points
                    "cx": (x1 + x2) / 2,  # Float center for precise sorting
                }
            )

    # Sort detections by cx (Left-to-Right)
    detections.sort(key=lambda d: d["cx"])

    # Creation of Json file
    # This is the structure of Labelme json
    dummy_json = {
        "version": "5.11.2.dev8+g34f3aa9a8", # This is mine, may want to edit.
        "flags": {},
        "shapes": [
            # will be populated
        ],
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    for item in detections:
        shape = {
            "label": item["label"], # Use the actual predicted label
            "points": item["bbox_norm"],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        }
        dummy_json["shapes"].append(shape)

    json_filename = os.path.splitext(img_name)[0] + ".json"

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(dummy_json, f, indent=2)

    console.print(f"  [yellow]JSON saved as:[/yellow] {json_filename}")

    console.print("\n" + "=" * 40)
    console.print(f"Image: [cyan]{img_name}[/cyan]")
    console.print("Detections (Labelme Format):")

    str_res = ""
    for i, d in enumerate(detections):
        str_res += d["label"]
        # Log position
        console.print(f"  [green]{d['label']}[/green] | points: {d['bbox_norm']}")

    console.print(f"Result String: [bold white]{str_res}[/bold white]")

    src = os.path.join(IMAGE_DIR, img_name)

console.print("\n[bold yellow]Processing Complete.[/bold yellow]")
