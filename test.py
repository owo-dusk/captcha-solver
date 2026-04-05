# This file is part of owo-dusk.
#
# Copyright (c) 2024-present EchoQuill
#
# Portions of this file are based on code by EchoQuill, licensed under the
# GNU General Public License v3.0 (GPL-3.0).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Helps test Yolo model. Run this to test accuracy against a real dataset
"""

from ultralytics import YOLO
import cv2
import os
import random
import shutil

from rich.console import Console
#from rich.text import Text

"""
Configuration
"""
MODEL_PATH = "/mnt/mintuser-home/mint-profile/Desktop/dusk-dev/runs/detect/runs/train/captcha_stabilise6/weights/best.pt" # Edit this!
IMAGE_DIR = "captcha"
MOVE_FOLDER = "f2"
SUCC_DIR = "core_succeeded"
# May want to change this to `0.6` or similar for real case situations
# But then again, for my case we do have the detail regarding how much letter counts
# Are in the captcha image, so I am allowing low threshold since I can filter them incase
# Count exceeds that of the letters in the captcha
CONF_THRES = 0.4 
IMG_SIZE = 384

WINDOW_NAME = "Captcha Detection"
WINDOW_W = 1200
WINDOW_H = 400
# I don't reccomend changing these
BOX_THICKNESS = 1
FONT_SCALE = 0.4
TEXT_THICKNESS = 1
TEXT_OFFSET = 5
""""""

console = Console()
model = YOLO(MODEL_PATH)

total_seen = 0
success_count = 0
failure_count = 0

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

files = os.listdir(IMAGE_DIR)
random.shuffle(files)

for img_name in files:
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    total_seen += 1

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    orig = img.copy()

    results = model.predict(
        source=img,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        verbose=False
    )

    detections = []

    for r in results:
        boxes_pixel = r.boxes.xyxy.cpu().numpy() 
        boxes_norm = r.boxes.xyxyn.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for i in range(len(boxes_pixel)):
            px1, py1, px2, py2 = boxes_pixel[i]
            nx1, ny1, nx2, ny2 = boxes_norm[i]
            
            cx = (nx1 + nx2) / 2
            
            detections.append({
                "label": model.names[int(classes[i])],
                "conf": float(confs[i]),
                "bbox": [int(px1), int(py1), int(px2), int(py2)], # Store as ints for CV2
                #"bbox_norm": [float(nx1), float(ny1), float(nx2), float(ny2)],
                "cx": float(cx)
            })

    detections.sort(key=lambda d: d["cx"])

    console.print("\n" + "=" * 40)
    console.print(f"Image: [cyan]{img_name}[/cyan]")
    console.print("Detections:")
    str_res = ""

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        #nx1, ny1, nx2, ny2 = d["bbox_norm"]

        console.print(
            f"{i:02d} | char='[steel_blue1]{d['label']}[/steel_blue1]' "
            f"conf=[plum2]{d['conf']:.2f}[/plum2] "
            #f"bbox=({x1},{y1},{x2},{y2})"
            #f"bbox_norm=({nx1},{ny1},{nx2},{ny2})"
        )

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)
        cv2.putText(
            orig,
            f"{d['label']} {d['conf']:.2f}",
            (x1, y1 - TEXT_OFFSET),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 255, 0),
            TEXT_THICKNESS,
            cv2.LINE_AA
        )
        str_res+=d['label']
    console.print(f"[medium_purple1]{str_res}[/medium_purple1]")
    print()

    cv2.imshow(WINDOW_NAME, orig)
    # Change title to that of image captcha result
    cv2.setWindowTitle(WINDOW_NAME, str_res) 
    key = cv2.waitKey(0)

    src = os.path.join(IMAGE_DIR, img_name)
    dst = ""

    if key == 27:
        # Stop the code incase ESC button is pressed
        break

    if key in (113, 81):  # Q
        # When `Q` key is pressed, the image is considered to be failed
        # And yes, I actually manually go through each images. Manually... ;-;
        dst = os.path.join(MOVE_FOLDER, img_name)
        failure_count += 1
        result_text = "[red]FAIL[/red]"
        shutil.move(src, dst)
    else:
        # Any other keys? Success!
        success_count+=1
        counter = 1
        dst = os.path.join(SUCC_DIR, f"{str_res}.png")

        while os.path.exists(dst):
            dst = os.path.join(SUCC_DIR, f"{str_res}_{counter}.png")
            counter += 1

        shutil.move(src, dst)


    accuracy = (success_count / total_seen) * 100

    console.print(
        f"Moved to: {dst} | "
        f"Accuracy: [bold thistle3]{accuracy:.2f}%[/bold thistle3] "
        f"(✔ {success_count} / ✘ {failure_count})"
    )

cv2.destroyAllWindows()

