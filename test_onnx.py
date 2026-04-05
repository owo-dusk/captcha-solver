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
Test the model after it has been converted to ONNX format.
"""

import onnxruntime
import numpy as np
import cv2

"""Configurations"""
# Make sure to edit the patchs as required.
ONNX_MODEL_PATH = "/mnt/mintuser-home/mint-profile/Desktop/dusk-dev/runs/detect/runs/train/captcha_test_stabilise/weights/best.onnx"
IMAGE_PATH = "succeeded/captcha-52972.png"
IMG_SIZE = 384
CLASSES = "abcdefghijklmnopqrstuvwxyz"
CONF_THRES = 0.3

# Loads the model
session = onnxruntime.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

inputs = session.get_inputs()


"""for i in inputs:
    print(i.name, i.shape, i.type)"""
#   images [1, 3, 384, 384] tensor(float)
# [1, 3, 384, 384] -> [batch, channels, height, width]


input_name = inputs[0].name


def letterbox(img, new_size=384, color=(114,114,114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)

    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh))

    pad_w = new_size - nw
    pad_h = new_size - nh

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img_padded

# Read image.
img = cv2.imread(IMAGE_PATH)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = letterbox(img, IMG_SIZE)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Actual detection part
outputs = session.run(None, {input_name: img})[0]
detections_raw = outputs[0]

detections = []

for det in detections_raw:
    x1, y1, x2, y2, conf, cls_id = det

    if conf < CONF_THRES:
        continue

    cls_id = int(cls_id)

    detections.append({
        "char": CLASSES[cls_id],
        "conf": float(conf),
        "cx": float((x1 + x2) / 2)
    })


detections.sort(key=lambda d: d["cx"])

captcha = "".join(d["char"] for d in detections)

print("Predicted CAPTCHA:", captcha)
print("Details:")
for d in detections:
    print(f"  {d['char']}  conf={d['conf']:.2f}")
