# Use ultralytics to train a YOLO model
from ultralytics import YOLO

# Make sure to edit path!
model = YOLO("/mnt/mintuser-home/mint-profile/Desktop/dusk-dev/runs/detect/runs/train/captcha_test9/weights/best.pt")

# Made sure to remove the annoying rotate. 
# That was causing issues with `m` and `w`
model.train(
    data="data.yaml",
    epochs=40,
    batch=16,
    imgsz=384,
    workers=0,
    project="runs/train",
    name="captcha_test_stabilise",
    patience=10,
    fliplr=0.0,
    flipud=0.0,
)
