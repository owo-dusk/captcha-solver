"""
Helps move images to annotate and train from a directory of images.
This expects images to contain their label on the image name
"""

import os
import shutil

# ===== CONFIG =====
look_dir = "succeeded/failures"
move_dir = "under-annotation"
#quant = 30
lookfor = ["l", "t", "v", "i", "j", "p", "q", "r"]
avoid_chars = []

# ==================

quant = int(
    input("How many images are required?:\n")
)
print() # new line

os.makedirs(move_dir, exist_ok=True)

priority_all = []
priority_some = []

for file in os.listdir(look_dir):
    if not file.endswith(".png"):
        continue

    name = os.path.splitext(file)[0]

    # skip filenames with any avoided character
    if any(char in name for char in avoid_chars):
        continue

    # check whether letters are present
    has_all = all(char in name for char in lookfor) # preferred
    has_some = any(char in name for char in lookfor)

    if has_all:
        priority_all.append(file)
    elif has_some:
        priority_some.append(file)

selected = priority_all[:]

if len(selected) < quant:
    # from all available selected images, we only go with how many images we want.
    remaining = quant - len(selected)
    selected.extend(priority_some[:remaining])

# move selected files
for file in selected:
    src = os.path.join(look_dir, file)
    dst = os.path.join(move_dir, file)
    shutil.move(src, dst)

print(f"Moved {len(selected)} images to {move_dir}")
