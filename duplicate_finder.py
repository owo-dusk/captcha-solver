"""
Prints exact-matching images. Since I fetch captchas myself, and recieved some captchas
from other sources, I need to ensure captchas are not used more than once.
"""

import os
import hashlib

# -------- CONFIG --------
IMAGE_DIR = "trainable"   # folder to scan
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
# ------------------------

def hash_file(path, chunk_size=8192):
    """Hash file content (byte-level, exact match)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

hash_map = {}

for fname in os.listdir(IMAGE_DIR):
    if not fname.lower().endswith(IMAGE_EXTS):
        continue

    path = os.path.join(IMAGE_DIR, fname)
    file_hash = hash_file(path)

    hash_map.setdefault(file_hash, []).append(fname)

# Report duplicates
duplicates_found = False

print("\nDuplicate images")
print("-----------------------------------")

for h, files in hash_map.items():
    if len(files) > 1:
        duplicates_found = True
        print(f"\nDuplicate group ({len(files)} images):")
        for f in files:
            print("  -", f)

if not duplicates_found:
    print("No exact duplicates found.")

