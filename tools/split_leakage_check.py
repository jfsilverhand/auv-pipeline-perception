#!/usr/bin/env python3
"""Frame-level split leakage check: groups train/valid/test images by their source video.

Usage:
    python3 split_leakage_check.py DATA_ROOT

DATA_ROOT must contain train/, valid/ (or val/) and test/ folders; inside each, an img/ or
images/ subfolder is used if present, otherwise the folder itself.

A file's source video is the part of its name before "_mp4-NNNN" or "-NNNN_jpg" (Roboflow
frame-export naming). Files without this pattern are counted as "non-video" and are only
checked by source id after removing the "aug_" prefix and the Roboflow ".rf.<hash>" suffix.

Read-only: it never writes to DATA_ROOT. The dataset itself is not part of this repository.
"""
import os
import re
import sys
from collections import defaultdict

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_RE = re.compile(r"^(.*?)(?:_mp4-\d+|-\d+_jpg)")
SPLITS = {"train": ("train",), "valid": ("valid", "val"), "test": ("test",)}


def split_dir(root, names):
    for n in names:
        d = os.path.join(root, n)
        if os.path.isdir(d):
            for sub in ("img", "images"):
                if os.path.isdir(os.path.join(d, sub)):
                    return os.path.join(d, sub)
            return d
    return None


def images(d):
    return [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT] if d else []


def video_root(fname):
    m = VIDEO_RE.match(fname)
    return m.group(1) if m else None


def source_id(fname):
    return re.sub(r"\.rf\..*$|\.jpg$", "", fname).replace("aug_", "")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    groups, nonvideo, totals = {}, {}, {}
    for s, names in SPLITS.items():
        files = images(split_dir(root, names))
        g = defaultdict(list)
        nv = [f for f in files if video_root(f) is None]
        for f in files:
            if video_root(f) is not None:
                g[video_root(f)].append(f)
        groups[s], nonvideo[s], totals[s] = g, nv, len(files)
        print(f"{s:>5}: images={len(files)}  video frames={len(files) - len(nv)}  "
              f"non-video={len(nv)}  distinct videos={len(g)}")

    print()
    for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
        common = set(groups[a]) & set(groups[b])
        frames = sum(len(groups[b][v]) for v in common)
        pct = 100.0 * frames / totals[b] if totals[b] else 0.0
        print(f"{a} & {b}: shared videos={len(common)}, their {b} frames={frames} ({pct:.1f}% of {b})")

    tr = {source_id(f) for f in nonvideo["train"]}
    te = {source_id(f) for f in nonvideo["test"]}
    print(f"\nnon-video test source ids={len(te)}, also present in train={len(tr & te)}")


if __name__ == "__main__":
    main()
