"""Маска на частите от колата, които камерата вижда (те не са под и не бива да влизат в картата).

  .venv/Scripts/python.exe src/make_mask.py data/odom/<папка>/rear --camera 1

Търси наситено зелено, опряно в долния край и стоящо на едно място през целия запис.
Записва calib/cam<N>_mask.png (бяло = игнорирай); FloorProjector я ползва сам.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

import common as c

HSV_LOW, HSV_HIGH = (35, 60, 40), (95, 255, 255)
MIN_SHARE = 0.8   # пикселът е зелен в поне толкова от кадрите
GROW_PX = 12


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--camera", type=int, required=True)
    a = p.parse_args()
    files = sorted(Path(a.folder).glob("*.jpg"))
    if not files:
        raise SystemExit(f"Няма кадри в {a.folder}")
    picked = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, 60).astype(int))]
    share = np.zeros(c.SIZE[::-1], np.float32)
    for f in picked:
        frame = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        share += cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), HSV_LOW, HSV_HIGH) > 0
    green = (share / len(picked) >= MIN_SHARE).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    _, labels = cv2.connectedComponents(green)
    bottom = [i for i in np.unique(labels[-5:]) if i]
    mask = np.isin(labels, bottom).astype(np.uint8) * 255
    if not mask.any():
        raise SystemExit("Не намирам зелена част, опряна в долния край.")
    mask = cv2.dilate(mask, np.ones((2 * GROW_PX + 1,) * 2, np.uint8))
    path = c.CALIB_DIR / f"cam{a.camera}_mask.png"
    cv2.imencode(".png", mask)[1].tofile(str(path))
    print(f"Изрязани {(mask > 0).mean():.0%} от кадъра -> {path}")


if __name__ == "__main__":
    main()
