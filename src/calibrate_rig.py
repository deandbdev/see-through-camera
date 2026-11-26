"""Къде е задната камера спрямо предната, сметнато от общ запис с маркерите.

Двете камери не се припокриват, но виждат едни и същи маркери в различни моменти:
позицията на колата (предна камера + одометрия) ги свързва.

  1) .venv/Scripts/python.exe src/static_map.py --rear --record
     Започни с маркерите отпред и карай напред, докато задната камера ги види добре.
  2) .venv/Scripts/python.exe src/calibrate_rig.py data/odom/<папка>         # само показва
     .venv/Scripts/python.exe src/calibrate_rig.py data/odom/<папка> --save  # и записва rig.json
"""
import argparse
import json

import cv2
import numpy as np

from localize import Fusion, compose
from motion import DualOdometry, ransac_rigid, rot
from projection import FloorProjector
from static_map import RIG_PATH, MapPose, load_rig, rear_offset, replay_source
from tags import TagLocalizer

MIN_SAMPLES = 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--save", action="store_true")
    a = p.parse_args()

    rig = load_rig()
    front = FloorProjector(rig["front_camera"])
    rear = FloorProjector(rig["rear_camera"])
    tags, rear_tags = TagLocalizer(front), TagLocalizer(rear)
    # одометрията ползва и задната (с досегашното ѝ положение): пред стена предната лъже
    fusion = Fusion(front, tags, DualOdometry(front, rear, rear_offset(rig)))
    map_pose = MapPose("tags")

    seen_local, seen_vehicle = [], []  # центрове на маркери: спрямо задната камера / спрямо колата
    for _, frame, rear_frame in replay_source(a.folder, use_rear=True):
        fusion.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                      None if rear_frame is None else cv2.cvtColor(rear_frame, cv2.COLOR_BGR2GRAY))
        pose = map_pose.update(fusion)
        if pose is None or rear_frame is None:
            continue
        for marker_id, (_, floor) in rear_tags.detect(cv2.cvtColor(rear_frame, cv2.COLOR_BGR2GRAY)).items():
            m = tags.markers[marker_id]
            seen_local.append(floor.mean(0))
            seen_vehicle.append((np.array([m["x"], m["y"]]) - pose[:2]) @ rot(pose[2]))

    if len(seen_local) < MIN_SAMPLES:
        raise SystemExit(f"Задната камера е видяла маркери само {len(seen_local)} пъти "
                         f"(трябват {MIN_SAMPLES}). Карай по-напред, след като ги подминеш.")
    local, vehicle = np.array(seen_local), np.array(seen_vehicle)
    old = rear_offset(rig)
    before = np.linalg.norm(local @ rot(old[2]).T + old[:2] - vehicle, axis=1)
    fit = ransac_rigid(local, vehicle, iterations=500, threshold=5.0)
    if fit is None:
        raise SystemExit("Наблюденията не са съгласувани; повтори записа по-бавно.")
    theta, t, inliers = fit
    after = np.linalg.norm(local @ rot(theta).T + t - vehicle, axis=1)[inliers]
    spread = float(np.linalg.norm(local[inliers] - local[inliers].mean(0), axis=1).std())

    new = {**rig, "right_cm": round(float(t[0]), 1), "back_cm": round(float(-t[1]), 1),
           "yaw_deg": round(float(-np.degrees(theta)) % 360, 1)}
    print(f"Наблюдения: {len(local)}, съгласувани: {int(inliers.sum())}")
    for key, name in (("back_cm", "назад, см"), ("right_cm", "надясно, см"), ("yaw_deg", "посока, °")):
        print(f"  {name:12s} в rig.json {rig[key]:6.1f}   по записа {new[key]:6.1f}")
    print(f"Грешка на маркерите през задната камера: преди {np.median(before):.1f} см, "
          f"след {np.median(after):.1f} см (медиана)")
    if spread < 8:
        print("ВНИМАНИЕ: маркерите са видени почти на едно място спрямо камерата, посоката е несигурна.")
    if a.save:
        RIG_PATH.write_text(json.dumps(new, indent=2))
        print(f"Записано в {RIG_PATH}")


if __name__ == "__main__":
    main()
