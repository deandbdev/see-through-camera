"""Стъпка 8, измерване с известен отговор: изкуствена фигура, която ходи по пода.

  .venv/Scripts/python.exe src/test_dynamic.py data/odom/20261125_134157

Върху записаните кадри (само предна камера) лепим изрязан човек (от примерната снимка на
ultralytics), висок FIGURE_CM, на известно място в света. Сравняваме с картата от чистите кадри:
колко от пода е "замърсен" без и със динамичния слой, и колко точно намираме мястото на фигурата.
Това е лепенка, а не истински обект: без сянка, без ъгъл на гледане. И трите карти ползват
позите от чистите кадри.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import common as c
from dynamic import Detector, locate
from motion import rot
from projection import FloorProjector
from static_map import MapBuilder, replay_source

FIGURE_CM = 25.0
PATH_Y = 55.0                      # фигурата ходи напречно на тази линия (между маркерите)
PATH_X, PERIOD_S = (10.0, 50.0), 8.0
CONTAMINATED = 90                  # разлика (сбор по трите канала), над която пикселът е сгрешен


def cutout(detector):
    """Най-едрият човек от примерната снимка: (изображение, маска)."""
    import ultralytics

    path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    mask = max((m for n, _, m in detector.detect(image) if n == "person"), key=lambda m: (m > 0).sum())
    ys, xs = np.nonzero(mask)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return image[box], mask[box]


def position(t):
    phase = (t / PERIOD_S) % 1.0
    share = 2 * phase if phase < 0.5 else 2 - 2 * phase
    return np.array((PATH_X[0] + share * (PATH_X[1] - PATH_X[0]), PATH_Y))


def paste(frame, figure, mask, proj, pose, world):
    """Кадър с фигурата, стъпила в world; None, ако не се вижда."""
    local = rot(-pose[2]) @ (world - pose[:2])
    if not 20 < local[1] < 160 or abs(local[0]) > 0.7 * local[1]:
        return None
    foot, ok_foot = proj.project_3d(local, 0)
    head, ok_head = proj.project_3d(local, FIGURE_CM)
    if not (ok_foot and ok_head):
        return None
    h = int(round(foot[1] - head[1]))
    if not 10 < h < 2000:
        return None
    w = max(2, int(h * figure.shape[1] / figure.shape[0]))
    img, alpha = cv2.resize(figure, (w, h)), cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    x0, y0 = int(foot[0] - w / 2), int(foot[1]) - h
    left, top = max(x0, 0), max(y0, 0)
    right, bottom = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if right - left < 5 or bottom - top < 5:
        return None
    out = frame.copy()
    part = (slice(top - y0, bottom - y0), slice(left - x0, right - x0))
    cv2.copyTo(img[part], alpha[part], out[top:bottom, left:right])
    return out


class SavedPoses:
    """Вместо MapPose: позите от чистия запис, за да се сравнява само лепенето, а не позиционирането."""

    def __init__(self, poses):
        self.poses = iter(poses)

    def update(self, fusion):
        return next(self.poses)


def aligned(source, target):
    """Изображението и видяното на source в пикселите на target (картите започват от различни места)."""
    shift = (source.origin - target.origin) * source.s * np.array((1, -1))
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    size = target.image.shape[1::-1]
    image = cv2.warpAffine(source.image, M, size, flags=cv2.INTER_NEAREST)
    seen = cv2.warpAffine(np.isfinite(source.dist).astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
    return image, seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    a = p.parse_args()

    detector = Detector()
    figure, figure_mask = cutout(detector)
    proj = FloorProjector(0)
    frames = [(stamp, frame) for stamp, frame, _ in replay_source(a.folder, False)]
    start = frames[0][0]

    clean = MapBuilder()
    poses = [clean.process(frame, None, stamp) for stamp, frame in frames]

    pasted, errors, shown, found = [], [], 0, 0
    for (stamp, frame), pose in zip(frames, poses):
        world = position(stamp - start)
        out = None if pose is None else paste(frame, figure, figure_mask, proj, pose, world)
        pasted.append(frame if out is None else out)
        if out is None:
            continue
        shown += 1
        hits = []
        for name, _, mask in detector.detect(out):
            spot = locate(mask, proj)
            if name == "person" and spot is not None and not spot[4]:
                estimate = pose[:2] + rot(pose[2]) @ spot[:2]
                hits.append(float(np.linalg.norm(estimate - world)))
        if hits and min(hits) < 40:  # по-далечните са други хора в записа
            found += 1
            errors.append(min(hits))

    maps = {}
    for label, objects in (("без слоя", False), ("със слоя", True)):
        builder = MapBuilder(objects=objects)
        builder.map_pose = SavedPoses(poses)
        for (stamp, _), frame in zip(frames, pasted):
            builder.process(frame, None, stamp)
        maps[label] = builder.floor_map

    truth_image, truth_seen = clean.floor_map.image, np.isfinite(clean.floor_map.dist)
    rows, pictures = [], []
    _, (x0, y0) = clean.floor_map.cropped()
    h, w = clean.floor_map.cropped()[0].shape[:2]
    crop = (slice(y0, y0 + h), slice(x0, x0 + w))
    pictures.append(truth_image[crop])
    ppc2 = clean.floor_map.s ** 2
    for label, floor_map in maps.items():
        image, seen = aligned(floor_map, clean.floor_map)
        both = seen & truth_seen
        wrong = (np.abs(image.astype(int) - truth_image.astype(int)).sum(-1) > CONTAMINATED) & both
        wrong = cv2.morphologyEx(wrong.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        rows.append((label, int(wrong.sum() / ppc2), round(float(wrong.sum() / both.sum()) * 100, 2),
                     round(float((truth_seen & ~seen).sum() / truth_seen.sum()) * 100, 2)))
        mark = image.copy()
        mark[wrong] = (0, 0, 255)
        pictures.append(mark[crop])

    stamp_name = time.strftime("%Y%m%d_%H%M%S")
    path = c.DATA_DIR / "metrics" / f"dynamic_{stamp_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(("recording", "map", "wrong_cm2", "wrong_percent", "lost_coverage_percent",
                         "figure_frames", "found_frames", "position_error_median_cm"))
        median = round(float(np.median(errors)), 1) if errors else ""
        writer.writerows((Path(a.folder).name, *row, shown, found, median) for row in rows)
    cv2.imencode(".png", np.hstack(pictures))[1].tofile(str(path.with_suffix(".png")))

    print(f"Фигурата се вижда в {shown} кадъра, открита в {found}")
    if errors:
        print(f"Грешка в мястото ѝ: медиана {np.median(errors):.1f} см, 90% под {np.percentile(errors, 90):.1f} см")
    for label, area, share, lost in rows:
        print(f"{label}: сгрешен под {area} см² ({share}%), изгубено покритие {lost}%")
    print(f"Записано: {path} (.png: чиста | без слоя | със слоя; червено = разлика)")


if __name__ == "__main__":
    main()
