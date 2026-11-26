"""Стъпка 6: позициониране с ArUco и сравнение с одометрията.

  .venv/Scripts/python.exe src/localize.py             # на живо
  .venv/Scripts/python.exe src/localize.py --record    # и записва кадрите
  .venv/Scripts/python.exe src/localize.py --replay data/odom/<папка>

Жълто = само одометрия (подравнена веднъж, при първия видян маркер).
Зелено = коригирана позиция (нулира се при всеки видян маркер). R = начало, ESC = изход.
Кадър по кадър се записва в data/metrics/aruco_<време>.csv
"""
import argparse
import csv
import json
import time

import cv2
import numpy as np

import common as c
from calibrate import label
from motion import Odometry, rot
from odometry import live_frames, replay_frames
from projection import FloorProjector
from tags import TagLocalizer


def compose(a, b):
    """Поза b, изразена в системата на поза a."""
    return np.array([*(a[:2] + rot(a[2]) @ b[:2]), a[2] + b[2]])


def inverse(a):
    return np.array([*(-rot(-a[2]) @ a[:2]), -a[2]])


def wrap_deg(theta):
    return float((np.degrees(theta) + 180) % 360 - 180)


class Fusion:
    """Одометрия + маркери. Позите в света са None, докато не е видян първи маркер."""

    def __init__(self, projector, localizer, odom=None):
        self.odom = odom or Odometry(projector)
        self.tags = localizer
        self.first = self.latest = None  # преобразувания одометрия -> свят
        self.travel = self.travel_since_tag = 0.0
        self.prev = np.zeros(2)
        self.frames_without_tag = 0
        self.jumps = []  # (изминат път без маркер, грешка см, грешка градуси)

    def update(self, gray, rear_gray=None):
        """rear_gray се подава само когато одометрията е DualOdometry."""
        self.odom.update(gray) if rear_gray is None else self.odom.update(gray, rear_gray)
        step = float(np.linalg.norm(self.odom.pose[:2] - self.prev))
        if step >= 0.5:  # под това е шум
            self.travel += step
            self.travel_since_tag += step
            self.prev = self.odom.pose[:2].copy()

        self.found = self.tags.detect(gray)
        self.tag_pose = self.tags.locate(self.found)
        if self.tag_pose is not None:
            # грешката, натрупана докато не сме виждали маркер
            if self.latest is not None and self.frames_without_tag >= 5 and self.travel_since_tag >= 10:
                predicted = compose(self.latest, self.odom.pose)
                self.jumps.append((self.travel_since_tag,
                                   float(np.linalg.norm(predicted[:2] - self.tag_pose[:2])),
                                   wrap_deg(predicted[2] - self.tag_pose[2])))
                print(f"Маркер след {self.jumps[-1][0]:.0f} см без маркер: грешка "
                      f"{self.jumps[-1][1]:.1f} см, {self.jumps[-1][2]:+.1f}°")
            self.latest = compose(self.tag_pose, inverse(self.odom.pose))
            if self.first is None:
                self.first = self.latest
            self.travel_since_tag, self.frames_without_tag = 0.0, 0
        else:
            self.frames_without_tag += 1

        self.raw = None if self.first is None else compose(self.first, self.odom.pose)
        self.fused = None if self.latest is None else compose(self.latest, self.odom.pose)


def draw_map(localizer, raw_path, fused_path, size=600):
    img = np.zeros((size, size, 3), np.uint8)
    marks = np.array([(m["x"], m["y"]) for m in localizer.markers.values()], float)
    pts = np.concatenate([marks] + [np.array(p) for p in (raw_path, fused_path) if p])
    center = (pts.min(0) + pts.max(0)) / 2
    half = max(60.0, np.abs(pts - center).max() * 1.2)
    to_px = lambda p: (int(size / 2 + (p[0] - center[0]) / half * size / 2),
                       int(size / 2 - (p[1] - center[1]) / half * size / 2))
    for i, m in localizer.markers.items():
        quad = localizer.square @ rot(np.radians(m["yaw"] or 0)).T + (m["x"], m["y"])
        color = (255, 255, 255) if m["yaw"] is not None else (110, 110, 110)
        cv2.polylines(img, [np.array([to_px(p) for p in quad])], True, color, 1)
        cv2.putText(img, str(i), to_px((m["x"], m["y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for path, color in ((raw_path, (0, 220, 255)), (fused_path, (0, 255, 0))):
        if path:
            cv2.polylines(img, [np.array([to_px(p) for p in path])], False, color, 2)
            cv2.circle(img, to_px(path[-1]), 5, color, -1)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--replay", help="папка със записани кадри")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--headless", action="store_true")
    a = p.parse_args()

    proj = FloorProjector(a.camera)
    localizer = TagLocalizer(proj)
    print(f"Карта на маркерите: {localizer.path}")
    stamp_name = time.strftime("%Y%m%d_%H%M%S")
    record_dir = None
    if a.record and not a.replay:
        record_dir = c.DATA_DIR / "odom" / stamp_name
        record_dir.mkdir(parents=True)
        print(f"Записвам кадрите в {record_dir}")

    fusion = Fusion(proj, localizer)
    raw_path, fused_path, rows, errors, start = [], [], [], [], None
    source = replay_frames(a.replay) if a.replay else live_frames(a.camera)
    for stamp, frame in source:
        start = stamp if start is None else start
        if record_dir:
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                str(record_dir / f"{int(stamp * 1000):012d}.jpg"))
        fusion.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        raw, fused, tag = fusion.raw, fusion.fused, fusion.tag_pose
        if raw is not None:
            raw_path.append(tuple(raw[:2]))
            fused_path.append(tuple(fused[:2]))
        if tag is not None:
            errors.append((fusion.travel, float(np.linalg.norm(raw[:2] - tag[:2])), wrap_deg(raw[2] - tag[2])))
        fmt = lambda pose: ["", "", ""] if pose is None else [f"{pose[0]:.2f}", f"{pose[1]:.2f}", f"{wrap_deg(pose[2]):.2f}"]
        rows.append([f"{stamp - start:.3f}", f"{fusion.travel:.1f}", len(fusion.found), *fmt(raw), *fmt(tag), *fmt(fused)])

        if not a.headless:
            show = frame.copy()
            for marker_id, (px, _) in fusion.found.items():
                cv2.polylines(show, [px.astype(int)], True, (0, 255, 0), 3)
                cv2.putText(show, str(marker_id), px[0].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if fused is None:
                label(show, "show 2 markers to start")
            else:
                label(show, f"x {fused[0]:+.0f}  y {fused[1]:+.0f} cm  heading {wrap_deg(fused[2]):+.0f} deg")
                if tag is not None:
                    label(show, f"odometry error {errors[-1][1]:.1f} cm {errors[-1][2]:+.1f} deg", row=1)
                else:
                    label(show, f"no marker, {fusion.travel_since_tag:.0f} cm", row=1)
            cv2.imshow("localize", cv2.resize(show, (960, 540)))
            cv2.imshow("world", draw_map(localizer, raw_path, fused_path))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                fusion = Fusion(proj, localizer)
                raw_path, fused_path, rows, errors, start = [], [], [], [], None
                continue
        if a.seconds and stamp - start >= a.seconds:
            break
    source.close()
    cv2.destroyAllWindows()

    if rows:
        path = c.DATA_DIR / "metrics" / f"aruco_{stamp_name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow("time_s travel_cm n_markers odo_x odo_y odo_deg tag_x tag_y tag_deg fused_x fused_y fused_deg".split())
            w.writerows(rows)
        img = draw_map(localizer, raw_path, fused_path)
        cv2.imencode(".png", img)[1].tofile(str(path.with_suffix(".png")))
        print(f"Записано: {path} (и .png с траекториите)")

    # научените завъртания се пазят за следващите пъти
    for i, m in localizer.markers.items():
        localizer.data["markers"][str(i)] = m
    localizer.path.write_text(json.dumps(localizer.data, indent=2))

    if not errors:
        print("Няма позиция: камерата трябва да види поне два маркера едновременно (първия път).")
        return
    e = np.array(errors)
    print(f"Изминат път {e[-1, 0]:.0f} см, кадри с маркер: {len(e)}")
    print(f"Само одометрия срещу маркери: медиана {np.median(e[:, 1]):.1f} см, макс {e[:, 1].max():.1f} см, "
          f"в края {e[-1, 1]:.1f} см и {e[-1, 2]:+.1f}°")
    if localizer.side_ratios:
        off = (np.median(localizer.side_ratios) - 1) * 100
        print(f"Страна на маркерите по камера: {off:+.1f}% спрямо {localizer.size:g} см")
        if abs(off) > 3:
            print("ВНИМАНИЕ: камерата вероятно е мръднала след калибрацията на пода. Пусни пак "
                  "calibrate.py floor и не вярвай на сантиметрите дотогава.")
    if localizer.distances:
        print("Проверка на мащаба (разстояния между маркери):")
        print("\n".join(localizer.distance_report()))


if __name__ == "__main__":
    main()
