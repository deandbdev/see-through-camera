"""Стъпка 5: одометрия с оптичен поток, на живо или от запис.

  .venv/Scripts/python.exe src/odometry.py              # на живо
  .venv/Scripts/python.exe src/odometry.py --record     # и записва кадрите в data/odom/
  .venv/Scripts/python.exe src/odometry.py --replay data/odom/<папка>

R = нулирай,  SPACE = запиши текущия резултат в data/metrics/odometry.csv,  ESC = изход.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import common as c
from calibrate import label
from capture import CameraThread
from motion import Odometry
from projection import FloorProjector

CSV_PATH = c.DATA_DIR / "metrics" / "odometry.csv"


def live_frames(camera):
    cam, last = CameraThread(camera), None
    try:
        while True:
            frames = cam.snapshot()
            if not frames or frames[-1][0] == last:
                time.sleep(0.003)
                continue
            last, frame = frames[-1]
            yield last, frame
    finally:
        cam.close()


def replay_frames(folder):
    for path in sorted(Path(folder).glob("*.jpg")):
        yield int(path.stem) / 1000, cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def draw_path(path, size=500):
    """Траекторията отгоре; мащабът се нагажда сам."""
    img = np.zeros((size, size, 3), np.uint8)
    pts = np.array(path)
    half = max(50.0, np.abs(pts).max() * 1.15)
    to_px = lambda p: (int(size / 2 + p[0] / half * size / 2), int(size / 2 - p[1] / half * size / 2))
    step = 10 if half <= 100 else 50
    for g in np.arange(-(half // step) * step, half, step):
        cv2.line(img, to_px((g, -half)), to_px((g, half)), (45, 45, 45), 1)
        cv2.line(img, to_px((-half, g)), to_px((half, g)), (45, 45, 45), 1)
    cv2.polylines(img, [np.array([to_px(p) for p in pts])], False, (0, 200, 255), 2)
    cv2.circle(img, to_px((0, 0)), 4, (0, 255, 0), -1)
    cv2.circle(img, to_px(pts[-1]), 5, (0, 0, 255), -1)
    cv2.putText(img, f"grid {step} cm", (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--replay", help="папка със записани кадри")
    p.add_argument("--seconds", type=float, default=0, help="спри сам след толкова секунди (за тест)")
    p.add_argument("--headless", action="store_true", help="без прозорци (за тест и replay)")
    a = p.parse_args()

    odom = Odometry(FloorProjector(a.camera))
    record_dir = None
    if a.record and not a.replay:
        record_dir = c.DATA_DIR / "odom" / time.strftime("%Y%m%d_%H%M%S")
        record_dir.mkdir(parents=True)
        print(f"Записвам кадрите в {record_dir}")

    source = replay_frames(a.replay) if a.replay else live_frames(a.camera)
    path, travelled, frames, start, update_ms = [(0.0, 0.0)], 0.0, 0, None, []
    for stamp, frame in source:
        start = stamp if start is None else start
        if record_dir:
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                str(record_dir / f"{int(stamp * 1000):012d}.jpg"))
        t0 = time.perf_counter()
        ok = odom.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        update_ms.append((time.perf_counter() - t0) * 1000)
        frames += 1
        x, y, theta = odom.pose
        # пътят се трупа на стъпки от поне 0.5 см, за да не брои шума
        step = float(np.hypot(x - path[-1][0], y - path[-1][1]))
        if step >= 0.5:
            travelled += step
            path.append((x, y))

        done = a.seconds and stamp - start >= a.seconds
        if not a.headless:
            show = frame.copy()
            for (u, v), good in zip(odom.points, odom.inliers):
                cv2.circle(show, (int(u), int(v)), 3, (0, 255, 0) if good else (0, 0, 255), -1)
            label(show, f"x {x:+.1f}  y {y:+.1f} cm   turn {np.degrees(theta):+.1f} deg")
            label(show, f"path {travelled:.1f} cm   points {int(odom.inliers.sum())}"
                        f"{'' if ok else '   NO ESTIMATE'}   lost {odom.lost}", row=1)
            cv2.imshow("odometry", cv2.resize(show, (960, 540)))
            cv2.imshow("path", draw_path(path))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                odom.reset()
                path, travelled = [(0.0, 0.0)], 0.0
            if key == 32:
                CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                new_file = not CSV_PATH.exists()
                with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(["time", "x_cm", "y_cm", "turn_deg", "path_cm", "lost_frames"])
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{x:.1f}", f"{y:.1f}",
                                f"{np.degrees(theta):.1f}", f"{travelled:.1f}", odom.lost])
                print(f"Записано: x={x:+.1f} y={y:+.1f} см, завой {np.degrees(theta):+.1f}°, път {travelled:.1f} см")
        if done:
            break

    source.close()
    cv2.destroyAllWindows()
    if frames:
        x, y, theta = odom.pose
        print(f"{frames} кадъра, {np.median(update_ms):.1f} ms на кадър (медиана), без оценка: {odom.lost}")
        print(f"Крайна поза: x={x:+.1f} y={y:+.1f} см, завой {np.degrees(theta):+.1f}°, път {travelled:.1f} см")


if __name__ == "__main__":
    main()
