"""Стъпка 2: проверка на точността на измерване на пода.

  .venv/Scripts/python.exe src/measure.py --camera 0

Две проверки (камерата да НЕ е местена след calibrate.py floor):
  T      - премести дъската другаде на пода и натисни T: сравнява известните ѝ размери.
  клик   - два клика върху пода дават разстоянието в см (сравни с рулетка).
  R      - изчисти точките.   ESC - изход.
Резултатите от T се добавят в data/metrics/floor_accuracy.csv
"""
import argparse
import csv
import time

import cv2
import numpy as np

import common as c
from calibrate import label, load_lens

CSV_PATH = c.DATA_DIR / "metrics" / "floor_accuracy.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    K, dist = load_lens(args.camera)
    floor_path = c.CALIB_DIR / f"cam{args.camera}_floor.npz"
    if not floor_path.exists():
        raise SystemExit(f"Няма {floor_path}. Първо пусни: calibrate.py floor --camera {args.camera}")
    d = np.load(floor_path)
    H, square = d["H"], float(d["square_cm"])
    calib_center = c.board_points_cm(square).mean(0)

    clicks = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicks) == 2:
                clicks.clear()
            clicks.append((x, y))
            if len(clicks) == 2:
                a, b = c.pixels_to_floor(clicks, K, dist, H)
                print(f"Разстояние между точките: {np.linalg.norm(a - b):.1f} см")

    cap = c.open_camera(args.camera)
    cv2.namedWindow("measure")
    cv2.setMouseCallback("measure", on_mouse)
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = c.find_corners(gray, fast=True)
        # показваме суровия кадър; изкривяването се маха при преобразуването на точките
        show = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(show, c.PATTERN, corners.reshape(-1, 1, 2), True)
        for pt in clicks:
            cv2.circle(show, pt, 5, (0, 0, 255), -1)
        if len(clicks) == 2:
            a, b = c.pixels_to_floor(clicks, K, dist, H)
            cv2.line(show, clicks[0], clicks[1], (0, 0, 255), 2)
            label(show, f"{np.linalg.norm(a - b):.1f} cm", row=1)
        label(show, "T=test board  click 2 points  R=reset  ESC=quit")
        cv2.imshow("measure", show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            clicks.clear()
        if key in (ord("t"), ord("T")):
            corners = c.find_corners(gray)
            if corners is None:
                print("Дъската не се открива, опитай пак.")
                continue
            floor_pts = c.pixels_to_floor(corners, K, dist, H)
            offset = float(np.linalg.norm(floor_pts.mean(0) - calib_center))
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"Дъска на ~{offset:.0f} см от мястото на калибрацията:")
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            new_file = not CSV_PATH.exists()
            with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["time", "camera", "offset_cm", "what", "true_cm", "measured_cm", "error_pct"])
                for name, true, got in c.board_errors(floor_pts, square):
                    err = (got - true) / true * 100
                    print(f"  {name:9s} истина {true:5.2f} см, измерено {got:5.2f} см, грешка {err:+.1f}%")
                    w.writerow([stamp, args.camera, f"{offset:.1f}", name, f"{true:.3f}", f"{got:.3f}", f"{err:.2f}"])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
