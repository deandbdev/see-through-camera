"""Стъпка 4: две камери, синхронизация и обща карта отгоре.

Първо (веднъж) запиши къде е задната камера спрямо предната (измерено с рулетка):
  .venv/Scripts/python.exe src/dual.py --back 40 --right 0
После само:
  .venv/Scripts/python.exe src/dual.py

G = мрежа, S = запиши кадри в data/dual/, ESC = изход.
"""
import argparse
import json
import time

import cv2
import numpy as np

import common as c
from calibrate import label
from capture import CameraThread, synced_pair
from projection import FloorProjector

RIG_PATH = c.CALIB_DIR / "rig.json"


def load_rig(args):
    rig = json.loads(RIG_PATH.read_text()) if RIG_PATH.exists() else {
        "front_camera": 0, "rear_camera": 1, "back_cm": None, "right_cm": 0.0, "yaw_deg": 180.0}
    given = {"front_camera": args.front, "rear_camera": args.rear, "back_cm": args.back,
             "right_cm": args.right, "yaw_deg": args.yaw}
    rig.update({k: v for k, v in given.items() if v is not None})
    if rig["back_cm"] is None:
        raise SystemExit("Не знам къде е задната камера. Пусни с --back <см назад> [--right <см надясно>].")
    c.CALIB_DIR.mkdir(exist_ok=True)
    RIG_PATH.write_text(json.dumps(rig, indent=2))
    return rig


def compose(front, top_front, top_rear):
    """Обща карта; при припокриване предната камера е с предимство."""
    out = top_rear.copy()
    out[front.mask] = top_front[front.mask]
    return out


def draw_cameras(img, proj, rig):
    s = proj.px_per_cm
    for x, y, name in ((0, 0, "front"), (rig["right_cm"], -rig["back_cm"], "rear")):
        pt = (int((x + proj.width_cm / 2) * s), int((proj.far_cm - y) * s))
        cv2.circle(img, pt, 6, (0, 200, 255), -1)
        cv2.putText(img, name, (pt[0] + 10, pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--front", type=int, help="индекс на предната камера")
    p.add_argument("--rear", type=int, help="индекс на задната камера")
    p.add_argument("--back", type=float, help="задната камера е толкова см зад предната")
    p.add_argument("--right", type=float, help="... и толкова см надясно (отрицателно = наляво)")
    p.add_argument("--yaw", type=float, help="посока на задната камера, градуси (180 = право назад)")
    p.add_argument("--width", type=float, default=140, help="ширина на картата, см")
    p.add_argument("--range", type=float, default=150, help="докъде напред и назад, см")
    p.add_argument("--scale", type=float, default=3, help="пиксели на см")
    p.add_argument("--seconds", type=float, default=0, help="спри сам след толкова секунди (за тест)")
    a = p.parse_args()
    rig = load_rig(a)

    view = dict(width_cm=a.width, px_per_cm=a.scale, far_cm=a.range, near_cm=-(a.range + rig["back_cm"]))
    front = FloorProjector(rig["front_camera"], **view)
    rear = FloorProjector(rig["rear_camera"], pos_cm=(rig["right_cm"], -rig["back_cm"]),
                          yaw_deg=rig["yaw_deg"], **view)
    for name, proj in (("Предна", front), ("Задна", rear)):
        print(f"{name} камера: {proj.height_cm:.1f} см над пода, наклон {proj.pitch_deg:.0f}° надолу.")

    cams = [CameraThread(rig["front_camera"]), CameraThread(rig["rear_camera"])]
    grid, shown, deltas, last = True, 0, [], None
    start = time.perf_counter()
    try:
        while not (a.seconds and time.perf_counter() - start >= a.seconds):
            pair = synced_pair(*cams)
            if pair is None or (last is not None and pair[0] is last[0] and pair[1] is last[1]):
                if cv2.waitKey(2) & 0xFF == 27:
                    break
                continue
            last = pair
            frame_f, frame_r, delta = pair
            deltas.append(delta * 1000)
            top = compose(front, front.warp(frame_f), rear.warp(frame_r))
            show = top.copy()
            if grid:
                front.draw_grid(show, step_cm=20)
            draw_cameras(show, front, rig)
            shown += 1
            label(show, f"{shown / (time.perf_counter() - start):.0f} FPS  dt {delta * 1000:.0f} ms")
            # смаляваме само прозореца, ако картата не се побира на екрана
            k = min(1.0, 900 / show.shape[0])
            cv2.imshow("map", cv2.resize(show, None, fx=k, fy=k) if k < 1 else show)
            cv2.imshow("cameras", np.hstack([cv2.resize(f, (480, 270)) for f in (frame_f, frame_r)]))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("g"), ord("G")):
                grid = not grid
            if key in (ord("s"), ord("S")):
                out = c.DATA_DIR / "dual"
                out.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for name, img in (("front", frame_f), ("rear", frame_r), ("map", top)):
                    cv2.imencode(".png", img)[1].tofile(str(out / f"{stamp}_{name}.png"))
                print(f"Записано в {out}")
    finally:
        elapsed = time.perf_counter() - start
        for cam in cams:
            cam.close()
        cv2.destroyAllWindows()
    if deltas:
        print(f"Средно {shown / elapsed:.1f} FPS; разлика между камерите: "
              f"медиана {np.median(deltas):.0f} ms, 95% под {np.percentile(deltas, 95):.0f} ms")


if __name__ == "__main__":
    main()
