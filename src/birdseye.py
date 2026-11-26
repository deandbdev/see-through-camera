"""Стъпка 3: изглед отгоре на пода, на живо от една камера.

  .venv/Scripts/python.exe src/birdseye.py --camera 0

G = мрежа (10 см), S = запиши кадър в data/birdseye/, ESC = изход.
"""
import argparse
import time

import cv2

import common as c
from calibrate import label
from projection import FloorProjector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=float, default=120, help="ширина на изгледа, см")
    p.add_argument("--far", type=float, default=150, help="докъде напред, см")
    p.add_argument("--scale", type=float, default=5, help="пиксели на см")
    p.add_argument("--seconds", type=float, default=0, help="спри сам след толкова секунди (за тест)")
    a = p.parse_args()

    proj = FloorProjector(a.camera, a.width, a.far, a.scale)
    print(f"Камера на {proj.height_cm:.1f} см над пода, наклон {proj.pitch_deg:.0f}° надолу.")
    print("Сравни височината с линийка. G = мрежа, S = запис, ESC = изход.")

    cap = c.open_camera(a.camera)
    grid, frames, fps, warp_ms = True, 0, 0.0, 0.0
    start = mark = time.perf_counter()
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        top = proj.warp(frame)
        warp_ms = 0.9 * warp_ms + 0.1 * (time.perf_counter() - t0) * 1000

        show = top.copy()
        if grid:
            proj.draw_grid(show)
        label(show, f"{fps:.0f} FPS")
        cv2.imshow("bird's-eye", show)
        cv2.imshow("camera", cv2.resize(frame, (640, 360)))

        frames += 1
        total += 1
        now = time.perf_counter()
        if now - mark >= 1:
            fps, frames, mark = frames / (now - mark), 0, now

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or (a.seconds and now - start >= a.seconds):
            break
        if key in (ord("g"), ord("G")):
            grid = not grid
        if key in (ord("s"), ord("S")):
            out = c.DATA_DIR / "birdseye"
            out.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for name, img in (("raw", frame), ("top", top)):
                cv2.imencode(".png", img)[1].tofile(str(out / f"{stamp}_{name}.png"))
            print(f"Записано в {out}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Средно {total / (time.perf_counter() - start):.1f} FPS, проекцията отнема {warp_ms:.1f} ms на кадър.")


if __name__ == "__main__":
    main()
