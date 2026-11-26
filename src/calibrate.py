"""Стъпка 2: калибрация на камера с шахматна дъска.

1) Леща (веднъж за камерата; дъската се държи в ръка пред камерата):
     .venv/Scripts/python.exe src/calibrate.py lens --camera 0
2) Под (камерата в крайното си положение, дъската легнала на пода):
     .venv/Scripts/python.exe src/calibrate.py floor --camera 0
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import common as c

TARGET_VIEWS = 20
MIN_VIEWS = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, row=0):
    org = (15, 35 + 35 * row)
    cv2.putText(img, text, org, FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


TILT_DEG = 25          # от толкова нагоре кадърът се брои за наклонен
MAX_FLAT = 8           # най-много толкова почти фронтални кадъра
MIN_BOARD_PX = 250     # дъската да не е прекалено далече


def rough_tilt(corners):
    """Приблизителен наклон на дъската спрямо камерата (градуси), с номинална леща."""
    K = np.array([[900.0, 0, c.SIZE[0] / 2], [0, 900.0, c.SIZE[1] / 2], [0, 0, 1]])
    obj = np.c_[c.board_points_cm(), np.zeros(len(corners))]
    _, rvec, _ = cv2.solvePnP(obj, corners.astype(np.float64), K, None)
    return float(np.degrees(np.arccos(min(1.0, abs(cv2.Rodrigues(rvec)[0][2, 2])))))


def lens(args):
    shots_dir = Path(args.images) if args.images else capture_lens_views(args)
    solve_lens(args, shots_dir)


def capture_lens_views(args):
    shots_dir = c.DATA_DIR / "calib" / f"cam{args.camera}" / time.strftime("%Y%m%d_%H%M%S")
    shots_dir.mkdir(parents=True)
    cap = c.open_camera(args.camera)
    views, previous, last_shot, flat = [], None, 0.0, 0
    print("Дръж дъската БЛИЗО пред камерата (30-60 см). Мести я до краищата и ъглите на кадъра")
    print(f"и я НАКЛАНЯЙ силно (30-45°) настрани, напред и назад: поне {TARGET_VIEWS - MAX_FLAT} кадъра наклонени.")
    print("Кадър се взима сам, щом я задържиш неподвижно.")
    print(f"ENTER = готово (поне {MIN_VIEWS} кадъра), ESC = отказ.")

    while len(views) < TARGET_VIEWS:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = c.find_corners(gray, fast=True)
        show = frame.copy()
        for view in views:  # покрити зони
            for x, y in view:
                cv2.circle(show, (int(x), int(y)), 2, (0, 160, 0), -1)

        hint = "board not found"
        if corners is not None:
            cv2.drawChessboardCorners(show, c.PATTERN, corners.reshape(-1, 1, 2), True)
            still = previous is not None and np.abs(corners - previous).mean() < 1.5
            new = all(np.linalg.norm(corners - v, axis=1).mean() > 60 for v in views)
            tilt = rough_tilt(corners)
            close = np.ptp(corners, axis=0).max() >= MIN_BOARD_PX
            tilted = tilt >= TILT_DEG
            if not close:
                hint = "move CLOSER"
            elif not tilted and flat >= MAX_FLAT:
                hint = f"TILT more ({tilt:.0f} deg)"
            else:
                hint = f"tilt {tilt:.0f} deg - hold still" if new else "move to a new place"
            if still and new and close and (tilted or flat < MAX_FLAT) and time.time() - last_shot > 1.0:
                flat += not tilted
                views.append(corners)
                last_shot = time.time()
                cv2.imencode(".png", frame)[1].tofile(str(shots_dir / f"{len(views):02d}.png"))
                print(f"  кадър {len(views)}/{TARGET_VIEWS}")
        previous = corners

        if time.time() - last_shot < 0.25:  # примигване при снимка
            show = cv2.add(show, (80, 80, 80, 0))
        label(show, f"{len(views)}/{TARGET_VIEWS}  tilted {len(views) - flat}   ENTER=done  ESC=quit")
        label(show, hint, row=1)
        cv2.imshow("lens calibration", show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise SystemExit("Отказано, нищо не е записано.")
        if key == 13 and len(views) >= MIN_VIEWS:
            break

    cap.release()
    cv2.destroyAllWindows()
    return shots_dir


def solve_lens(args, shots_dir):
    # live откриването е бързо, но неточно: смятаме с точния детектор върху записаните кадри
    print(f"Смятане от {shots_dir} ...")
    images = [cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_GRAYSCALE)
              for f in sorted(shots_dir.glob("*.png"))]
    views = [v for v in (c.find_corners(img) for img in images) if v is not None]
    if len(views) < MIN_VIEWS:
        raise SystemExit(f"Дъската се открива само в {len(views)} кадъра, трябват поне {MIN_VIEWS}.")
    rms, K, dist, per_view = c.calibrate_intrinsics(views, args.square)
    # махаме явно лошите кадри (размазани и т.н.) и смятаме пак
    keep = [v for v, e in zip(views, per_view) if e < max(1.0, 2 * np.median(per_view))]
    if MIN_VIEWS <= len(keep) < len(views):
        print(f"Отхвърлени лоши кадри: {len(views) - len(keep)}")
        views = keep
        rms, K, dist, per_view = c.calibrate_intrinsics(views, args.square)

    c.CALIB_DIR.mkdir(exist_ok=True)
    path = c.CALIB_DIR / f"cam{args.camera}_lens.npz"
    np.savez(path, K=K, dist=dist, rms=rms, size=c.SIZE, square_cm=args.square)

    all_pts = np.concatenate(views)
    span = (all_pts.max(0) - all_pts.min(0)) / np.array(c.SIZE) * 100
    hfov = np.degrees(2 * np.arctan(c.SIZE[0] / (2 * K[0, 0])))
    print(f"RMS грешка: {rms:.3f} px  (добре: под 0.5, приемливо: под 1.0)")
    print(f"Покритие на кадъра: {span[0]:.0f}% ширина, {span[1]:.0f}% височина (целим над 80%)")
    print(f"Хоризонтален ъгъл на камерата: {hfov:.0f}°")
    print(f"Записано: {path}")
    if rms > 1.0:
        print("ВНИМАНИЕ: грешката е твърде голяма, повтори калибрацията. Най-често дъската не е")
        print("напълно плоска (залепи я върху твърда основа) или кадрите са размазани.")


def load_lens(camera):
    path = c.CALIB_DIR / f"cam{camera}_lens.npz"
    if not path.exists():
        raise SystemExit(f"Няма {path}. Първо пусни: calibrate.py lens --camera {camera}")
    d = np.load(path)
    return d["K"], d["dist"]


def floor(args):
    K, dist = load_lens(args.camera)
    cap = c.open_camera(args.camera)
    print("Сложи дъската плоско на пода, близо пред камерата. Камерата да е в крайното си положение.")
    print("SPACE = запис (когато ъглите са очертани), ESC = отказ.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = c.find_corners(gray, fast=True)
        show = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(show, c.PATTERN, corners.reshape(-1, 1, 2), True)
        label(show, "board flat on floor   SPACE=save  ESC=quit")
        cv2.imshow("floor calibration", show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise SystemExit("Отказано.")
        if key == 32:
            corners = c.find_corners(gray)
            if corners is not None:
                break
            print("Дъската не се открива добре, опитай пак.")

    cap.release()
    cv2.destroyAllWindows()
    H = c.floor_homography(corners, K, dist, args.square)
    path = c.CALIB_DIR / f"cam{args.camera}_floor.npz"
    np.savez(path, H=H, corners=corners, square_cm=args.square)
    shot = c.DATA_DIR / "calib" / f"cam{args.camera}_floor.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", frame)[1].tofile(str(shot))

    fit = np.linalg.norm(c.pixels_to_floor(corners, K, dist, H) - c.board_points_cm(args.square), axis=1)
    print(f"Остатъчна грешка върху същата дъска: средно {fit.mean() * 10:.1f} mm (това НЕ е проверка на точността)")
    print(f"Записано: {path}")
    print(f"Сега провери точността: measure.py --camera {args.camera}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["lens", "floor"])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--square", type=float, default=c.SQUARE_CM, help="измерена страна на квадрата, см")
    p.add_argument("--images", help="lens: смята наново от папка със записани кадри, без камера")
    a = p.parse_args()
    {"lens": lens, "floor": floor}[a.mode](a)
