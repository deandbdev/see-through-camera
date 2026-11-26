"""Дрон изглед "на живо": позицията, картата и рисуването вървят в отделни нишки.

  .venv/Scripts/python.exe src/live.py data/odom/<папка>              # прозорец, ESC = изход
  .venv/Scripts/python.exe src/live.py data/odom/<папка> --video      # записва каквото се вижда

Записът се подава с истинското си темпо, все едно са камери: ако нещо не смогва, кадрите се
изпускат, а не се чакат.
  нишка 1: позиция (оптичен поток + маркери) за всеки кадър, който смогне;
  нишка 2: дълбочина, точки и под - винаги от най-новия кадър, колкото успее;
  главна:  рисува дрона от наличното, без да чака другите две.
Накрая: колко кадъра в секунда е постигнала всяка част. R/F, T/G местят дрона.
"""
import argparse
import csv
import threading
import time

import cv2
import numpy as np

import common as c
import sim
from depth3d import MAX_HEIGHT_CM, MIN_HEIGHT_CM, CameraRays, DepthModel
from motion import rot
from static_map import (REAR_PENALTY_CM, STAMP_CM, STAMP_DEG, MapBuilder, compose, marker_errors, rear_offset,
                        replay_source, wrap_deg)
from surfels import Surfels, draw_points, draw_smooth

MAX_RENDER_FPS = 15.0


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None      # (номер, преден, заден, поза, време на кадъра) от нишка 1
        self.cloud = None       # (точки, цветове) на видеокартата от нишка 2
        self.done = False
        self.error = None
        self.pose_ms, self.map_ms, self.lag_s = [], [], []
        self.fed = self.skipped = 0


def track(shared, builder, source, started):
    """Нишка 1: позиция в темпото на записа. Закъснял кадър се изпуска."""
    first = None
    items = iter(source)
    item = next(items, None)
    while item is not None:
        stamp, frame, rear = item
        first = stamp if first is None else first
        wait = (stamp - first) - (time.perf_counter() - started)
        if wait > 0:
            time.sleep(wait)
        following = next(items, None)
        # ако следващият кадър вече е "пристигнал", този е остарял
        if following is not None and following[0] - first <= time.perf_counter() - started:
            shared.skipped += 1
            item = following
            continue
        t = time.perf_counter()
        builder.stamp = stamp  # за онзи, който иска да знае кой кадър е това
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        builder.fusion.update(gray, None if rear is None else cv2.cvtColor(rear, cv2.COLOR_BGR2GRAY))
        pose = builder.map_pose.update(builder.fusion)
        shared.pose_ms.append((time.perf_counter() - t) * 1000)
        shared.fed += 1
        if pose is not None:
            with shared.lock:
                shared.latest = (shared.fed, frame, rear, pose.copy(), stamp - first)
        item = following
    shared.done = True


class Mapper:
    """Нишка 2: от най-новия кадър прави дълбочина, точки и под."""

    def __init__(self, builder, smooth=True):
        self.builder, self.smooth = builder, smooth
        self.model = DepthModel()
        self.surfels = Surfels(builder.odom_projs)
        self.cameras = [None if p is None else CameraRays(p) for p in builder.odom_projs]
        self.rng = np.random.default_rng(0)
        self.last = None

    def above_floor(self, depth, camera):
        """Маска на пикселите над пода (или всичко, ако не знаем мащаба)."""
        cam = self.cameras[camera]
        d = depth[cam.v, cam.u].ravel()
        ab = cam.fit(d, self.rng)
        if ab is None:
            return np.full(depth.shape, 255, np.uint8)
        _, height, _ = cam.points(d, ab)
        mask = ((cam.usable & (height > MIN_HEIGHT_CM)).reshape(cam.shape) * 255).astype(np.uint8)
        return cv2.dilate(cv2.resize(mask, c.SIZE, interpolation=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8))

    def add(self, frame, rear, pose):
        import torch

        b = self.builder
        moved = self.last is None or np.hypot(*(pose[:2] - self.last[:2])) >= STAMP_CM \
            or abs(wrap_deg(pose[2] - self.last[2])) >= STAMP_DEG
        if not moved:
            return False
        if b.floor_map is None:
            from mapper import FloorMap
            b.floor_map = FloorMap(b.view_proj, center=tuple(pose[:2]))
            b.floor_map.balance = self.smooth
        cameras = [(frame, 0, pose)]
        if rear is not None:
            cameras.append((rear, 1, compose(pose, rear_offset(b.rig))))
        for image, index, where in cameras:
            depth = self.model(image)
            above = self.above_floor(depth, index)
            self.surfels.observe(image, depth, index, where)
            if index == 0:
                if getattr(b.fusion.odom, "source", "front") == "front":
                    b.floor_map.add(image, where, exclude=above)
            else:
                b.floor_map.add(image, where, b.rear_proj, REAR_PENALTY_CM, above)
        self.last = pose
        b.path.append(tuple(pose[:2]))
        xyz, color = self.surfels.cloud()
        self.published = (torch.as_tensor(xyz, device="cuda"), torch.as_tensor(color, device="cuda"))
        return True

    def run(self, shared, started):
        seen = 0
        try:
            while not shared.done:
                with shared.lock:
                    item = shared.latest
                if item is None or item[0] == seen:
                    time.sleep(0.005)
                    continue
                seen, frame, rear, pose, stamp = item
                t = time.perf_counter()
                if self.add(frame, rear, pose):
                    shared.cloud = self.published
                    shared.map_ms.append((time.perf_counter() - t) * 1000)
                    shared.lag_s.append(time.perf_counter() - started - stamp)
        except Exception as error:  # да не умре тихо
            shared.error = error
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--video", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--raw", action="store_true", help="без изглаждане, за сравнение")
    a = p.parse_args()

    from pathlib import Path
    use_rear = (Path(a.folder) / "rear").exists()
    builder = MapBuilder(use_rear=use_rear)
    mapper = Mapper(builder, not a.raw)
    draw = draw_points if a.raw else draw_smooth
    frames = list(replay_source(a.folder, use_rear))  # четенето от диска не се брои
    length = frames[-1][0] - frames[0][0]
    mapper.model(frames[0][1])  # загряване, преди да тръгне часовникът
    cam = sim.ChaseCamera()
    plain = np.zeros((4, 4, 3), np.float32)
    draw_points(np.zeros((sim.VIEW_H, sim.VIEW_W, 3), np.uint8),
                (np.eye(3), np.array((0.0, 0.0, -50.0)), np.array([[700, 0, 480], [0, 700, 270], [0, 0, 1.0]])),
                700, plain, plain.astype(np.uint8))

    shared = Shared()
    video, written = None, 0
    name = time.strftime("%Y%m%d_%H%M%S")
    out = c.DATA_DIR / "map"
    if a.video:
        out.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(out / f"{name}_live.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12,
                                (sim.VIEW_W + sim.SIDE_W, sim.VIEW_H))
    started = time.perf_counter()
    threads = [threading.Thread(target=track, args=(shared, builder, frames, started), daemon=True),
               threading.Thread(target=mapper.run, args=(shared, started), daemon=True)]
    for t in threads:
        t.start()

    render_ms, shown_at, truck_pose = [], [], None
    while not shared.done and shared.error is None:
        t = time.perf_counter()
        with shared.lock:
            item = shared.latest
        if item is None:
            time.sleep(0.01)
            continue
        _, front, rear, pose, _ = item
        # дронът догонва плавно, независимо колко често идват позициите
        step = min(1.0, 4.0 * (t - shown_at[-1])) if shown_at else 1.0
        cam.follow(pose, step)
        # и камиончето се плъзга плавно между измерените позиции
        truck_pose = pose.copy() if truck_pose is None or a.raw else truck_pose + min(1.0, 3 * step) * (pose - truck_pose)
        pose = truck_pose
        view = sim.render_floor(builder.floor_map, cam)
        cloud = shared.cloud
        if cloud is not None:
            matrices = cam.matrices()
            truck = float((np.array([*pose[:2], 8.0]) - matrices[1]) @ matrices[0][2])
            draw(view, matrices, sim.FOCAL, *cloud, near_cm=truck)
            sim.draw_boxes(view, cam, sim.truck_boxes(pose))
            draw(view, matrices, sim.FOCAL, *cloud, near_cm=0.55 * truck, far_cm=truck - 15)
        else:
            sim.draw_boxes(view, cam, sim.truck_boxes(pose))
        screen = np.zeros((sim.VIEW_H, sim.VIEW_W + sim.SIDE_W, 3), np.uint8)
        screen[:, :sim.VIEW_W] = view
        for row, image in enumerate((front, rear)):
            if image is not None:
                screen[row * 180:(row + 1) * 180, sim.VIEW_W:] = cv2.resize(image, (sim.SIDE_W, 180))
        render_ms.append((time.perf_counter() - t) * 1000)
        shown_at.append(t)
        fps = (len(shown_at[-15:]) - 1) / max(shown_at[-1] - shown_at[-15:][0], 1e-6)
        cv2.putText(screen, f"view {fps:4.1f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if video is not None:  # видеото върви в реално време: повтаряме кадъра колкото е стоял
            while written < (t - started) * 12:
                video.write(screen)
                written += 1
        if not a.headless:
            cv2.imshow("live", screen)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            letter = chr(key).lower()
            if letter in "rf":
                cam.height = float(np.clip(cam.height + (10 if letter == "r" else -10), 25, 250))
            elif letter in "tg":
                cam.back = float(np.clip(cam.back + (10 if letter == "t" else -10), 10, 200))
        time.sleep(max(0.0, 1.0 / MAX_RENDER_FPS - (time.perf_counter() - t)))
    elapsed = time.perf_counter() - started
    shared.done = True
    for t in threads:
        t.join(timeout=10)
    cv2.destroyAllWindows()
    if video is not None:
        video.release()
    if shared.error:
        raise SystemExit(f"Грешка в нишката на картата: {shared.error}")

    gaps = np.diff(shown_at)
    result = {
        "recording_s": round(length, 1), "run_s": round(elapsed, 1),
        "view_fps": round(len(shown_at) / elapsed, 1), "view_ms": round(float(np.median(render_ms))),
        "worst_view_gap_ms": round(float(np.percentile(gaps, 95)) * 1000),
        "pose_fps": round(shared.fed / elapsed, 1), "pose_ms": round(float(np.median(shared.pose_ms))),
        "frames_skipped": shared.skipped,
        "map_updates_per_s": round(len(shared.map_ms) / elapsed, 2), "map_ms": round(float(np.median(shared.map_ms))),
        "map_lag_s": round(float(np.median(shared.lag_s)), 2),
        "points": 0 if shared.cloud is None else len(shared.cloud[0]),
    }
    errors = marker_errors(builder.floor_map, builder.localizer)
    result["marker_errors_cm"] = " ".join(f"{i}:{e:.1f}" for i, e in sorted(errors.items()))
    for key, value in result.items():
        print(f"  {key:20s} {value}")
    path = c.DATA_DIR / "metrics" / f"live_{name}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(result.keys())
        w.writerow(result.values())
    cv2.imencode(".png", builder.floor_map.cropped()[0])[1].tofile(str(path.with_suffix(".png")))
    print(f"Записано: {path}" + (f" и {out / f'{name}_live.mp4'}" if a.video else ""))


if __name__ == "__main__":
    main()
