"""Памет от минало минаване + честен изглед, в реално време (трите нишки на live.py).

  .venv/Scripts/python.exe src/live_memory.py              # прозорец, ESC = изход
  .venv/Scripts/python.exe src/live_memory.py --video      # записва каквото се вижда
  .venv/Scripts/python.exe src/live_memory.py --landmarks  # позиция по ориентири, не по маркери

Паметта (134157) се прави предварително, после записът 134822 върви с истинското си темпо.
Цветовете са като в memory_drive.py: пълен = видяно сега, приглушен = това минаване,
синкав = от паметта, червен = не съвпада, щриховка = мястото на предмета е предположено от модел,
тъмна мрежа = невиждано. R/F, T/G местят дрона.
"""
import argparse
import csv
import threading
import time

import cv2
import numpy as np

import common as c
import sim
from landmarks import LandmarkPose, Landmarks, build_map, marker_centres
from live import MAX_RENDER_FPS, Shared, track
from mapper import FloorMap
from memory_drive import (CONFLICT_FRAMES, LIVE_RUN, MEMORY_RUN, NEW_CELL_CM, build_memory, faded)
from static_map import STAMP_CM, STAMP_DEG, MapBuilder, replay_source, wrap_deg
from surfels import Surfels, draw_points, pack

RED = np.array((40, 40, 235), np.uint8)
NOW_VIEW = dict(width_cm=100.0, near_cm=40.0, far_cm=120.0, px_per_cm=4.0)  # като картата на пода


class Mapper:
    """Нишка 2: дълбочина, паметта за това минаване и проверка срещу старата памет."""

    def __init__(self, builder, observer, memory):
        import torch

        self.torch = torch
        self.builder, self.observer = builder, observer
        self.floor_memory, xyz, color = memory
        self.mem_xyz = xyz
        self.mem_color = faded(color, 0.6)
        self.mem_keys = pack(xyz, NEW_CELL_CM)
        shifts = [(dx * 4000 + dy) * 4000 + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        cells = np.unique(self.mem_keys)
        self.mem_around = np.unique(np.concatenate([cells + s for s in shifts]))
        self.mem_seen = np.isfinite(self.floor_memory.dist)
        self.misses = np.zeros(len(xyz), np.int16)
        self.new_cells = {}
        self.floor = FloorMap(builder.view_proj)
        self.floor.origin = self.floor_memory.origin.copy()
        self.drive = Surfels([observer.proj])
        self.last = None
        self.published = None
        self.parts_ms = {"depth": [], "memory": [], "check": [], "compose": []}

    def add(self, frame, pose):
        moved = self.last is None or np.hypot(*(pose[:2] - self.last[:2])) >= STAMP_CM \
            or abs(wrap_deg(pose[2] - self.last[2])) >= STAMP_DEG
        if not moved:
            return False
        self.last = pose.copy()
        t = [time.perf_counter()]
        seen = self.observer(frame, pose)
        t.append(time.perf_counter())
        self.floor.add(frame, pose, exclude=seen["above"])
        self.drive.observe(frame, seen["depth"], 0, pose)
        drive_xyz, drive_color = self.drive.cloud(every=8)
        t.append(time.perf_counter())

        through, agrees = self.observer.seen_through(self.mem_xyz, pose, seen["grid"])
        self.misses[through] += 1
        self.misses[agrees] = 0
        gone = self.misses >= CONFLICT_FRAMES
        now_xyz, now_color = seen["xyz"], seen["color"].copy()
        if len(now_xyz):
            cells = pack(now_xyz, NEW_CELL_CM)
            ij = self.floor_memory.world_to_map(now_xyz[:, :2]).astype(int)
            inside = ((ij >= 0) & (ij < self.mem_seen.shape[0])).all(1)
            over_floor = np.zeros(len(cells), bool)
            over_floor[inside] = self.mem_seen[ij[inside, 1], ij[inside, 0]]
            for cell in np.unique(cells[over_floor & ~np.isin(cells, self.mem_around)]):
                self.new_cells[cell] = self.new_cells.get(cell, 0) + 1
            confirmed = np.array([k for k, v in self.new_cells.items() if v >= CONFLICT_FRAMES], np.int64)
            now_color[np.isin(cells, confirmed)] = RED
        t.append(time.perf_counter())

        # един общ облак: видяното сега и в това минаване заменят паметта на същото място
        covered = np.unique(np.concatenate([pack(now_xyz, NEW_CELL_CM / 2), pack(drive_xyz, NEW_CELL_CM / 2)]))
        keep = ~np.isin(pack(self.mem_xyz, NEW_CELL_CM / 2), covered) | gone
        mem_color = self.mem_color[keep]
        mem_color[gone[keep]] = RED
        xyz = np.concatenate([now_xyz, drive_xyz, self.mem_xyz[keep]])
        color = np.concatenate([now_color, faded(drive_color, 0.18), mem_color])
        to_gpu = lambda a: self.torch.as_tensor(a, device="cuda")
        self.published = (to_gpu(xyz), to_gpu(color), faded(self.floor.image, 0.18),
                          np.isfinite(self.floor.dist).astype(np.uint8))
        t.append(time.perf_counter())
        for name, a, b in zip(self.parts_ms, t, t[1:]):
            self.parts_ms[name].append((b - a) * 1000)
        return True

    def run(self, shared, started):
        done = 0
        try:
            while not shared.done:
                with shared.lock:
                    item = shared.latest
                if item is None or item[0] == done:
                    time.sleep(0.005)
                    continue
                done, frame, _, pose, stamp = item
                t = time.perf_counter()
                if self.add(frame, pose):
                    shared.cloud = self.published
                    shared.map_ms.append((time.perf_counter() - t) * 1000)
                    shared.lag_s.append(time.perf_counter() - started - stamp)
        except Exception as error:
            shared.error = error
            raise


class LandmarkBuilder:
    """Позиция без маркери, в местата на MapBuilder за live.track: одометрия всеки кадър,
    а в отделна нишка поправка по ориентирите.

    truth = {време на кадъра: поза по маркерите}: само за да се скрият листовете с маркери
    от търсенето на ориентири и за да се измери грешката. Позицията не ги ползва.
    """

    PERIOD_S = 0.33  # най-много три поправки в секунда

    def __init__(self, projector, landmarks, centres, truth):
        self.ours, self.landmarks = LandmarkPose(projector), landmarks
        self.centres, self.truth = centres, truth
        self.fusion = self.map_pose = self
        self.lock = threading.Lock()
        self.stamp, self.current = None, None
        self.errors, self.locate_ms = [], []
        self.attempts = self.fixes = 0
        self.first_fix_s = None

    def update(self, what, rear=None):
        """Вика се два пъти от live.track: с кадъра (одометрия) и със себе си (позата)."""
        if what is not self:
            self.ours.move(what)
            with self.lock:
                self.current = (self.stamp, what, self.ours.odom.pose.copy())
            return None
        with self.lock:
            pose = self.ours.pose()
        truth = self.truth.get(self.stamp)
        if pose is not None and truth is not None:
            self.errors.append((float(np.hypot(*(pose[:2] - truth[:2]))),
                                abs(wrap_deg(pose[2] - truth[2]))))
        return pose

    def run(self, shared, started):
        done = None
        try:
            while not shared.done:
                with self.lock:
                    item = self.current
                if item is None or item[0] == done or self.truth.get(item[0]) is None:
                    time.sleep(0.005)
                    continue
                done, gray, odom_pose = item
                t = time.perf_counter()
                fix, _ = self.landmarks.locate(gray, (self.truth[done], self.centres))
                self.locate_ms.append((time.perf_counter() - t) * 1000)
                self.attempts += 1
                if fix is not None:
                    self.fixes += 1
                    with self.lock:
                        self.ours.correct(fix, odom_pose)
                    if self.first_fix_s is None:
                        self.first_fix_s = time.perf_counter() - started
                time.sleep(max(0.0, self.PERIOD_S - (time.perf_counter() - t)))
        except Exception as error:
            shared.error = error
            raise


class Screen:
    """Главната нишка: сглобява изгледа от готовото."""

    def __init__(self, mapper, now_proj, age_min):
        self.mapper, self.now_proj, self.age = mapper, now_proj, age_min
        floor = mapper.floor_memory
        self.s, self.origin = floor.s, floor.origin
        self.memory_image = faded(floor.image, 0.55)
        self.memory_mask = mapper.mem_seen.astype(np.uint8)
        self.to_world = np.array([[1 / self.s, 0, self.origin[0] + 0.5 / self.s],
                                  [0, -1 / self.s, self.origin[1] - 0.5 / self.s], [0, 0, 1.0]])
        w, h = sim.VIEW_W, sim.VIEW_H
        yy, xx = np.indices((h, w))
        self.hatch = (xx + yy) % 9 < 2
        self.legend = np.zeros((h - 180, sim.SIDE_W, 3), np.uint8)
        rows = (((255, 255, 255), "seen now: full colour"), ((200, 200, 200), "seen this drive: dim"),
                ((215, 160, 110), f"memory, {age_min:.0f} min old: pale blue"), ((40, 40, 235), "changed / conflicts"),
                ((150, 150, 150), "hatched: position guessed by AI"), ((90, 90, 90), "dark grid: never seen"))
        for i, (color, text) in enumerate(rows):
            y = 35 + 28 * i
            cv2.rectangle(self.legend, (10, y - 12), (26, y + 2), color, -1)
            cv2.putText(self.legend, text, (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    def floor_now(self, view, to_img, frame, pose):
        """Подът, който камерата вижда в този момент, направо от кадъра (без да чака картата)."""
        p = self.now_proj
        top = p.warp(frame)
        s = p.px_per_cm
        cos, sin = np.cos(pose[2]), np.sin(pose[2])
        local = np.array([[1 / s, 0, -p.width_cm / 2 + 0.5 / s], [0, -1 / s, p.far_cm - 0.5 / s], [0, 0, 1.0]])
        world = np.array([[cos, -sin, pose[0]], [sin, cos, pose[1]], [0, 0, 1.0]])
        H = to_img @ world @ local
        size = (sim.VIEW_W, sim.VIEW_H)
        color = cv2.warpPerspective(top, H, size)
        mask = cv2.warpPerspective(cv2.erode(p.mask.astype(np.uint8), np.ones((3, 3), np.uint8)), H, size,
                                   flags=cv2.INTER_NEAREST)
        cv2.copyTo(color, mask, view)

    def draw(self, cam, frame, pose, published):
        R, C, K = cam.matrices()
        to_img = K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])
        size = (sim.VIEW_W, sim.VIEW_H)
        view = np.full((sim.VIEW_H, sim.VIEW_W, 3), 28, np.uint8)
        sim.draw_grid(view, cam)
        layers = [(self.memory_image, self.memory_mask)]
        if published is not None:
            layers.append((published[2], published[3]))
        for image, mask in layers:
            H = to_img @ self.to_world
            color = cv2.warpPerspective(image, H, size)
            cv2.copyTo(color, cv2.warpPerspective(mask, H, size, flags=cv2.INTER_NEAREST), view)
        self.floor_now(view, to_img, frame, pose)

        if published is not None:
            xyz, color = published[0], published[1]
            truck = float((np.array([*pose[:2], 8.0]) - C) @ R[2])
            far = view.copy()
            drawn = draw_points(far, (R, C, K), sim.FOCAL, xyz, color, near_cm=truck)
            near = view.copy()
            hide_near = draw_points(near, (R, C, K), sim.FOCAL, xyz, color, near_cm=0.55 * truck, far_cm=truck - 15)
            for layer, mask in ((far, drawn), (near, hide_near)):
                lines = mask & self.hatch
                layer[lines] = layer[lines] // 3 * 2
            view[drawn] = far[drawn]
            sim.draw_boxes(view, cam, sim.truck_boxes(pose))
            view[hide_near] = near[hide_near]
        else:
            sim.draw_boxes(view, cam, sim.truck_boxes(pose))
        return view


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory", default=MEMORY_RUN)
    p.add_argument("--drive", default=LIVE_RUN)
    p.add_argument("--video", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--landmarks", action="store_true")
    a = p.parse_args()

    from datetime import datetime

    from projection import FloorProjector

    when = lambda name: datetime.strptime(name, "%Y%m%d_%H%M%S")
    age_min = (when(a.drive) - when(a.memory)).total_seconds() / 60
    print("Правя паметта от миналото минаване...")
    floor, xyz, color, observer = build_memory(c.DATA_DIR / "odom" / a.memory)
    builder = MapBuilder()
    mapper = Mapper(builder, observer, (floor, xyz, color))
    screen_maker = Screen(mapper, FloorProjector(0, **NOW_VIEW), age_min)
    frames = list(replay_source(c.DATA_DIR / "odom" / a.drive, False))
    length = frames[-1][0] - frames[0][0]
    tracker = builder
    if a.landmarks:
        print("Правя картата от ориентири...")
        points, desc, owner = build_map(c.DATA_DIR / "odom" / a.memory)
        referee = MapBuilder()  # еталон по маркерите, сметнат предварително
        truth = {stamp: referee.process(image, None, stamp) for stamp, image, _ in frames}
        proj = referee.fusion.odom.proj
        tracker = LandmarkBuilder(FloorProjector(0), Landmarks(points, desc, owner, proj),
                                  marker_centres(referee.localizer), truth)
        print(f"{len(points)} ориентира")
    observer.model(frames[0][1])  # загряване
    warm = np.zeros((4, 3), np.float32)
    draw_points(np.zeros((sim.VIEW_H, sim.VIEW_W, 3), np.uint8),
                (np.eye(3), np.array((0.0, 0.0, -50.0)), np.array([[700, 0, 480], [0, 700, 270], [0, 0, 1.0]])),
                700, warm, warm.astype(np.uint8))

    shared = Shared()
    cam = sim.ChaseCamera(85.0, 90.0)
    name = time.strftime("%Y%m%d_%H%M%S")
    out = c.DATA_DIR / "map"
    video, written = None, 0
    if a.video:
        video = cv2.VideoWriter(str(out / f"{name}_live_memory.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12,
                                (sim.VIEW_W + sim.SIDE_W, sim.VIEW_H))
    started = time.perf_counter()
    threads = [threading.Thread(target=track, args=(shared, tracker, frames, started), daemon=True),
               threading.Thread(target=mapper.run, args=(shared, started), daemon=True)]
    if a.landmarks:
        threads.append(threading.Thread(target=tracker.run, args=(shared, started), daemon=True))
    for t in threads:
        t.start()

    render_ms, shown_at, truck_pose, snapshots = [], [], None, 0
    while not shared.done and shared.error is None:
        t = time.perf_counter()
        with shared.lock:
            item = shared.latest
        if item is None:
            time.sleep(0.01)
            continue
        _, frame, _, pose, _ = item
        step = min(1.0, 4.0 * (t - shown_at[-1])) if shown_at else 1.0
        cam.follow(pose, step)
        truck_pose = pose.copy() if truck_pose is None else truck_pose + min(1.0, 3 * step) * (pose - truck_pose)
        view = screen_maker.draw(cam, frame, truck_pose, shared.cloud)
        screen = np.zeros((sim.VIEW_H, sim.VIEW_W + sim.SIDE_W, 3), np.uint8)
        screen[:, :sim.VIEW_W] = view
        screen[:180, sim.VIEW_W:] = cv2.resize(frame, (sim.SIDE_W, 180))
        screen[180:, sim.VIEW_W:] = screen_maker.legend
        render_ms.append((time.perf_counter() - t) * 1000)
        shown_at.append(t)
        recent = shown_at[-15:]
        fps = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)
        cv2.putText(screen, f"view {fps:4.1f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if video is not None:
            while written < (t - started) * 12:
                video.write(screen)
                written += 1
        if t - started > (snapshots + 1) * length / 4 and snapshots < 3:
            snapshots += 1
            cv2.imencode(".png", screen)[1].tofile(str(out / f"{name}_live_memory_{snapshots}.png"))
        if not a.headless:
            cv2.imshow("live memory", screen)
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
        t.join(timeout=15)
    cv2.destroyAllWindows()
    if video is not None:
        video.release()
    if shared.error:
        raise SystemExit(f"Грешка в нишката на картата: {shared.error}")

    median = lambda values: round(float(np.median(values))) if values else 0
    result = {
        "recording_s": round(length, 1), "run_s": round(elapsed, 1),
        "view_fps": round(len(shown_at) / elapsed, 1), "view_ms": median(render_ms),
        "worst_view_gap_ms": round(float(np.percentile(np.diff(shown_at), 95)) * 1000),
        "pose_fps": round(shared.fed / elapsed, 1), "frames_skipped": shared.skipped,
        "map_updates_per_s": round(len(shared.map_ms) / elapsed, 2), "map_ms": median(shared.map_ms),
        "map_lag_s": round(float(np.median(shared.lag_s)), 2),
        **{f"map_{k}_ms": median(v) for k, v in mapper.parts_ms.items()},
        "memory_points": len(mapper.mem_xyz),
        "memory_points_flagged": int((mapper.misses >= CONFLICT_FRAMES).sum()),
    }
    result["position"] = "landmarks" if a.landmarks else "markers"
    if a.landmarks:
        errors = np.array(tracker.errors)
        result.update({
            "first_fix_s": round(tracker.first_fix_s or -1, 2), "fixes": f"{tracker.fixes}/{tracker.attempts}",
            "locate_ms": median(tracker.locate_ms),
            "pose_err_cm_median": round(float(np.median(errors[:, 0])), 1),
            "pose_err_cm_max": round(float(errors[:, 0].max()), 1),
            "pose_err_deg_median": round(float(np.median(errors[:, 1])), 1),
        })
    for key, value in result.items():
        print(f"  {key:22s} {value}")
    path = c.DATA_DIR / "metrics" / f"live_memory_{name}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(result.keys())
        w.writerow(result.values())
    print(f"Записано: {path}" + (f" и {out / f'{name}_live_memory.mp4'}" if a.video else ""))


if __name__ == "__main__":
    main()
