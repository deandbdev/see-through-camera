"""Опит 1 + 2: "Виждам зад ъгъла" с памет от минало минаване и изглед, който си признава.

  .venv/Scripts/python.exe src/memory_drive.py                # памет от 134157, каране 134822

Без генеративен модел: нищо не се дорисува. Какво значат цветовете:
  пълен цвят        - камерата го вижда в момента;
  леко приглушено   - видяно в това минаване, но сега е извън кадъра;
  синкаво и бледо   - само от паметта (минаното пътуване);
  червено           - паметта и видяното не съвпадат (преместено, изчезнало или ново);
  щриховка          - предмет: мястото му е ПРЕДПОЛОЖЕНО от модел за дълбочина (не е мерено);
  тъмна мрежа       - никога невиждано.
Позицията идва от маркерите (и двете минавания са в една и съща координатна система).
"""
import argparse
import csv
import time
from datetime import datetime

import cv2
import numpy as np

import common as c
import sim
from depth3d import MAX_HEIGHT_CM, MIN_HEIGHT_CM, CameraRays, DepthModel
from mapper import FloorMap
from motion import rot
from static_map import STAMP_CM, STAMP_DEG, MapBuilder, replay_source, wrap_deg
from surfels import Surfels, draw_points, pack

MEMORY_RUN, LIVE_RUN = "20261125_134157", "20261125_134822"
THROUGH_SHARE, THROUGH_CM, CONFLICT_FRAMES = 0.08, 3.0, 3
NEW_CELL_CM = 6.0
MEMORY_TINT = np.array((215, 160, 110), np.float32)  # BGR: синкаво


class Observer:
    """Какво вижда един кадър: маска над пода, точките на предметите, дълбочината по пиксели."""

    def __init__(self, proj):
        self.proj = proj
        self.model = DepthModel()
        self.coarse, self.fine = CameraRays(proj), CameraRays(proj, 2)
        self.rng = np.random.default_rng(0)

    def __call__(self, frame, pose):
        depth = self.model(frame)
        out = dict(depth=depth, above=np.full(frame.shape[:2], 255, np.uint8), grid=None,
                   xyz=np.zeros((0, 3), np.float32), color=np.zeros((0, 3), np.uint8))
        cam = self.coarse
        ab = cam.fit(depth[cam.v, cam.u].ravel(), self.rng)
        if ab is None:
            return out
        _, height, z = cam.points(depth[cam.v, cam.u].ravel(), ab)
        mask = ((cam.usable & (height > MIN_HEIGHT_CM)).reshape(cam.shape) * 255).astype(np.uint8)
        out["above"] = cv2.dilate(cv2.resize(mask, c.SIZE, interpolation=cv2.INTER_NEAREST),
                                  np.ones((9, 9), np.uint8))
        out["grid"] = z.reshape(cam.shape)
        cam = self.fine
        local, height, z = cam.points(depth[cam.v, cam.u].ravel(), ab)
        grid = z.reshape(cam.shape).astype(np.float32)
        jump = cv2.morphologyEx(grid, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) / grid
        keep = cam.usable & (height > MIN_HEIGHT_CM) & (height < MAX_HEIGHT_CM) & (z < 150) & (jump.ravel() < 0.06)
        world = local[keep] @ rot(pose[2]).T + pose[:2]
        out["xyz"] = np.c_[world, height[keep]].astype(np.float32)
        out["color"] = frame[cam.v, cam.u].reshape(-1, 3)[keep]
        return out

    def seen_through(self, xyz, pose, grid):
        """Кои от точките xyz са в кадъра и камерата вижда ясно ЗАД тях; кои се потвърждават."""
        through, agrees = np.zeros(len(xyz), bool), np.zeros(len(xyz), bool)
        if grid is None or not len(xyz):
            return through, agrees
        proj, cam = self.proj, self.coarse
        board = ((xyz[:, :2] - pose[:2]) @ rot(pose[2])) @ proj.axes + proj.origin
        up = np.sign((-proj.R.T @ proj.t)[2])
        p = np.c_[board, up * xyz[:, 2]] @ proj.R.T + proj.t
        depth = p[:, 2]
        near = np.flatnonzero((depth > 15) & (depth < 150) & (np.abs(p[:, 0]) < 0.7 * depth) &
                              (np.abs(p[:, 1]) < 0.42 * depth))
        if not len(near):
            return through, agrees
        px = cv2.projectPoints(p[near], np.zeros(3), np.zeros(3), proj.K, proj.dist)[0].reshape(-1, 2)
        col, row = (px[:, 0] // 4).astype(int), (px[:, 1] // 4).astype(int)
        on = (col >= 2) & (col < cam.shape[1] - 2) & (row >= 2) & (row < cam.shape[0] - 2)
        near, col, row = near[on], col[on], row[on]
        # най-близкото в малък прозорец: по ръбовете да не вдигаме фалшива тревога
        nearest = cv2.erode(grid.astype(np.float32), np.ones((5, 5), np.uint8))[row, col]
        seen = grid[row, col]
        usable = cam.usable.reshape(cam.shape)[row, col]
        through[near] = usable & (nearest > depth[near] * (1 + THROUGH_SHARE) + THROUGH_CM)
        agrees[near] = usable & (np.abs(seen - depth[near]) < depth[near] * THROUGH_SHARE + THROUGH_CM)
        return through, agrees


def stamp_now(floor_map, frame, pose, above):
    """Само този кадър, върху платно с координатите на floor_map: (изображение, маска)."""
    one = FloorMap.__new__(FloorMap)
    one.__dict__.update(floor_map.__dict__)
    one.image = np.zeros_like(floor_map.image)
    one.dist = np.full(floor_map.dist.shape, np.inf, np.float32)
    one.add(frame, pose, exclude=above)
    return one.image, np.isfinite(one.dist)


def build_memory(folder):
    builder = MapBuilder()
    observer = Observer(builder.fusion.odom.proj)
    floor, points, last = None, Surfels([observer.proj]), None
    for stamp, frame, _ in replay_source(folder, False):
        pose = builder.process(frame, None, stamp)
        if pose is None:
            continue
        if last is not None and np.hypot(*(pose[:2] - last[:2])) < STAMP_CM and \
                abs(wrap_deg(pose[2] - last[2])) < STAMP_DEG:
            continue
        last = pose.copy()
        seen = observer(frame, pose)
        if floor is None:
            floor = FloorMap(builder.view_proj, center=tuple(pose[:2]))
        floor.add(frame, pose, exclude=seen["above"])
        points.observe(frame, seen["depth"], 0, pose)
        points.cloud(every=8)
    xyz, color = points.cloud()
    return floor, xyz, color, observer


class Shown:
    """Каквото sim.render_floor иска от една карта."""

    def __init__(self, like, image, mask):
        self.s, self.origin, self.image = like.s, like.origin, image
        self.dist = np.where(mask, 0.0, np.inf).astype(np.float32)


def faded(image, strength):
    return (image.astype(np.float32) * (1 - strength) + MEMORY_TINT * strength).astype(np.uint8)


def draw_layer(view, matrices, xyz, color, alpha, hatch=True):
    """Предметите от едно състояние: с прозрачност и щриховка (мястото им е предположено)."""
    if not len(xyz):
        return
    layer = view.copy()
    mask = draw_points(layer, matrices, sim.FOCAL, xyz, color)
    if hatch:
        yy, xx = np.indices(mask.shape)
        lines = mask & ((xx + yy) % 9 < 2)
        layer[lines] = (layer[lines] * 0.65).astype(np.uint8)
    view[mask] = (layer[mask] * alpha + view[mask] * (1 - alpha)).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory", default=MEMORY_RUN)
    p.add_argument("--drive", default=LIVE_RUN)
    a = p.parse_args()

    when = lambda name: datetime.strptime(name, "%Y%m%d_%H%M%S")
    age_min = (when(a.drive) - when(a.memory)).total_seconds() / 60
    memory_floor, mem_xyz, mem_color, observer = build_memory(c.DATA_DIR / "odom" / a.memory)
    mem_seen = np.isfinite(memory_floor.dist)
    mem_tinted = faded(memory_floor.image, 0.55)
    mem_faded_color = faded(mem_color, 0.6)
    print(f"памет: {mem_seen.sum() / memory_floor.s ** 2 / 1e4:.2f} м² под, {len(mem_xyz)} точки, "
          f"на {age_min:.1f} минути")

    builder = MapBuilder()
    drive_floor = FloorMap(builder.view_proj)
    drive_floor.origin = memory_floor.origin.copy()
    drive = Surfels([observer.proj])
    misses = np.zeros(len(mem_xyz), np.int16)
    mem_cells = np.unique(pack(mem_xyz, NEW_CELL_CM))
    shifts = [(dx * 4000 + dy) * 4000 + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    mem_around = np.unique(np.concatenate([mem_cells + s for s in shifts]))
    new_cells = {}

    out = c.DATA_DIR / "map"
    stamp_name = time.strftime("%Y%m%d_%H%M%S")
    video = cv2.VideoWriter(str(out / f"{stamp_name}_memory.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12,
                            (sim.VIEW_W + sim.SIDE_W, sim.VIEW_H))
    camera = sim.ChaseCamera(85.0, 90.0)
    ahead_only, colour_gap, point_gap, rows = [], [], [], []
    mem_checked = np.zeros(len(mem_xyz), bool)
    n = 0
    px_per_cm2 = memory_floor.s ** 2
    for stamp, frame, _ in replay_source(c.DATA_DIR / "odom" / a.drive, False):
        pose = builder.process(frame, None, stamp)
        if pose is None:
            continue
        n += 1
        seen = observer(frame, pose)
        now_image, now_mask = stamp_now(drive_floor, frame, pose, seen["above"])
        drive_floor.add(frame, pose, exclude=seen["above"])
        drive.observe(frame, seen["depth"], 0, pose)
        drive_seen = np.isfinite(drive_floor.dist)

        # вярна ли е паметта: там, където камерата гледа сега
        both = now_mask[::3, ::3] & mem_seen[::3, ::3]
        if both.sum() > 200:
            now_px = now_image[::3, ::3][both].astype(np.float32)
            mem_px = memory_floor.image[::3, ::3][both].astype(np.float32)
            gain = now_px.mean(0) / np.maximum(mem_px.mean(0), 1)  # без разликата в експозицията
            colour_gap.append(float(np.abs(now_px - mem_px * gain).mean()))
        through, agrees = observer.seen_through(mem_xyz, pose, seen["grid"])
        misses[through] += 1
        misses[agrees] = 0
        mem_checked |= through | agrees
        gone = misses >= CONFLICT_FRAMES
        if len(seen["xyz"]) and n % 10 == 0 and len(mem_xyz):
            sample = seen["xyz"][::max(1, len(seen["xyz"]) // 400)]
            pool = mem_xyz[::max(1, len(mem_xyz) // 20000)]
            near = np.sqrt(((sample[:, None] - pool[None]) ** 2).sum(-1).min(1))
            point_gap.extend(near.tolist())

        # ново: предмет сега там, където паметта е видяла под и няма нищо наоколо
        new_now = np.zeros(len(seen["xyz"]), bool)
        if len(seen["xyz"]):
            cells = pack(seen["xyz"], NEW_CELL_CM)
            ij = memory_floor.world_to_map(seen["xyz"][:, :2]).astype(int)
            inside = ((ij >= 0) & (ij < mem_seen.shape[0])).all(1)
            over_floor = np.zeros(len(cells), bool)
            over_floor[inside] = mem_seen[ij[inside, 1], ij[inside, 0]]
            candidate = over_floor & ~np.isin(cells, mem_around)
            for cell in np.unique(cells[candidate]):
                new_cells[cell] = new_cells.get(cell, 0) + 1
            confirmed = np.array([k for k, v in new_cells.items() if v >= CONFLICT_FRAMES], np.int64)
            new_now = np.isin(cells, confirmed)

        only_memory = mem_seen & ~drive_seen
        ahead_only.append(float(only_memory.sum() / px_per_cm2 / 1e4))

        # картината: от най-слабото доверие към най-силното
        floor_image = np.zeros_like(memory_floor.image)
        floor_image[mem_seen] = mem_tinted[mem_seen]
        floor_image[drive_seen] = faded(drive_floor.image[drive_seen], 0.18)
        floor_image[now_mask] = now_image[now_mask]
        camera.follow(pose, 0.15 if n > 1 else 1.0)
        view = sim.render_floor(Shown(memory_floor, floor_image, mem_seen | drive_seen | now_mask), camera)
        matrices = camera.matrices()
        draw_layer(view, matrices, mem_xyz[~gone], mem_faded_color[~gone], 0.6)
        old_xyz, old_color = drive.cloud(every=8)
        draw_layer(view, matrices, old_xyz, faded(old_color, 0.18), 0.9)
        draw_layer(view, matrices, seen["xyz"][~new_now], seen["color"][~new_now], 1.0)
        red = np.tile(np.array((40, 40, 235), np.uint8), (int(gone.sum()), 1))
        draw_layer(view, matrices, mem_xyz[gone], red, 0.55, hatch=False)
        red = np.tile(np.array((40, 40, 235), np.uint8), (int(new_now.sum()), 1))
        draw_layer(view, matrices, seen["xyz"][new_now], red, 0.75, hatch=False)
        sim.draw_boxes(view, camera, sim.truck_boxes(pose))

        screen = np.zeros((sim.VIEW_H, sim.VIEW_W + sim.SIDE_W, 3), np.uint8)
        screen[:, :sim.VIEW_W] = view
        screen[:180, sim.VIEW_W:] = cv2.resize(frame, (sim.SIDE_W, 180))
        legend = (((255, 255, 255), "seen now: full colour"), ((200, 200, 200), "seen this drive: slightly dim"),
                  ((215, 160, 110), f"memory, {age_min:.0f} min old: pale blue"), ((40, 40, 235), "changed / conflicts"),
                  ((150, 150, 150), "hatched: position guessed by AI"), ((90, 90, 90), "dark grid: never seen"))
        for i, (color, text) in enumerate(legend):
            y = 215 + 28 * i
            cv2.rectangle(screen, (sim.VIEW_W + 10, y - 12), (sim.VIEW_W + 26, y + 2), color, -1)
            cv2.putText(screen, text, (sim.VIEW_W + 34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
                        cv2.LINE_AA)
        cv2.putText(screen, f"shown only from memory: {ahead_only[-1]:.2f} m2", (sim.VIEW_W + 10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 190, 150), 1, cv2.LINE_AA)
        video.write(screen)
        if n in (1, 80, 220, 380):
            cv2.imencode(".png", screen)[1].tofile(str(out / f"{stamp_name}_memory_{n}.png"))
    video.release()

    changed = int((misses >= CONFLICT_FRAMES).sum())
    result = {
        "memory_run": a.memory, "drive": a.drive, "age_min": round(age_min, 1),
        "memory_floor_m2": round(float(mem_seen.sum() / px_per_cm2 / 1e4), 2),
        "only_from_memory_start_m2": round(ahead_only[0], 2), "only_from_memory_end_m2": round(ahead_only[-1], 2),
        "floor_colour_gap_median": round(float(np.median(colour_gap)), 1),
        "object_distance_median_cm": round(float(np.median(point_gap)), 1),
        "object_distance_p90_cm": round(float(np.percentile(point_gap, 90)), 1),
        "memory_points_checked": int(mem_checked.sum()),
        "memory_points_flagged_gone": changed,
        "flagged_share_of_checked_percent": round(100 * changed / max(1, int(mem_checked.sum())), 1),
        "new_cells": sum(v >= CONFLICT_FRAMES for v in new_cells.values()),
    }
    for key, value in result.items():
        print(f"  {key:34s} {value}")
    path = c.DATA_DIR / "metrics" / f"memory_{stamp_name}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(result.keys())
        w.writerow(result.values())
    print(f"Записано: {out / f'{stamp_name}_memory.mp4'} и {path}")


if __name__ == "__main__":
    main()
