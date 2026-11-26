"""Предметите с истински пиксели вместо кубчета: гъст облак от цветни точки, рисуван на GPU.

  .venv/Scripts/python.exe src/surfels.py data/odom/20261125_135744

Дълбочината (depth3d.py) дава мястото на всеки пиксел над пода. Пазим ги в клетки по CELL_CM,
като цветът е от най-близкия поглед (не се осреднява, за да не се размазва). Виртуалната камера
ги проектира със z-буфер. Невиденото остава празно.

Проверка: част от кадрите не участват; рисуваме тяхната гледна точка от останалите и я
сравняваме с истинския кадър (само пиксели над пода) - спрямо кубчетата на depth3d.py.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import common as c
import depth3d as d3
import sim
from motion import rot
from static_map import MapBuilder, compose, rear_offset, replay_source

CELL_CM = 0.6
STEP = 2                  # през толкова пиксела от кадъра
HOLD = 6                  # на блокове по толкова ключови кадъра; всеки трети блок е скрит
CARVE_SHARE, CARVE_CM = 0.08, 3.0  # колко по-далече трябва да се вижда, за да е "през" точката
CARVE_MISSES = 3          # толкова кадъра да са гледали през нея, преди да я трием


def pack(xyz, size):
    """Номер на клетката с дадения размер, в която попада всяка точка."""
    cell = np.floor(xyz / size).astype(np.int64) + 2000
    return (cell[:, 0] * 4000 + cell[:, 1]) * 4000 + cell[:, 2]


class Surfels:
    """Облакът расте в движение: новите кадри чакат в parts и се сливат на порции."""

    def __init__(self, projectors, carve=False):
        self.carve = carve  # трие точки, през чието място камерата вижда по-надалеч
        self.cameras = [None if p is None else d3.CameraRays(p, STEP) for p in projectors]
        self.rng = np.random.default_rng(0)
        self.clear()

    def clear(self):
        self.parts = []
        empty = np.zeros(0, np.int64)
        # по една точка на клетка: (ключ, разстояние на погледа, точка, цвят)
        self.state = (empty, np.zeros(0, np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
        self.coarse = (empty, empty)  # (клетка 3 cm, в колко кадъра е виждана)
        self.misses = np.zeros(0, np.int16)  # на всяка точка от state: колко пъти е "прозирана"
        self.carved = 0
        self._cloud = None

    def observe(self, frame, depth, camera, pose, moving=None):
        """moving: маска на хора и играчки - не влизат в неподвижния облак."""
        cam = self.cameras[camera]
        d = depth[cam.v, cam.u].ravel()
        ab = cam.fit(d, self.rng)
        if ab is None:
            return
        local, height, z = cam.points(d, ab)
        grid = z.reshape(cam.shape).astype(np.float32)
        jump = cv2.morphologyEx(grid, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) / grid
        if self.carve:
            self._carve(cam, grid, jump, pose)
        keep = (cam.usable & (height > d3.MIN_HEIGHT_CM) & (height < d3.MAX_HEIGHT_CM) &
                (z < d3.MAX_RANGE_CM) & (jump.ravel() < d3.EDGE))
        if moving is not None:
            keep &= moving[cam.v, cam.u].ravel() == 0
        world = local[keep] @ rot(pose[2]).T + pose[:2]
        xyz = np.c_[world, height[keep]].astype(np.float32)
        self.parts.append((pack(xyz, CELL_CM), z[keep].astype(np.float32), xyz, frame[cam.v, cam.u].reshape(-1, 3)[keep]))

    def _carve(self, cam, grid, jump, pose):
        """Старите точки, зад чието място новият кадър ясно вижда нещо друго, получават "грешка"."""
        if self.parts:
            self._merge()
        xyz = self.state[2]
        if not len(xyz):
            return
        proj = cam.proj
        local = (xyz[:, :2] - pose[:2]) @ rot(pose[2])
        board = local @ proj.axes + proj.origin
        up = np.sign((-proj.R.T @ proj.t)[2])
        in_cam = np.c_[board, up * xyz[:, 2]] @ proj.R.T + proj.t
        depth = in_cam[:, 2]
        near = np.flatnonzero((depth > 10) & (depth < d3.MAX_RANGE_CM) &
                              (np.abs(in_cam[:, 0]) < 0.8 * depth) & (np.abs(in_cam[:, 1]) < 0.5 * depth))
        if not len(near):
            return
        px = cv2.projectPoints(in_cam[near], np.zeros(3), np.zeros(3), proj.K, proj.dist)[0].reshape(-1, 2)
        col, row = (px[:, 0] // STEP).astype(int), (px[:, 1] // STEP).astype(int)
        on = (col >= 1) & (col < cam.shape[1] - 1) & (row >= 1) & (row < cam.shape[0] - 1)
        near, col, row = near[on], col[on], row[on]
        usable = cam.usable.reshape(cam.shape)[row, col] & (jump[row, col] < d3.EDGE)
        seen = grid[row, col]
        through = usable & (seen > depth[near] * (1 + CARVE_SHARE) + CARVE_CM)
        agrees = usable & (np.abs(seen - depth[near]) < depth[near] * CARVE_SHARE + CARVE_CM)
        self.misses[near[through]] += 1
        self.misses[near[agrees]] = np.maximum(self.misses[near[agrees]] - 1, 0)
        keep = self.misses < CARVE_MISSES
        if not keep.all():
            self.carved += int((~keep).sum())
            self.state = tuple(column[keep] for column in self.state)
            self.misses = self.misses[keep]
            self._cloud = None

    def _merge(self):
        views = [np.unique(pack(part[2], 3.0)) for part in self.parts]
        keys, counts = self.coarse
        keys = np.concatenate([keys, *views])
        counts = np.concatenate([counts, *(np.ones(len(v), np.int64) for v in views)])
        keys, inverse = np.unique(keys, return_inverse=True)
        self.coarse = keys, np.bincount(inverse, counts).astype(np.int64)

        stacked = [np.concatenate([self.state[i], *(p[i] for p in self.parts)]) for i in range(4)]
        added = sum(len(p[0]) for p in self.parts)
        misses = np.concatenate([self.misses, np.zeros(added, np.int16)])
        order = np.lexsort((stacked[1], stacked[0]))  # по клетка, после най-близкият поглед
        first = order[np.r_[True, np.diff(stacked[0][order]) != 0]]
        self.state = tuple(column[first] for column in stacked)
        self.misses = misses[first]
        self.parts = []

    def cloud(self, every=1):
        """(точки, цветове). every: слива новите кадри чак щом се натрупат толкова."""
        if self.parts and (len(self.parts) >= every or self._cloud is None):
            self._merge()
            self._cloud = None
        if self._cloud is None:
            _, _, xyz, color = self.state
            # клетка, около която (в 3 cm) се е виждало само от един-два кадъра, най-вероятно е шум
            keys, counts = self.coarse
            good = counts[np.searchsorted(keys, pack(xyz, 3.0))] >= d3.MIN_VIEWS if len(xyz) else slice(None)
            self._cloud = xyz[good], color[good]
        return self._cloud


def draw_points(img, matrices, focal, xyz, colors, point_cm=CELL_CM * 1.6, near_cm=5.0, far_cm=1e9):
    """Точките върху img със z-буфер на GPU; всяка покрива квадратче според разстоянието си."""
    import torch

    R, C, K = matrices
    device = "cuda"
    h, w = img.shape[:2]
    if not len(xyz):
        return np.zeros((h, w), bool)
    p = (torch.as_tensor(xyz, device=device) - torch.as_tensor(C, device=device, dtype=torch.float32))
    p = p @ torch.as_tensor(R.T, device=device, dtype=torch.float32)
    z = p[:, 2]
    front = (z > near_cm) & (z <= far_cm)
    u = p[:, 0] / z * float(K[0, 0]) + float(K[0, 2])
    v = p[:, 1] / z * float(K[1, 1]) + float(K[1, 2])
    front &= (u > -4) & (u < w + 4) & (v > -4) & (v < h + 4)  # извън кадъра не се смята
    ids = front.nonzero().squeeze(1)
    if not len(ids):
        return np.zeros((h, w), bool)
    u, v, z = u[ids].round().long(), v[ids].round().long(), z[ids]
    radius = (focal * point_cm / z / 2).ceil().clamp(0, 4).long()
    # един ключ на точка: (разстояние, номер); най-малкият в пиксела печели
    key = ((z * 100).long() << 24) + torch.arange(len(z), device=device)
    empty = torch.iinfo(torch.int64).max
    buffer = torch.full((h * w,), empty, dtype=torch.int64, device=device)
    for r in range(5):
        sel = (radius == r).nonzero().squeeze(1)
        if not len(sel):
            continue
        offs = torch.arange(-r, r + 1, device=device)
        uu = (u[sel][:, None] + offs.repeat(len(offs))[None]).reshape(-1)
        vv = (v[sel][:, None] + offs.repeat_interleave(len(offs))[None]).reshape(-1)
        on = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        buffer.scatter_reduce_(0, (vv * w + uu).where(on, torch.zeros_like(uu)),
                               key[sel].repeat_interleave(len(offs) ** 2).where(on, torch.full_like(uu, empty)),
                               "amin")
    hit = buffer != empty
    winner = ids[buffer[hit] & 0xFFFFFF]
    colors = torch.as_tensor(colors, device=device)
    packed = torch.cat([hit.nonzero(), colors[winner].long()], 1).cpu().numpy()  # едно прехвърляне
    img.reshape(-1, 3)[packed[:, 0]] = packed[:, 1:].astype(img.dtype)
    mask = np.zeros(h * w, bool)
    mask[packed[:, 0]] = True
    return mask.reshape(h, w)


def draw_smooth(img, matrices, focal, xyz, colors, min_blob=150, **kwargs):
    """draw_points с евтино изглаждане: запушени дупки, без разпръснати петна, меки ръбове.

    Нищо ново не се измисля освен цвета на дупки от няколко пиксела.
    """
    layer = np.zeros_like(img)
    mask = draw_points(layer, matrices, focal, xyz, colors, **kwargs).astype(np.uint8)
    if not mask.any():
        return
    x, y, w, h = cv2.boundingRect(mask)
    x0, y0 = max(x - 6, 0), max(y - 6, 0)
    box = (slice(y0, y + h + 6), slice(x0, x + w + 6))
    layer, mask = layer[box], mask[box]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_blob
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    layer[mask == 0] = 0
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    spread = cv2.dilate(layer, np.ones((5, 5), np.uint8))  # цвят за дупките и за ръба
    filled = np.where(mask[..., None] > 0, layer, spread)
    filled = cv2.medianBlur(filled, 5)
    alpha = cv2.GaussianBlur(closed.astype(np.float32), (5, 5), 0)[..., None]
    region = img[box]
    region[:] = (filled * alpha + region * (1 - alpha)).astype(np.uint8)


def camera_matrices(proj, pose, scale):
    """Истинската камера като виртуална (без изкривяване), за сравнение с кадъра ѝ."""
    up = np.sign((-proj.R.T @ proj.t)[2])
    to_board = np.zeros((3, 3))
    to_board[:2, :2], to_board[2, 2] = proj.axes.T, up
    turn = np.eye(3)
    turn[:2, :2] = rot(-pose[2])
    R = proj.R @ to_board @ turn
    offset = proj.R @ np.array([*proj.origin, 0.0]) + proj.t - R @ np.array([*pose[:2], 0.0])
    C = -np.linalg.solve(R, offset)
    K = proj.K.copy()
    K[:2] *= scale
    return R, C, K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    a = p.parse_args()

    use_rear = (Path(a.folder) / "rear").exists()
    builder = MapBuilder(use_rear=use_rear, obstacles=True)
    keyframes = []
    original = builder.obstacles.observe

    def record(frame, camera, pose, moving=None):
        keyframes.append((frame.copy(), camera, pose.copy()))
        return original(frame, camera, pose, moving)

    builder.obstacles.observe = record
    car = None
    for stamp, frame, rear in replay_source(a.folder, use_rear):
        pose = builder.process(frame, rear, stamp)
        car = pose if pose is not None else car
    floor = builder.floor_map

    # два облака от едни и същи кадри: кубчета и точки; скритите не участват
    cubes = builder.obstacles
    cubes.clear()
    points = Surfels(builder.odom_projs)
    hidden = []
    counts = [0, 0]
    depth_ms = []
    for frame, camera, pose in keyframes:
        k = counts[camera]
        counts[camera] += 1
        if (k // HOLD) % 3 == 2:
            if k % HOLD == HOLD // 2:
                hidden.append((frame, camera, pose))
            continue
        t = time.perf_counter()
        original(frame, camera, pose)
        depth_ms.append((time.perf_counter() - t) * 1000)
        points.observe(frame, cubes.model(frame), camera, pose)
    voxel_xyz, voxel_color = cubes.voxels()
    xyz, color = points.cloud()
    print(f"ключови кадри {len(keyframes)}, скрити за проверка {len(hidden)}; "
          f"кубчета {len(voxel_xyz)}, точки {len(xyz)}")

    # проверка върху скритите кадри
    scale = 0.5
    size = (int(c.SIZE[0] * scale), int(c.SIZE[1] * scale))
    scores = {"cubes": [[], []], "points": [[], []]}  # (покритие, грешка)
    examples = []
    for n, (frame, camera, pose) in enumerate(hidden):
        proj = builder.odom_projs[camera]
        cam = points.cameras[camera]
        d = cubes.model(frame)
        sample = d[cam.v, cam.u].ravel()
        ab = cam.fit(sample, points.rng)
        if ab is None:
            continue
        _, height, z = cam.points(sample, ab)
        above = ((height > 6) & (height < d3.MAX_HEIGHT_CM) & (z < d3.MAX_RANGE_CM) & cam.usable)
        above = cv2.resize((above.reshape(cam.shape) * 255).astype(np.uint8), c.SIZE, interpolation=cv2.INTER_NEAREST)
        target = cv2.undistort(frame, proj.K, proj.dist)
        target_above = cv2.resize(cv2.undistort(above, proj.K, proj.dist), size, interpolation=cv2.INTER_NEAREST) > 0
        target = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
        if target_above.sum() < 2000:
            continue
        matrices = camera_matrices(proj, pose, scale)
        focal = float(matrices[2][0, 0])
        renders = {}
        for name in scores:
            canvas = np.zeros((size[1], size[0], 3), np.uint8)
            if name == "points":
                mask = draw_points(canvas, matrices, focal, xyz, color)
            else:
                d3.draw_voxels(canvas, matrices, focal, voxel_xyz, voxel_color.astype(np.float32))
                mask = canvas.any(-1)
            both = mask & target_above
            scores[name][0].append(both.sum() / target_above.sum())
            if both.sum() > 500:
                scores[name][1].append(float(np.abs(canvas[both].astype(int) - target[both].astype(int)).mean()))
            renders[name] = canvas
        shown = np.where(target_above[..., None], target, target // 4).astype(np.uint8)
        examples.append((target_above.sum(), np.hstack([shown, renders["cubes"], renders["points"]])))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = [(name, len(s[1]), round(100 * float(np.median(s[0])), 1), round(float(np.median(s[1])), 1))
            for name, s in scores.items() if s[1]]
    path = c.DATA_DIR / "metrics" / f"surfels_{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(("recording", "version", "hidden_frames", "coverage_percent", "mean_abs_error"))
        w.writerows((Path(a.folder).name, *r) for r in rows)
    for name, n, cover, error in rows:
        print(f"{name:7s} скрити кадри {n}: покрива {cover}% от предметите, грешка в цвета {error}")
    sheet = []
    for _, row in sorted(examples, key=lambda e: -e[0])[:3]:
        sheet.append(row)
    for i, text in enumerate(("hidden real frame", "cubes", "points")):
        cv2.putText(sheet[0], text, (i * size[0] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imencode(".png", np.vstack(sheet))[1].tofile(str(path.with_suffix(".png")))

    # видео: дрон, кубчета (ляво) срещу точки (дясно)
    out = c.DATA_DIR / "map"
    middle = np.array(builder.path).mean(0)
    video = cv2.VideoWriter(str(out / f"{stamp}_surfels.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                            (2 * sim.VIEW_W, sim.VIEW_H))
    camera = sim.ChaseCamera(120.0, 105.0)
    plain = np.full((sim.VIEW_H, sim.VIEW_W, 3), (24, 20, 20), np.uint8)
    sim.draw_grid = lambda img, cam: img.__setitem__(slice(None), plain)
    render_ms = []
    for n in range(240):
        yaw = car[2] + np.radians(65) * np.sin(2 * np.pi * n / 240)
        camera.pose = np.array([*middle, yaw])
        matrices = camera.matrices()
        left = sim.render_floor(floor, camera)
        right = left.copy()
        d3.draw_voxels(left, matrices, sim.FOCAL, voxel_xyz, voxel_color, near_cm=45)
        t = time.perf_counter()
        draw_points(right, matrices, sim.FOCAL, xyz, color, near_cm=45)
        render_ms.append((time.perf_counter() - t) * 1000)
        for view, text in ((left, "cubes"), (right, "real pixels")):
            sim.draw_boxes(view, camera, sim.truck_boxes(car))
            cv2.putText(view, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        screen = np.hstack([left, right])
        video.write(screen)
        if n in (0, 60, 180):
            cv2.imencode(".png", screen)[1].tofile(str(out / f"{stamp}_surfels_{n}.png"))
    video.release()
    print(f"дълбочина {np.median(depth_ms):.0f} ms на кадър, рисуване на точките {np.median(render_ms):.0f} ms")
    print(f"Записано: {out / stamp}_surfels*, {path}")


if __name__ == "__main__":
    main()
