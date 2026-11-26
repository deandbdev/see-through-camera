"""Обемни предмети около колата: дълбочина от един кадър, мащабирана спрямо пода.

  .venv/Scripts/python.exe src/depth3d.py data/odom/<папка>            # картинки и видео
  .venv/Scripts/python.exe src/sim.py data/odom/<папка> --obstacles    # в стенда

Depth Anything V2 Small дава само относителна дълбочина. Подът е известен от калибрацията,
затова намираме a, b така, че 1/разстояние = a*d + b по пода, и после всеки пиксел получава
сантиметри. Каквото е по-високо от MIN_HEIGHT_CM, не е под: трупа се в кубчета (воксели) и
не се лепи върху картата на пода. Невидените страни остават празни - нищо не се измисля.
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

import common as c
from motion import rot

MODEL = c.DATA_DIR.parent / "models" / "depth-anything-v2-small"
STEP = 4                   # през толкова пиксела взимаме точки
VOXEL_CM = 2.5
MIN_HEIGHT_CM, MAX_HEIGHT_CM = 4.0, 90.0
MAX_RANGE_CM = 150.0       # по-далече грешката на дълбочината расте бързо
MIN_VIEWS = 3              # в толкова кадъра трябва да е видяно кубчето
EDGE = 0.06                # по-голям скок в дълбочината = ръб с "летящи" точки
SIZE_CM = 600.0


class DepthModel:
    def __init__(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if not MODEL.exists():
            raise SystemExit(f"Няма модел в {MODEL}.")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(self.device).eval()

    def __call__(self, frame):
        """BGR кадър -> относителна дълбочина (по-голямо = по-близо) със същия размер."""
        x = self.processor(images=frame[:, :, ::-1].copy(), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**x).predicted_depth
            out = self.torch.nn.functional.interpolate(out[None], frame.shape[:2], mode="bilinear")
        return out[0, 0].cpu().numpy()


class CameraRays:
    """Лъчите на една камера през решетка от пиксели и къде всеки от тях среща пода."""

    def __init__(self, proj, step=STEP):
        self.proj = proj
        w, h = c.SIZE
        u, v = np.meshgrid(np.arange(step // 2, w, step), np.arange(step // 2, h, step))
        self.u, self.v, self.shape = u, v, u.shape
        px = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
        und = cv2.undistortPoints(px, proj.K, proj.dist).reshape(-1, 2)
        self.rays = np.c_[und, np.ones(len(und))]
        down = self.rays @ proj.R  # посоките в координатите на дъската
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (proj.R.T @ proj.t)[2] / down[:, 2]
        local, hits = proj.pixels_to_local(px)
        body = np.zeros(len(u.ravel()), bool) if proj.body is None else proj.body[v.ravel(), u.ravel()] > 0
        self.usable = ~body
        self.floor = (self.usable & hits & (z > 0) & (local[:, 1] > 25) & (local[:, 1] < 150) &
                      (np.abs(local[:, 0]) < 90))
        self.inv_floor = np.where(self.floor, 1.0 / np.where(z > 0, z, 1.0), 0.0)
        self.up = np.sign((-proj.R.T @ proj.t)[2])

    def fit(self, d, rng):
        """a, b с 1/z = a*d + b по пода (RANSAC: на пода може да стоят предмети) или None."""
        x, y = d[self.floor], self.inv_floor[self.floor]
        if len(x) < 500:
            return None
        every = max(1, len(x) // 4000)  # права с два параметъра: няколко хиляди точки стигат
        x, y = x[::every], y[::every]
        best = np.zeros(len(x), bool)
        for _ in range(150):
            i, j = rng.choice(len(x), 2, replace=False)
            if abs(x[i] - x[j]) < 0.2:
                continue
            a = (y[i] - y[j]) / (x[i] - x[j])
            inliers = np.abs(a * x + y[i] - a * x[i] - y) < 0.04 * y
            if a > 0 and inliers.sum() > best.sum():
                best = inliers
        if best.sum() < max(500 // every, 0.25 * len(x)):
            return None
        a, b = np.polyfit(x[best], y[best], 1, w=1 / y[best])
        return a, b

    def points(self, d, ab):
        """(x, y спрямо камерата на пода, височина) за всяка точка от решетката и z по оста."""
        z = 1.0 / np.maximum(ab[0] * d + ab[1], 1e-4)
        board = (self.rays * z[:, None] - self.proj.t) @ self.proj.R
        local = (board[:, :2] - self.proj.origin) @ self.proj.axes.T
        return local, board[:, 2] * self.up, z


class Obstacles:
    """Натрупва точките над пода в кубчета в координатите на света."""

    def __init__(self, projectors):
        self.model = DepthModel()
        self.cameras = [None if p is None else CameraRays(p) for p in projectors]
        self.rng = np.random.default_rng(0)
        self.clear()

    def clear(self):
        self.n = int(SIZE_CM / VOXEL_CM)
        self.nz = int(MAX_HEIGHT_CM / VOXEL_CM)
        total = self.n * self.n * self.nz
        self.views = np.zeros(total, np.int32)
        self.color = np.zeros((total, 3), np.float32)
        self.count = np.zeros(total, np.float32)
        self.center = None
        self.frames = self.failed = 0
        self._cache = None

    def observe(self, frame, camera, pose, moving=None):
        """Добавя кадъра на камера 0 (предна) или 1 (задна) от тази поза.

        moving: маска на хора и играчки (dynamic.py) - те не влизат в неподвижните кубчета.
        Връща маска (uint8, размер на кадъра) на пикселите, които не бива да се лепят върху пода.
        """
        cam = self.cameras[camera]
        if self.center is None:
            self.center = pose[:2].copy()
        depth = self.last_depth = self.model(frame)  # ползва я и surfels.py
        d = depth[cam.v, cam.u].ravel()
        ab = cam.fit(d, self.rng)
        self.frames += 1
        if ab is None:  # не се вижда достатъчно под, за да знаем мащаба
            self.failed += 1
            return np.full(frame.shape[:2], 255, np.uint8)  # щом не знаем кое е под, не лепим нищо
        local, height, z = cam.points(d, ab)

        grid = z.reshape(cam.shape).astype(np.float32)
        jump = cv2.morphologyEx(grid, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) / grid
        above = cam.usable & (height > MIN_HEIGHT_CM)
        solid = above & (height < MAX_HEIGHT_CM) & (z < MAX_RANGE_CM) & (jump.ravel() < EDGE)
        if moving is not None:
            solid &= moving[cam.v, cam.u].ravel() == 0

        world = local[solid] @ rot(pose[2]).T + pose[:2]
        ij = np.floor((world - self.center + SIZE_CM / 2) / VOXEL_CM).astype(int)
        k = np.floor(height[solid] / VOXEL_CM).astype(int)
        inside = (ij >= 0).all(1) & (ij < self.n).all(1) & (k >= 0) & (k < self.nz)
        index = (ij[inside, 0] * self.n + ij[inside, 1]) * self.nz + k[inside]
        colors = frame[cam.v, cam.u].reshape(-1, 3)[solid][inside].astype(np.float32)
        np.add.at(self.color, index, colors)
        np.add.at(self.count, index, 1.0)
        self.views[np.unique(index)] += 1
        self._cache = None

        mask = (above.reshape(cam.shape) * 255).astype(np.uint8)
        mask = cv2.resize(mask, c.SIZE, interpolation=cv2.INTER_NEAREST)
        return cv2.dilate(mask, np.ones((9, 9), np.uint8))

    def voxels(self, min_views=MIN_VIEWS):
        """(центрове x, y, z в света, цветове) на достатъчно пъти видените кубчета."""
        if self._cache is None:
            index = np.flatnonzero(self.views >= min_views)
            k = index % self.nz
            i, j = np.divmod(index // self.nz, self.n)
            xyz = np.stack([(i + 0.5) * VOXEL_CM - SIZE_CM / 2, (j + 0.5) * VOXEL_CM - SIZE_CM / 2,
                            (k + 0.5) * VOXEL_CM], 1)
            if len(index):
                xyz[:, :2] += self.center
            self._cache = xyz, (self.color[index] / np.maximum(self.count[index], 1)[:, None])
        return self._cache


def draw_voxels(img, cam_matrices, focal, xyz, colors, near_cm=0.0, far_cm=1e9):
    """Кубчетата като квадратчета, от далечните към близките (само тези между near и far)."""
    R, C, K = cam_matrices
    if not len(xyz):
        return
    pts = (xyz - C) @ R.T
    keep = (pts[:, 2] > max(near_cm, 5.0)) & (pts[:, 2] <= far_cm)
    pts, colors, height = pts[keep], colors[keep], xyz[keep, 2]
    px = pts @ K.T
    px = px[:, :2] / px[:, 2:]
    half = np.maximum(focal * VOXEL_CM * 0.65 / pts[:, 2], 1.0)
    h, w = img.shape[:2]
    on = (px[:, 0] > -20) & (px[:, 0] < w + 20) & (px[:, 1] > -20) & (px[:, 1] < h + 20)
    shade = (0.8 + 0.25 * np.clip(height / 40.0, 0, 1))[:, None]
    colors = np.clip(colors * shade, 0, 255)
    for i in np.argsort(-pts[:, 2]):
        if on[i]:
            x, y, r = px[i, 0], px[i, 1], half[i]
            cv2.rectangle(img, (int(x - r), int(y - r)), (int(x + r), int(y + r)), colors[i].tolist(), -1)


def main():
    import sim
    from static_map import MapBuilder, replay_source

    p = argparse.ArgumentParser()
    p.add_argument("folder")
    a = p.parse_args()

    use_rear = (Path(a.folder) / "rear").exists()
    builder = MapBuilder(use_rear=use_rear, obstacles=True)
    started = time.time()
    car = None
    for stamp, frame, rear in replay_source(a.folder, use_rear):
        pose = builder.process(frame, rear, stamp)
        car = pose if pose is not None else car
    obstacles = builder.obstacles
    xyz, colors = obstacles.voxels()
    print(f"{obstacles.frames} кадъра с дълбочина, {obstacles.failed} без достатъчно под, "
          f"{len(xyz)} кубчета, {time.time() - started:.0f} s")

    out = c.DATA_DIR / "map"
    out.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d_%H%M%S")
    floor = builder.floor_map

    # отгоре: подът и предметите върху него (по-високите отгоре)
    top = floor.image.copy()
    px = floor.world_to_map(xyz[:, :2]).astype(int)
    half = int(np.ceil(VOXEL_CM * floor.s / 2))
    for i in np.argsort(xyz[:, 2]):
        cv2.rectangle(top, tuple(px[i] - half), tuple(px[i] + half), colors[i].tolist(), -1)
    everything = np.concatenate([np.stack(np.nonzero(np.isfinite(floor.dist))[::-1], 1), px])
    (x0, y0), (x1, y1) = np.maximum(everything.min(0) - 40, 0), everything.max(0) + 40
    cv2.imencode(".png", top[y0:y1, x0:x1])[1].tofile(str(out / f"{name}_3d_top.png"))

    # отстрани: камерата обикаля около средата на записаното
    middle = np.array(builder.path).mean(0)
    video = cv2.VideoWriter(str(out / f"{name}_3d_orbit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                            (sim.VIEW_W, sim.VIEW_H))
    camera = sim.ChaseCamera(130.0, 110.0)
    for n in range(240):
        yaw = car[2] + np.radians(70) * np.sin(2 * np.pi * n / 240)
        camera.pose = np.array([*middle, yaw])
        view = sim.render_floor(floor, camera)
        draw_voxels(view, camera.matrices(), sim.FOCAL, xyz, colors, near_cm=45)
        sim.draw_boxes(view, camera, sim.truck_boxes(car))
        video.write(view)
        if n in (0, 60, 180):
            cv2.imencode(".png", view)[1].tofile(str(out / f"{name}_3d_view{n}.png"))
    video.release()
    print(f"Записано: {out / name}_3d_*")


if __name__ == "__main__":
    main()
