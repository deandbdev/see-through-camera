"""Проекция на пода от птичи поглед.

Координати на колата (см): началото е на пода точно под предната камера, x надясно, y напред.
Всяка камера има позиция pos_cm=(x, y) и посока yaw_deg (0 = напред, 180 = назад).
"""
import cv2
import numpy as np

import common as c


class FloorProjector:
    def __init__(self, camera, width_cm=120.0, far_cm=150.0, px_per_cm=5.0,
                 near_cm=0.0, pos_cm=(0.0, 0.0), yaw_deg=0.0):
        lens = c.CALIB_DIR / f"cam{camera}_lens.npz"
        floor = c.CALIB_DIR / f"cam{camera}_floor.npz"
        for path in (lens, floor):
            if not path.exists():
                raise SystemExit(f"Няма {path}. Първо пусни calibrate.py.")
        lens, floor = np.load(lens), np.load(floor)
        self.K, self.dist = lens["K"], lens["dist"]
        body = c.CALIB_DIR / f"cam{camera}_mask.png"  # части от колата в кадъра (make_mask.py)
        self.body = cv2.imdecode(np.fromfile(str(body), np.uint8), cv2.IMREAD_GRAYSCALE) \
            if body.exists() else None
        self.px_per_cm, self.width_cm, self.far_cm = px_per_cm, width_cm, far_cm
        self.near_cm = near_cm  # долният край на картата (отрицателен = зад колата)
        self.pos = np.array(pos_cm, np.float64)
        yaw = np.radians(yaw_deg)
        # редове: осите "надясно" и "напред" на камерата в координатите на колата
        self.heading = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self._pose(floor["corners"], float(floor["square_cm"]))
        self._build_maps()

    def _pose(self, corners, square_cm):
        """Позиция на камерата спрямо пода от кадъра с дъската."""
        obj = np.zeros((len(corners), 3), np.float64)
        obj[:, :2] = c.board_points_cm(square_cm)
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), self.K, self.dist)
        if not ok:
            raise SystemExit("Не мога да сметна позата на камерата, повтори calibrate.py floor.")
        R, _ = cv2.Rodrigues(rvec)
        cam = (-R.T @ tvec).ravel()  # камерата в координатите на дъската
        right, forward = R[0, :2], R[2, :2]  # осите "надясно" и "напред", легнали на пода
        forward = forward / np.linalg.norm(forward)
        right = right - forward * (right @ forward)
        right /= np.linalg.norm(right)
        self.height_cm = abs(cam[2])
        self.pitch_deg = float(np.degrees(np.arcsin(min(1.0, abs(R[2, 2])))))  # наклон надолу
        self.R, self.t = R, tvec.ravel()
        self.origin = cam[:2]
        self.axes = np.array([right, forward])  # дъска -> кола

    def to_local(self, pts_cm):
        """Координати на колата -> координати спрямо тази камера."""
        return (np.asarray(pts_cm, np.float64).reshape(-1, 2) - self.pos) @ self.heading.T

    def floor_to_pixels(self, pts_cm):
        """Точки на пода (спрямо камерата, см) -> пиксели в суровия кадър и маска кои се виждат."""
        pts = np.asarray(pts_cm, np.float64).reshape(-1, 2)
        board = pts @ self.axes + self.origin
        cam = np.c_[board, np.zeros(len(board))] @ self.R.T + self.t
        px = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(-1, 2)

        # Далеч извън кадъра полиномът на изкривяването се "прегъва", затова
        # режем и по ъгъл, не само по пиксели.
        w, h = c.SIZE
        border = np.array([(x, y) for x in np.linspace(0, w - 1, 33) for y in (0, h - 1)] +
                          [(x, y) for y in np.linspace(0, h - 1, 33) for x in (0, w - 1)], np.float32)
        und = cv2.undistortPoints(border.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        max_r = np.linalg.norm(und, axis=1).max()

        z = cam[:, 2]
        radius = np.linalg.norm(cam[:, :2], axis=1) / np.maximum(z, 1e-6)
        visible = ((z > 1) & (radius <= max_r) &
                   (px[:, 0] >= 0) & (px[:, 0] <= w - 1) & (px[:, 1] >= 0) & (px[:, 1] <= h - 1))
        return px, visible

    def pixels_to_local(self, px):
        """Сурови пиксели -> точки на пода спрямо камерата (см) и маска кои лъчи стигат пода."""
        px = np.asarray(px, np.float64).reshape(-1, 1, 2)
        if not len(px):  # OpenCV гърми при празен масив
            return np.empty((0, 2)), np.empty(0, bool)
        und = cv2.undistortPoints(px, self.K, self.dist).reshape(-1, 2)
        rays = np.c_[und, np.ones(len(und))] @ self.R  # посоките в координатите на дъската
        cam = -self.R.T @ self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -cam[2] / rays[:, 2]
        hits = np.isfinite(s) & (s > 0)
        board = cam[:2] + rays[:, :2] * np.where(hits, s, 0)[:, None]
        return (board - self.origin) @ self.axes.T, hits

    def height_at(self, px, ground_cm):
        """Височина над пода на лъча през пиксела, на ground_cm хоризонтално от камерата."""
        und = cv2.undistortPoints(np.float64(px).reshape(1, 1, 2), self.K, self.dist).reshape(2)
        ray = np.array([*und, 1.0]) @ self.R  # в координатите на дъската
        cam_z = float((-self.R.T @ self.t)[2])
        flat = np.hypot(ray[0], ray[1])
        return abs(cam_z) + ground_cm / max(flat, 1e-9) * ray[2] * np.sign(cam_z)

    def project_3d(self, local, z_cm):
        """Точка (спрямо камерата, см) на височина z_cm -> пиксел и дали е пред камерата."""
        board = np.asarray(local, np.float64) @ self.axes + self.origin
        up = np.sign((-self.R.T @ self.t)[2])
        cam = self.R @ np.array([*board, up * z_cm]) + self.t
        px = cv2.projectPoints(cam.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(2)
        return px, bool(cam[2] > 1)

    def _build_maps(self):
        s = self.px_per_cm
        self.size = (int(round(self.width_cm * s)), int(round((self.far_cm - self.near_cm) * s)))
        u, v = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]))
        x = (u + 0.5) / s - self.width_cm / 2
        y = self.far_cm - (v + 0.5) / s  # напред е нагоре в изображението
        px, visible = self.floor_to_pixels(self.to_local(np.stack([x, y], -1)))
        if self.body is not None:
            i = np.where(visible[:, None], px, 0).round().astype(int)
            visible = visible & (self.body[i[:, 1], i[:, 0]] == 0)
        px[~visible] = -1
        px = px.reshape(self.size[1], self.size[0], 2).astype(np.float32)
        self.mask = visible.reshape(self.size[1], self.size[0])  # кои пиксели са реално видени
        self._maps = cv2.convertMaps(px[..., 0], px[..., 1], cv2.CV_16SC2)

    def warp(self, frame):
        """Суров кадър -> изглед отгоре. Невидяното е черно."""
        out = cv2.remap(frame, *self._maps, cv2.INTER_LINEAR)
        out[~self.mask] = 0
        return out

    def draw_grid(self, img, step_cm=10):
        s, (w, h) = self.px_per_cm, self.size
        for x in np.arange(-(self.width_cm / 2 // step_cm) * step_cm, self.width_cm / 2 + 1e-6, step_cm):
            u = int((x + self.width_cm / 2) * s)
            cv2.line(img, (u, 0), (u, h), (0, 200, 255) if x == 0 else (90, 90, 90), 1)
        for y in np.arange(np.ceil(self.near_cm / step_cm) * step_cm, self.far_cm + 1e-6, step_cm):
            v = int((self.far_cm - y) * s)
            cv2.line(img, (0, v), (w, v), (90, 90, 90), 1)
            cv2.putText(img, f"{y:.0f}", (3, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
