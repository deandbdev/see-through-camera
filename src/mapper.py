"""Статична карта на пода: кадрите отгоре се лепят в световни координати.

Всеки пиксел пази от какво разстояние е видян. По-близък (по-остър) поглед го
презаписва, по-далечен - не.
"""
import cv2
import numpy as np

from motion import rot

REFRESH_CM = 10.0  # поглед от почти същото разстояние също обновява (подът може да се е сменил)


class FloorMap:
    def __init__(self, projector, size_cm=600.0, center=(0.0, 0.0)):
        """projector: FloorProjector в координатите на колата; px_per_cm определя и картата."""
        self.proj = projector
        self.s = projector.px_per_cm
        n = int(round(size_cm * self.s))
        self.origin = np.array([center[0] - size_cm / 2, center[1] + size_cm / 2])  # горе-ляво, см
        self.image = np.zeros((n, n, 3), np.uint8)
        self.dist = np.full((n, n), np.inf, np.float32)  # разстояние на най-добрия поглед
        self._view_dists = {}
        self.balance = False  # изравнява яркостта на новия кадър с вече залепеното

    def _view_dist(self, proj):
        """Разстояние от камерата за всеки пиксел на изгледа ѝ отгоре."""
        if id(proj) not in self._view_dists:
            w, h = proj.size
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            x = (u + 0.5) / self.s - proj.width_cm / 2
            y = proj.far_cm - (v + 0.5) / self.s
            dist = np.hypot(x, y).astype(np.float32)
            dist[~proj.mask] = np.inf
            self._view_dists[id(proj)] = dist
        return self._view_dists[id(proj)]

    def world_to_map(self, pts):
        pts = np.asarray(pts, np.float64)
        return np.stack([(pts[..., 0] - self.origin[0]) * self.s - 0.5,
                         (self.origin[1] - pts[..., 1]) * self.s - 0.5], -1)

    def map_to_world(self, px):
        px = (np.asarray(px, np.float64) + 0.5) / self.s
        return np.stack([self.origin[0] + px[..., 0], self.origin[1] - px[..., 1]], -1)

    def _view_to_map(self, proj, pose):
        """Афинна матрица: пиксел от изгледа отгоре (камера) -> пиксел от картата."""
        src = np.array([(0, 0), (100, 0), (0, 100)], np.float32)
        local = np.stack([(src[:, 0] + 0.5) / self.s - proj.width_cm / 2,
                          proj.far_cm - (src[:, 1] + 0.5) / self.s], -1)
        world = local @ rot(pose[2]).T + pose[:2]
        return cv2.getAffineTransform(src, self.world_to_map(world).astype(np.float32))

    def add(self, frame, pose, proj=None, penalty_cm=0.0, exclude=None):
        """pose е позата на камерата, заснела кадъра (по подразбиране предната).

        penalty_cm прави по-лоша камера да отстъпва: брои се, все едно е толкова по-далече.
        exclude: маска върху суровия кадър с пиксели, които не бива да влизат (хора и т.н.).
        """
        proj = proj or self.proj
        top = proj.warp(frame)
        M = self._view_to_map(proj, pose)
        w, h = proj.size
        corners = cv2.transform(np.array([[(0, 0), (w, 0), (w, h), (0, h)]], np.float64), M)[0]
        x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, self.image.shape[1])
        if x0 >= x1 or y0 >= y1:
            return
        M = M.copy()
        M[:, 2] -= (x0, y0)
        size = (int(x1 - x0), int(y1 - y0))
        patch = cv2.warpAffine(top, M, size, flags=cv2.INTER_LINEAR)
        # NEAREST и свит ръб, за да не се лепи черна рамка от невидяното
        view_dist = self._view_dist(proj)
        if exclude is not None:
            view_dist = np.where(proj.warp(exclude) > 0, np.float32(np.inf), view_dist)
        dist = cv2.warpAffine(view_dist, M, size, flags=cv2.INTER_NEAREST,
                              borderValue=float("inf"))
        valid = cv2.erode(np.isfinite(dist).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        dist = dist + penalty_cm
        roi_dist = self.dist[y0:y1, x0:x1]
        better = valid & (dist <= roi_dist + REFRESH_CM)
        if self.balance:
            both = (valid & np.isfinite(roi_dist))[::4, ::4]
            if both.sum() > 200:
                old = self.image[y0:y1, x0:x1][::4, ::4][both].mean(0)
                new = patch[::4, ::4][both].mean(0)
                gain = np.clip(old / np.maximum(new, 1.0), 0.8, 1.25) ** 0.8  # не изцяло: да не "тече"
                patch = cv2.multiply(patch, (*gain, 1.0))
        self.image[y0:y1, x0:x1][better] = patch[better]
        roi_dist[better] = np.minimum(roi_dist, dist)[better]

    def car_view(self, pose, width_cm=140.0, ahead_cm=120.0, behind_cm=80.0):
        """Изрязък от картата около колата, с посоката ѝ нагоре."""
        s = self.s
        size = (int(width_cm * s), int((ahead_cm + behind_cm) * s))
        dst = np.array([(0, 0), (100, 0), (0, 100)], np.float32)
        local = np.stack([(dst[:, 0] + 0.5) / s - width_cm / 2, ahead_cm - (dst[:, 1] + 0.5) / s], -1)
        world = local @ rot(pose[2]).T + pose[:2]
        M = cv2.getAffineTransform(dst, self.world_to_map(world).astype(np.float32))
        view = cv2.warpAffine(self.image, M, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        cx, cy = int(width_cm / 2 * s), int(ahead_cm * s)
        draw_car(view, np.array([(cx, cy - 6 * s), (cx - 4 * s, cy + 4 * s), (cx + 4 * s, cy + 4 * s)]))
        return view

    def cropped(self, margin_cm=10.0):
        """Само попълнената част на картата и отместването ѝ в пиксели."""
        ys, xs = np.nonzero(np.isfinite(self.dist))
        if not len(xs):
            return self.image[:1, :1].copy(), (0, 0)
        m = int(margin_cm * self.s)
        x0, y0 = max(xs.min() - m, 0), max(ys.min() - m, 0)
        return self.image[y0:ys.max() + m, x0:xs.max() + m].copy(), (x0, y0)


def draw_car(img, triangle):
    pts = triangle.astype(np.int32)
    cv2.fillPoly(img, [pts], (0, 200, 255))
    cv2.polylines(img, [pts], True, (0, 0, 0), 1)
