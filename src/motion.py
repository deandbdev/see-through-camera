"""Одометрия по пода с оптичен поток (Lucas-Kanade) от предната камера.

Точки по пода се следят в суровия кадър, преобразуват се в сантиметри и от тях се
намира твърдото движение (завъртане + преместване) спрямо ключов кадър. Ключовият
кадър се сменя рядко, за да не се натрупва шум, докато колата стои или пълзи.

Поза: (x, y, theta) - x надясно, y напред спрямо началото, theta > 0 е завой наляво.
"""
import cv2
import numpy as np

import common as c

NEAR_CM, FAR_CM, HALF_WIDTH_CM = 30.0, 160.0, 90.0  # откъде по пода вземаме точки
MAX_POINTS = 300
MIN_POINTS = 25
KEY_DIST_CM, KEY_ANGLE_DEG = 5.0, 4.0
INLIER_CM = 1.0
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def rot(theta):
    s, co = np.sin(theta), np.cos(theta)
    return np.array([[co, -s], [s, co]])


def rigid_fit(src, dst):
    """R, t с dst ≈ src @ R.T + t (най-малки квадрати, без мащаб)."""
    ms, md = src.mean(0), dst.mean(0)
    h = (src - ms).T @ (dst - md)
    theta = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    return theta, md - rot(theta) @ ms


def ransac_rigid(src, dst, iterations=100, threshold=INLIER_CM, seed=0):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(src), bool)
    for _ in range(iterations):
        i = rng.choice(len(src), 2, replace=False)
        theta, t = rigid_fit(src[i], dst[i])
        inliers = np.linalg.norm(src @ rot(theta).T + t - dst, axis=1) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        return None
    theta, t = rigid_fit(src[best], dst[best])
    return theta, t, best


class Odometry:
    def __init__(self, projector):
        self.proj = projector
        self.mask = self._floor_mask()
        self.reset()

    def reset(self):
        self.pose = np.zeros(3)
        self.key_gray = None
        self.lost = 0          # колко кадъра не са могли да се оценят
        self.quality = 0.0     # дял от следените точки, които се движат като под
        self.dropped = 0
        self.points = np.empty((0, 2), np.float32)  # за показване
        self.inliers = np.empty(0, bool)

    def _floor_mask(self):
        """Пикселите от суровия кадър, които гледат пода в избраната зона."""
        w, h = c.SIZE
        u, v = np.meshgrid(np.arange(0, w, 4), np.arange(0, h, 4))
        local, hits = self.proj.pixels_to_local(np.stack([u, v], -1))
        ok = hits & (local[:, 1] > NEAR_CM) & (local[:, 1] < FAR_CM) & (np.abs(local[:, 0]) < HALF_WIDTH_CM)
        small = ok.reshape(u.shape).astype(np.uint8) * 255
        mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.proj.body is not None:  # части от колата не са под
            mask[cv2.dilate(self.proj.body, np.ones((9, 9), np.uint8)) > 0] = 0
        return mask

    def _new_keyframe(self, gray):
        pts = cv2.goodFeaturesToTrack(gray, MAX_POINTS, 0.01, 10, mask=self.mask, blockSize=5)
        if pts is None or len(pts) < MIN_POINTS:
            self.key_gray = None
            return
        self.key_gray = self.prev_gray = gray
        self.prev_pts = pts.astype(np.float32)
        self.key_floor = self.proj.pixels_to_local(pts)[0]
        self.key_count = len(pts)
        self.dropped = 0
        self.key_pose = self.pose.copy()

    def update(self, gray):
        """Подава се следващият сив кадър. Връща True, ако позата е оценена от него."""
        if self.key_gray is None:
            self._new_keyframe(gray)
            return False

        pts, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **LK)
        back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, pts, None, **LK)
        good = (st.ravel() == 1) & (st_back.ravel() == 1) & \
               (np.linalg.norm((back - self.prev_pts).reshape(-1, 2), axis=1) < 0.7)
        pts, key_floor = pts[good], self.key_floor[good]
        floor, hits = self.proj.pixels_to_local(pts)

        fit = None
        if hits.sum() >= MIN_POINTS:
            pts, floor, key_floor = pts[hits], floor[hits], key_floor[hits]
            fit = ransac_rigid(floor, key_floor)  # сегашни -> ключови координати
        if fit is None or fit[2].sum() < MIN_POINTS:
            self.lost += 1
            self.quality = 0.0
            self._new_keyframe(gray)  # движението в този интервал се губи
            return False

        theta, t, inliers = fit
        x, y, a = self.key_pose
        self.pose = np.array([*(np.array([x, y]) + rot(a) @ t), a + theta])
        self.points, self.inliers = pts.reshape(-1, 2), inliers
        self.dropped += int((~inliers).sum())
        self.quality = float(inliers.sum() / (inliers.sum() + self.dropped))

        # оттук нататък следим само точките, които се движат като пода
        self.prev_gray, self.prev_pts, self.key_floor = gray, pts[inliers], key_floor[inliers]
        moved = np.linalg.norm(t) > KEY_DIST_CM or abs(np.degrees(theta)) > KEY_ANGLE_DEG
        if moved or inliers.sum() < max(2 * MIN_POINTS, 0.5 * self.key_count):
            self._new_keyframe(gray)
        return True


FRONT_OK = 0.90      # при поне такъв дял съгласувани точки вярваме на предната
REAR_MARGIN = 0.05   # задната трябва да е с толкова по-добра, за да я предпочетем


def _compose(a, b):
    return np.array([*(a[:2] + rot(a[2]) @ b[:2]), a[2] + b[2]])


def _inverse(a):
    return np.array([*(-rot(-a[2]) @ a[:2]), -a[2]])


class DualOdometry:
    """Одометрия от двете камери: за всеки кадър се взема тази, която по-добре вижда под.

    Стена или шкаф пред камерата не се движат като под: делът съгласувани точки пада.
    """

    def __init__(self, front, rear, rear_offset):
        self.front, self.rear = Odometry(front), Odometry(rear)
        self.offset = np.asarray(rear_offset, np.float64)  # поза на задната камера спрямо колата
        self.reset()

    def reset(self):
        self.front.reset()
        self.rear.reset()
        self.pose = np.zeros(3)
        self.lost = self.rear_frames = 0
        self.source = "front"
        self.points, self.inliers = self.front.points, self.front.inliers

    @staticmethod
    def _step(odom, gray):
        before = odom.pose.copy()
        ok = odom.update(gray)
        return ok, _compose(_inverse(before), odom.pose)

    def update(self, gray, rear_gray=None):
        ok_f, step_f = self._step(self.front, gray)
        ok_r, step_r = (False, None) if rear_gray is None else self._step(self.rear, rear_gray)
        q_f, q_r = self.front.quality, self.rear.quality
        use_rear = ok_r and (not ok_f or (q_f < FRONT_OK and q_r > q_f + REAR_MARGIN))
        self.points, self.inliers = self.front.points, self.front.inliers
        if use_rear:
            # движението на задната камера -> движение на колата
            step = _compose(_compose(self.offset, step_r), _inverse(self.offset))
            self.source = "rear"
            self.rear_frames += 1
        elif ok_f:
            step, self.source = step_f, "front"
        else:
            self.lost += 1
            return False
        self.pose = _compose(self.pose, step)
        return True
