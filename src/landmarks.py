"""Позиция по ориентири вместо по маркери.

  .venv/Scripts/python.exe src/landmarks.py            # карта от 134157, проверка на 134822
  .venv/Scripts/python.exe src/landmarks.py --memory 20261125_134822 --drive 20261125_134157

Ориентир = забележима точка (SIFT) с 3D място в света. Строи се от минаването в паметта
чрез триангулация между кадрите. При новото минаване кадърът се сравнява с картата и от
съвпаденията се смята (x, y, посока); между две такива поправки кара одометрията.

Маркерите са залепени в тези записи, затова всичко на пода около тях (целият лист A4) се
СКРИВА преди търсенето на ориентири и в двете минавания. Те дават само координатите на
паметта и еталона, с който се мери грешката.
"""
import argparse
import csv
import time

import cv2
import numpy as np

import common as c
from localize import compose, inverse, wrap_deg
from motion import Odometry, rot
from projection import FloorProjector
from static_map import MapBuilder, replay_source

FEATURES = 1500
KEY_CM, KEY_DEG = 3.0, 3.0   # ключов кадър за картата след такова движение
RATIO = 0.8
MIN_VIEWS = 3
MIN_PARALLAX_DEG = 2.5
MAX_REPROJ_PX = 2.5
MIN_INLIERS = 15             # под това поправката не се приема
RANSAC_PX = 4.0
EVERY = 5                    # на колко кадъра се търси поправка
BLEND = 0.35
HIDE_CM = 22.0               # радиус около маркер: листът A4 с надписите
MAX_JUMP_CM = 30.0           # по-голям скок се приема само след няколко поредни съгласни


def camera_model(proj):
    """cam = M @ (x надясно, y напред, z нагоре спрямо колата) + m"""
    up = np.sign((-proj.R.T @ proj.t)[2])
    block = np.eye(3)
    block[:2, :2] = proj.axes.T
    block[2, 2] = up
    M = proj.R @ block
    m = proj.R @ np.array([*proj.origin, 0.0]) + proj.t
    return M, m


def yaw_matrix(theta):
    """Свят -> кола за завъртането (само около вертикалата)."""
    R = np.eye(3)
    R[:2, :2] = rot(-theta)
    return R


def world_to_camera(pose, M, m):
    R = M @ yaw_matrix(pose[2])
    return R, m - R @ np.array([pose[0], pose[1], 0.0])


class Features:
    """SIFT в суровия кадър; частите от колата не се гледат."""

    def __init__(self, proj):
        self.proj = proj
        self.sift = cv2.SIFT_create(nfeatures=FEATURES, contrastThreshold=0.02)
        self.mask = None if proj.body is None else np.where(proj.body > 0, 0, 255).astype(np.uint8)

    def __call__(self, frame, hide=None):
        """hide = (поза, центрове на маркери): точките по пода около тях се изхвърлят."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points, desc = self.sift.detectAndCompute(gray, self.mask)
        if desc is None:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 128), np.float32)
        px = np.array([p.pt for p in points], np.float64)
        if hide is not None:
            pose, centres = hide
            local, hits = self.proj.pixels_to_local(px)
            world = local @ rot(pose[2]).T + pose[:2]
            near = (np.linalg.norm(world[:, None] - centres[None], axis=2) < HIDE_CM).any(1)
            keep = ~(hits & near)
            px, desc = px[keep], desc[keep]
        und = cv2.undistortPoints(px.reshape(-1, 1, 2), self.proj.K, self.proj.dist).reshape(-1, 2)
        return px, und, desc


def marker_centres(localizer):
    return np.array([(m["x"], m["y"]) for m in localizer.markers.values()], np.float64)


def match(a, b):
    """Двойки (i, j): най-близък, ясно по-добър от втория и взаимен."""
    if len(a) < 2 or len(b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a, b, k=2)
    back = {m.queryIdx: m.trainIdx for m in matcher.match(b, a)}
    return [(m.queryIdx, m.trainIdx) for m, n in forward
            if m.distance < RATIO * n.distance and back.get(m.trainIdx) == m.queryIdx]


def triangulate(views, focal):
    """views: [(R, t, und)] -> точка в света или None, ако не е надеждна."""
    rows = []
    for R, t, u in views:
        P = np.c_[R, t]
        rows += [u[0] * P[2] - P[0], u[1] * P[2] - P[1]]
    h = np.linalg.svd(np.array(rows))[2][-1]
    if abs(h[3]) < 1e-9:
        return None
    X = h[:3] / h[3]
    errors, rays = [], []
    for R, t, u in views:
        cam = R @ X + t
        if cam[2] < 10 or cam[2] > 350:
            return None
        errors.append(np.linalg.norm(cam[:2] / cam[2] - u) * focal)
        ray = X - (-R.T @ t)
        rays.append(ray / np.linalg.norm(ray))
    rays = np.array(rays)
    parallax = np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1)))
    if np.max(errors) > MAX_REPROJ_PX or parallax < MIN_PARALLAX_DEG or not -4 < X[2] < 100:
        return None
    return X


def build_map(folder):
    """-> (точки (N,3), дескриптори (K,128), към коя точка е всеки дескриптор)"""
    builder = MapBuilder()
    proj = builder.fusion.odom.proj
    features = Features(proj)
    centres = marker_centres(builder.localizer)
    M, m = camera_model(proj)
    keys, last = [], None
    for stamp, frame, _ in replay_source(folder, False):
        pose = builder.process(frame, None, stamp)
        if pose is None:
            continue
        if last is not None and np.hypot(*(pose[:2] - last[:2])) < KEY_CM and \
                abs(wrap_deg(pose[2] - last[2])) < KEY_DEG:
            continue
        last = pose.copy()
        px, und, desc = features(frame, (pose, centres))
        keys.append((world_to_camera(pose, M, m), und, desc))

    parent = {}  # обединение на съвпаденията в "следи": (кадър, точка)

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(keys)):
        for step in (1, 3, 8):
            if i + step < len(keys):
                for a, b in match(keys[i][2], keys[i + step][2]):
                    parent[find((i, a))] = find((i + step, b))
    tracks = {}
    for node in list(parent):
        tracks.setdefault(find(node), []).append(node)

    focal = proj.K[0, 0]
    points, descriptors, owner = [], [], []
    for nodes in tracks.values():
        by_frame = {}
        for frame_i, k in nodes:
            by_frame.setdefault(frame_i, k)
        if len(by_frame) < MIN_VIEWS:
            continue
        order = sorted(by_frame)
        chosen = order if len(order) <= 8 else [order[j] for j in np.linspace(0, len(order) - 1, 8).astype(int)]
        X = triangulate([(*keys[f][0], keys[f][1][by_frame[f]]) for f in chosen], focal)
        if X is None:
            continue
        # по няколко вида на точката: отблизо и отдалеч изглежда различно
        for f in sorted({order[0], order[len(order) // 2], order[-1]}):
            descriptors.append(keys[f][2][by_frame[f]])
            owner.append(len(points))
        points.append(X)
    return np.array(points, np.float64), np.array(descriptors, np.float32), np.array(owner, np.int64)


def refine(pose, X, und, M, m, focal, iterations=8):
    """Само трите неизвестни (x, y, посока): височината и наклонът на камерата са известни."""
    def residuals(p):
        R, t = world_to_camera(p, M, m)
        cam = X @ R.T + t
        return ((cam[:, :2] / np.maximum(cam[:, 2:], 1) - und) * focal).ravel()

    pose = pose.copy()
    for _ in range(iterations):
        r = residuals(pose)
        J = np.empty((len(r), 3))
        for k, h in enumerate((0.01, 0.01, 1e-4)):
            d = np.zeros(3)
            d[k] = h
            J[:, k] = (residuals(pose + d) - r) / h
        w = np.minimum(1.0, 2.0 / np.maximum(np.abs(r), 1e-9))  # Хубер: грешните тежат малко
        A = J.T @ (J * w[:, None])
        step = np.linalg.solve(A + 1e-9 * np.eye(3), -J.T @ (w * r))
        pose += step
        if np.abs(step[:2]).max() < 0.01 and abs(step[2]) < 1e-4:
            break
    return pose, np.linalg.norm(residuals(pose).reshape(-1, 2), axis=1)


class Landmarks:
    """Позата на колата от един кадър спрямо картата от ориентири."""

    def __init__(self, points, descriptors, owner, proj):
        self.points, self.desc, self.owner = points, descriptors, owner
        self.features = Features(proj)
        self.M, self.m = camera_model(proj)
        self.focal = proj.K[0, 0]
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.matcher.add([self.desc])
        self.matcher.train()

    def locate(self, frame, hide=None):
        """-> (поза (x, y, theta), брой съгласни ориентири) или (None, брой)."""
        _, und, desc = self.features(frame, hide)
        if len(desc) < MIN_INLIERS:
            return None, 0
        best = {}
        for found in self.matcher.knnMatch(desc, k=6):
            first = self.owner[found[0].trainIdx]
            # вторият трябва да е ДРУГА точка, не друг вид на същата
            second = next((f.distance for f in found[1:] if self.owner[f.trainIdx] != first), None)
            if second is None or found[0].distance < RATIO * second:
                if first not in best or found[0].distance < best[first][0]:
                    best[first] = (found[0].distance, found[0].queryIdx)
        if len(best) < MIN_INLIERS:
            return None, len(best)
        X = self.points[list(best)]
        u = und[[q for _, q in best.values()]]
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            X, u, np.eye(3), None, iterationsCount=300, reprojectionError=RANSAC_PX / self.focal,
            confidence=0.999, flags=cv2.SOLVEPNP_AP3P)
        if not ok or inliers is None or len(inliers) < MIN_INLIERS:
            return None, 0 if inliers is None else len(inliers)
        inliers = inliers.ravel()
        R = cv2.Rodrigues(rvec)[0]
        yaw = self.M.T @ R
        origin = R.T @ (self.m - tvec.ravel())  # началото на колата в света
        pose = np.array([origin[0], origin[1], np.arctan2(yaw[0, 1], yaw[0, 0])])
        if abs(origin[2]) > 15:  # колата не лети: решението е грешно
            return None, len(inliers)
        pose, errors = refine(pose, X[inliers], u[inliers], self.M, self.m, self.focal)
        good = int((errors < RANSAC_PX).sum())
        return (pose if good >= MIN_INLIERS else None), good


class LandmarkPose:
    """Одометрия всеки кадър + поправка от ориентирите, когато има такава."""

    def __init__(self, projector):
        self.odom = Odometry(projector)
        self.correction = None  # одометрия -> свят
        self.waiting = []       # далечни предложения, които още не са потвърдени

    def move(self, gray):
        self.odom.update(gray)
        return self.pose()

    def pose(self):
        return None if self.correction is None else compose(self.correction, self.odom.pose)

    def correct(self, fix, odom_at_fix):
        """fix е позата за кадъра, при който одометрията е била odom_at_fix (може да е по-стар)."""
        proposal = compose(fix, inverse(odom_at_fix))
        if self.correction is None:
            self.correction = proposal
            return True
        pose = compose(self.correction, odom_at_fix)
        diff = fix - pose
        diff[2] = np.radians(wrap_deg(diff[2]))
        if np.hypot(*diff[:2]) > MAX_JUMP_CM:
            self.waiting = [w for w in self.waiting
                            if np.hypot(*(compose(w, odom_at_fix)[:2] - fix[:2])) < MAX_JUMP_CM / 2]
            self.waiting.append(proposal)
            if len(self.waiting) < 3:
                return False
            self.correction, self.waiting = proposal, []
            return True
        self.waiting = []
        self.correction = compose(pose + BLEND * diff, inverse(odom_at_fix))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory", default="20261125_134157")
    p.add_argument("--drive", default="20261125_134822")
    p.add_argument("--every", type=int, default=EVERY)
    a = p.parse_args()
    cv2.setRNGSeed(0)

    t0 = time.perf_counter()
    points, desc, owner = build_map(c.DATA_DIR / "odom" / a.memory)
    build_s = time.perf_counter() - t0
    out = c.DATA_DIR / "map"
    np.savez_compressed(out / f"landmarks_{a.memory}.npz", points=points, desc=desc, owner=owner)
    print(f"карта от {a.memory}: {len(points)} ориентира ({(points[:, 2] < 3).sum()} на пода), "
          f"за {build_s:.0f} s")

    truth_builder = MapBuilder()  # еталонът: позата по маркерите
    proj = truth_builder.fusion.odom.proj
    landmarks = Landmarks(points, desc, owner, proj)
    centres = marker_centres(truth_builder.localizer)
    ours = LandmarkPose(FloorProjector(0))
    rows, locate_ms, alone = [], [], []
    start = first_fix = frozen = None
    attempts = fixes = 0
    for i, (stamp, frame, _) in enumerate(replay_source(c.DATA_DIR / "odom" / a.drive, False)):
        start = stamp if start is None else start
        truth = truth_builder.process(frame, None, stamp)
        ours.move(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fix, inliers = None, ""
        if i % a.every == 0 and truth is not None:
            t0 = time.perf_counter()
            fix, inliers = landmarks.locate(frame, (truth, centres))
            locate_ms.append((time.perf_counter() - t0) * 1000)
            attempts += 1
            if fix is not None:
                fixes += 1
                if ours.correct(fix, ours.odom.pose.copy()) and first_fix is None:
                    first_fix = stamp - start
                    frozen = ours.correction.copy()  # гола одометрия от тук нататък
        pose = ours.pose()
        error = lambda q: ["", ""] if q is None or truth is None else \
            [float(np.hypot(*(q[:2] - truth[:2]))), abs(wrap_deg(q[2] - truth[2]))]
        if fix is not None and truth is not None:
            alone.append(error(fix))
        dead = None if frozen is None else compose(frozen, ours.odom.pose)
        rows.append([stamp - start, inliers, *error(pose), *error(fix), *error(dead)])

    path = c.DATA_DIR / "metrics" / f"landmarks_{a.memory[-6:]}_{a.drive[-6:]}_{time.strftime('%H%M%S')}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow("time_s inliers pose_err_cm pose_err_deg fix_err_cm fix_err_deg odo_err_cm odo_err_deg".split())
        w.writerows(rows)
    err = np.array([(r[2], r[3]) for r in rows if r[2] != ""], float)
    alone = np.array(alone, float)
    if not len(err):
        print("Нито една приета поправка.")
        return
    print(f"проверка на {a.drive}: {len(rows)} кадъра, първа позиция след {first_fix:.1f} s")
    print(f"  успешни поправки: {fixes}/{attempts}")
    print(f"  самостоятелна поправка: медиана {np.median(alone[:, 0]):.1f} см / {np.median(alone[:, 1]):.1f}°, "
          f"p90 {np.percentile(alone[:, 0], 90):.1f} см, макс {alone[:, 0].max():.1f} см")
    print(f"  позата във всеки кадър: медиана {np.median(err[:, 0]):.1f} см / {np.median(err[:, 1]):.1f}°, "
          f"p90 {np.percentile(err[:, 0], 90):.1f} см, макс {err[:, 0].max():.1f} см")
    dead = np.array([(r[6], r[7]) for r in rows if r[6] != ""], float)
    print(f"  само одометрия (без поправки): медиана {np.median(dead[:, 0]):.1f} см, макс {dead[:, 0].max():.1f} см, "
          f"в края {dead[-1, 0]:.1f} см / {dead[-1, 1]:.1f}°")
    print(f"  търсене: медиана {np.median(locate_ms):.0f} ms")
    print(f"Записано: {path}")


if __name__ == "__main__":
    main()
