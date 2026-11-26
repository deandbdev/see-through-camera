"""Помага ли Difix на нашия 3D рендер? Мерим, не само гледаме.

  .venv/Scripts/python.exe src/difix_eval.py render   # прави рендерите (дълбочина + точки)
  .venv/Scripts/python.exe src/difix_eval.py clean    # пуска Difix и мери

Две проверки:
1. скрити кадри: рисуваме гледната точка на кадър, който не е участвал, и сравняваме с него;
2. дрон изглед: четат ли се маркерите още и къде на пода се оказват (спрямо рулетката).
Картата на пода е от всички кадри, затова проверка 1 е по-лека за пода, отколкото за предметите.
"""
import argparse
import csv
import time

import cv2
import numpy as np

import common as c

RECORDING = "20261125_134157"
SIZE = (1024, 576)
CACHE = c.DATA_DIR / "difix" / "renders.npz"
BACKGROUND = (24, 20, 20)


class Viewpoint:
    """Произволна виртуална камера за sim.render_floor."""

    def __init__(self, R, C, K):
        self.R, self.C, self.K = R, C, K
        self.pose = np.zeros(3)

    def matrices(self):
        return self.R, self.C, self.K


def render():
    import sim
    import surfels
    from depth3d import DepthModel
    from static_map import MapBuilder, replay_source

    builder = MapBuilder(obstacles=True)
    keyframes = []
    original = builder.obstacles.observe

    def record(frame, camera, pose, moving=None):
        keyframes.append((frame.copy(), camera, pose.copy()))
        return original(frame, camera, pose, moving)

    builder.obstacles.observe = record
    for stamp, frame, _ in replay_source(c.DATA_DIR / "odom" / RECORDING, False):
        builder.process(frame, None, stamp)

    points, model = surfels.Surfels(builder.odom_projs), builder.obstacles.model
    hidden = []
    for k, (frame, camera, pose) in enumerate(keyframes):
        if (k // surfels.HOLD) % 3 == 2:  # всеки трети блок е скрит
            if k % surfels.HOLD == surfels.HOLD // 2:
                hidden.append((frame, camera, pose))
            continue
        points.observe(frame, model(frame), camera, pose)
    xyz, color = points.cloud()
    print(f"ключови кадри {len(keyframes)}, скрити {len(hidden)}, точки {len(xyz)}")

    plain = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, np.uint8)
    sim.VIEW_W, sim.VIEW_H = SIZE
    sim.draw_grid = lambda img, cam: img.__setitem__(slice(None), plain)

    def view_from(camera):
        sim.FOCAL = float(camera.matrices()[2][1, 1])
        image = sim.render_floor(builder.floor_map, camera)
        surfels.draw_points(image, camera.matrices(), sim.FOCAL, xyz, color)
        return image

    held, truth = [], []
    for frame, cam_index, pose in hidden:
        proj = builder.odom_projs[cam_index]
        camera = Viewpoint(*surfels.camera_matrices(proj, pose, SIZE[0] / c.SIZE[0]))
        held.append(view_from(camera))
        truth.append(cv2.resize(cv2.undistort(frame, proj.K, proj.dist), SIZE, interpolation=cv2.INTER_AREA))

    drone, matrices = [], []
    path = np.array(builder.path)
    for i, where in enumerate(np.linspace(0, len(path) - 1, 24).astype(int)):
        chase = sim.ChaseCamera(75.0, 95.0)
        sim.FOCAL, sim.AHEAD_CM = 0.75 * SIZE[0], 30.0
        heading = np.radians(12) * np.sin(i / 3.0)
        chase.pose = np.array([*path[where], heading])
        camera = Viewpoint(*chase.matrices())
        drone.append(view_from(camera))
        matrices.append(np.stack([np.c_[camera.R, camera.C], np.c_[camera.K, np.zeros(3)]]))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, held=np.array(held), truth=np.array(truth), drone=np.array(drone),
                        matrices=np.array(matrices))
    print(f"Записано: {CACHE}")


def clean(fixer="difix"):
    import json

    if fixer == "nf":
        from nanofix import NanoFix as Fixer
    else:
        from difix import Difix as Fixer

    data = np.load(CACHE)
    model = Fixer()
    model(data["held"][0])  # загряване
    marks = json.loads((c.CALIB_DIR / "markers.json").read_text(encoding="utf-8-sig"))["markers"]
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    seconds = []

    def run(image):
        t = time.perf_counter()
        out = model(image)
        seconds.append(time.perf_counter() - t)
        return out

    error = {"ours": [], "difix": []}
    sharp = {"ours": [], "difix": [], "truth": []}
    held_out = []
    for image, truth in zip(data["held"], data["truth"]):
        cleaned = run(image)
        for name, version in (("ours", image), ("difix", cleaned)):
            error[name].append(float(np.abs(version.astype(int) - truth.astype(int)).mean()))
        for name, version in (("ours", image), ("difix", cleaned), ("truth", truth)):
            sharp[name].append(float(cv2.Laplacian(cv2.cvtColor(version, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()))
        held_out.append(cleaned)

    def markers(image, R, C, K):
        to_floor = np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], -R @ C]))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        found = {}
        for quad, marker_id in zip(corners, [] if ids is None else ids.ravel()):
            p = to_floor @ np.r_[quad.reshape(4, 2).mean(0), 1.0]
            found[int(marker_id)] = p[:2] / p[2]
        return found

    read = {"ours": 0, "difix": 0}
    where = {"ours": [], "difix": []}
    false = {"ours": 0, "difix": 0}
    drone_out, total = [], 0
    for image, m in zip(data["drone"], data["matrices"]):
        R, C, K = m[0][:, :3], m[0][:, 3], m[1][:, :3]
        cleaned = run(image)
        drone_out.append(cleaned)
        baseline = markers(image, R, C, K)
        total += len(baseline)
        for name, version in (("ours", image), ("difix", cleaned)):
            for marker_id, position in markers(version, R, C, K).items():
                if marker_id in baseline:
                    read[name] += 1
                    truth_xy = np.array((marks[str(marker_id)]["x"], marks[str(marker_id)]["y"]))
                    where[name].append(float(np.linalg.norm(position - truth_xy)))
                elif str(marker_id) not in marks:
                    false[name] += 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = c.DATA_DIR / "metrics" / f"{fixer}_{stamp}.csv"
    rows = [(name, len(held_out), round(float(np.mean(error[name])), 1), round(float(np.median(sharp[name])), 0),
             total, read[name], round(float(np.median(where[name])), 1) if where[name] else "", false[name])
            for name in ("ours", "difix")]
    rows = [(fixer if r[0] == "difix" else r[0], *r[1:]) for r in rows]
    header = ("version", "hidden_frames", "mean_abs_error", "sharpness", "markers_in_ours", "markers_read",
              "position_error_median_cm", "false_ids")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    for row in rows:
        print(dict(zip(header, row)))
    print(f"острота на истинските кадри {np.median(sharp['truth']):.0f}; {fixer} {np.median(seconds) * 1000:.0f} ms на кадър")

    pick = np.linspace(0, len(held_out) - 1, 3).astype(int)
    sheet = [np.hstack([data["truth"][i], data["held"][i], held_out[i]]) for i in pick]
    sheet += [np.hstack([np.zeros_like(drone_out[0]), data["drone"][i], drone_out[i]]) for i in (4, 12, 20)]
    for i, text in enumerate(("real hidden frame", "our render", f"after {fixer}")):
        cv2.putText(sheet[0], text, (i * SIZE[0] + 10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    sheet = cv2.resize(np.vstack(sheet), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", sheet)[1].tofile(str(path.with_suffix(".png")))
    print(f"Записано: {path} и .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("render", "clean"))
    p.add_argument("--fixer", choices=("difix", "nf"), default="difix")
    a = p.parse_args()
    render() if a.what == "render" else clean(a.fixer)
