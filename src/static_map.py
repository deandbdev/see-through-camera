"""Стъпка 7: натрупваща се карта на пода, на живо или от запис.

  .venv/Scripts/python.exe src/static_map.py                                # на живо
  .venv/Scripts/python.exe src/static_map.py --replay data/odom/<папка> --video
  .venv/Scripts/python.exe src/static_map.py --rear --record                # и задната камера

Задната камера (--rear) участва в картата и в одометрията: когато предната не вижда под
(стена, шкаф), движението се взема от задната и предните кадри не се лепят. Положението ѝ
спрямо предната е в calib/rig.json (виж calibrate_rig.py). При replay се ползва сама,
ако записът има подпапка rear/.

Позиция: оптичен поток, коригиран от ArUco маркерите (--no-correct: само при първия
видян маркер, за сравнение; --no-tags: без маркери, светът е началната поза).
--objects: хора, животни и играчки не влизат в картата (dynamic.py, YOLO).
MapBuilder(obstacles=True): предметите над пода отиват в 3D (depth3d.py), не върху пода.
R = изчисти, S = запиши, ESC = изход. Накрая картата се записва в data/map/.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import common as c
from calibrate import label
from capture import CameraThread, synced_pair
from dual import RIG_PATH
from dynamic import Detector, Tracker, exclusion, locate
from localize import Fusion, compose, inverse, wrap_deg
from mapper import FloorMap, draw_car
from motion import DualOdometry, rot
from odometry import replay_frames
from projection import FloorProjector
from tags import TagLocalizer

REAR_PENALTY_CM = 80.0  # задната камера само допълва, не презаписва видяното от предната
BLEND = 0.1                         # колко бързо позата се придърпва към маркерите
STAMP_CM, STAMP_DEG = 1.0, 1.0      # нов кадър в картата чак след такова движение


class MapPose:
    """Позата, с която лепим: гладка, за да не трепери картата от шума на маркерите."""

    def __init__(self, mode):
        self.mode = mode  # "tags", "first" или "none"
        self.correction = None

    def update(self, fusion):
        if self.mode == "none":
            return fusion.odom.pose.copy()
        if fusion.latest is None:
            return None  # още няма позиция в света
        if self.mode == "first":
            return fusion.raw
        if self.correction is None:
            self.correction = fusion.latest
        pose = compose(self.correction, fusion.odom.pose)
        if fusion.tag_pose is not None:
            diff = fusion.tag_pose - pose
            diff[2] = np.radians(wrap_deg(diff[2]))
            pose = pose + BLEND * diff
            self.correction = compose(pose, inverse(fusion.odom.pose))
        return pose


def load_rig():
    if not RIG_PATH.exists():
        raise SystemExit("Няма calib/rig.json. Пусни dual.py --back <см> [--right <см>].")
    return json.loads(RIG_PATH.read_text(encoding="utf-8-sig"))


def rear_offset(rig):
    """Позата на задната камера спрямо колата; theta > 0 е наляво, а yaw в rig.json - надясно."""
    return np.array([rig["right_cm"], -rig["back_cm"], -np.radians(rig["yaw_deg"])])


def live_source(front_index, rear_index=None):
    """(време, преден кадър, заден кадър или None)"""
    cams = [CameraThread(i) for i in (front_index, rear_index) if i is not None]
    last = None
    try:
        while True:
            if len(cams) == 2:
                pair = synced_pair(*cams)
                if pair is None or pair[0] is last:
                    time.sleep(0.003)
                    continue
                last = pair[0]
                yield time.perf_counter(), pair[0], pair[1]
            else:
                frames = cams[0].snapshot()
                if not frames or frames[-1][1] is last:
                    time.sleep(0.003)
                    continue
                stamp, last = frames[-1]
                yield stamp, last, None
    finally:
        for cam in cams:
            cam.close()


def replay_source(folder, use_rear):
    rear_dir = Path(folder) / "rear"
    for stamp, frame in replay_frames(folder):
        path = rear_dir / f"{int(round(stamp * 1000)):012d}.jpg"
        rear = None
        if use_rear and path.exists():
            rear = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        yield stamp, frame, rear


def save_frame(folder, stamp, frame):
    cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(
        str(folder / f"{int(stamp * 1000):012d}.jpg"))


class MapBuilder:
    """Кадър по кадър: позиция на колата и натрупване на картата. Ползва се и от sim.py."""

    def __init__(self, camera=0, use_rear=False, mode="tags", near=40.0, far=120.0, width=100.0, scale=4.0,
                 objects=False, obstacles=False, pixels=False):
        odom_proj = FloorProjector(camera)
        view = dict(width_cm=width, near_cm=near, far_cm=far, px_per_cm=scale)
        self.view_proj = FloorProjector(camera, **view)
        self.localizer = TagLocalizer(odom_proj)
        self.map_pose = MapPose(mode)
        self.rig = self.rear_proj = odom = None
        if use_rear:
            self.rig = load_rig()
            self.rear_proj = FloorProjector(self.rig["rear_camera"], **view)
            odom = DualOdometry(odom_proj, FloorProjector(self.rig["rear_camera"]), rear_offset(self.rig))
        self.fusion = Fusion(odom_proj, self.localizer, odom)
        self.odom_projs = (odom_proj, FloorProjector(self.rig["rear_camera"]) if use_rear else None)
        self.detector = Detector() if objects else None
        if obstacles or pixels:
            from depth3d import Obstacles
        self.obstacles = Obstacles(self.odom_projs) if obstacles or pixels else None
        if pixels:
            from surfels import Surfels
        self.surfels = Surfels(self.odom_projs) if pixels else None
        self.clear()

    def clear(self):
        self.floor_map, self.path, self.last_stamp_pose, self.pose = None, [], None, None
        self.tracker, self.found_objects, self.frame_index = Tracker(), ([], []), 0
        if self.obstacles:
            self.obstacles.clear()
        if getattr(self, "surfels", None):
            self.surfels.clear()

    def _objects(self, frames, pose, stamp):
        """Засича обектите в двата кадъра и ги следи по пода в координатите на света."""
        self.found_objects = tuple([] if f is None else self.detector.detect(f) for f in frames)
        if pose is None:
            return
        seen = []
        cameras = (pose, None if self.rig is None else compose(pose, rear_offset(self.rig)))
        for objects, proj, cam_pose in zip(self.found_objects, self.odom_projs, cameras):
            for name, conf, mask in objects:
                spot = locate(mask, proj)
                if spot is not None:
                    x, y, width, height, cut = spot
                    world = cam_pose[:2] + rot(cam_pose[2]) @ (x, y)
                    seen.append((name, conf, *world, width, height, cut))
        self.tracker.update(seen, stamp)

    def _volume(self, frame, camera, pose, moving):
        """Обемните неща от кадъра; връща какво да не се лепи върху пода."""
        above = self.obstacles.observe(frame, camera, pose, moving)
        if self.surfels:
            self.surfels.observe(frame, self.obstacles.last_depth, camera, pose, moving)
        return merge(moving, above)

    def process(self, frame, rear=None, stamp=None):
        """Връща позата (x, y, theta) или None, докато не е известна."""
        lost_before = self.fusion.odom.lost
        self.fusion.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                           None if rear is None else cv2.cvtColor(rear, cv2.COLOR_BGR2GRAY))
        self.pose = pose = self.map_pose.update(self.fusion)
        self.frame_index += 1
        if self.detector:
            self._objects((frame, rear), pose, self.frame_index / 15 if stamp is None else stamp)
        skip_front, skip_rear = (exclusion(found) for found in self.found_objects)
        if pose is None:
            return None
        if self.floor_map is None:
            self.floor_map = FloorMap(self.view_proj, center=tuple(pose[:2]))
        last = self.last_stamp_pose
        moved = last is None or np.hypot(*(pose[:2] - last[:2])) >= STAMP_CM             or abs(wrap_deg(pose[2] - last[2])) >= STAMP_DEG
        if moved and self.fusion.odom.lost == lost_before:
            if self.obstacles:  # пикселите над пода не се лепят върху него
                skip_front = self._volume(frame, 0, pose, skip_front)
                if rear is not None:
                    skip_rear = self._volume(rear, 1, compose(pose, rear_offset(self.rig)), skip_rear)
            if getattr(self.fusion.odom, "source", "front") == "front":  # иначе пред нея няма под
                self.floor_map.add(frame, pose, exclude=skip_front)
            if rear is not None:
                self.floor_map.add(rear, compose(pose, rear_offset(self.rig)), self.rear_proj, REAR_PENALTY_CM,
                                   skip_rear)
            self.last_stamp_pose = pose.copy()
            self.path.append(tuple(pose[:2]))
        return pose


def merge(a, b):
    """Обединение на две маски за изключване; None значи "нищо"."""
    return b if a is None else a if b is None else np.maximum(a, b)


def marker_errors(floor_map, localizer):
    """Къде са маркерите в готовата карта спрямо измерените с рулетка места."""
    corners, ids, _ = localizer.detector.detectMarkers(cv2.cvtColor(floor_map.image, cv2.COLOR_BGR2GRAY))
    result = {}
    if ids is not None:
        for quad, marker_id in zip(corners, ids.ravel()):
            m = localizer.markers.get(int(marker_id))
            if m:
                pos = floor_map.map_to_world(quad.reshape(4, 2).mean(0))
                result[int(marker_id)] = float(np.hypot(pos[0] - m["x"], pos[1] - m["y"]))
    return result


def world_view(floor_map, pose, path, size=700):
    img, (x0, y0) = floor_map.cropped()
    to_px = lambda p: (floor_map.world_to_map(p) - (x0, y0)).astype(np.int32)
    if len(path) > 1:
        cv2.polylines(img, [to_px(np.array(path))], False, (0, 255, 0), 2)
    if pose is not None:
        tri = np.array([(0, 6), (-4, -4), (4, -4)]) @ rot(pose[2]).T + pose[:2]
        draw_car(img, to_px(tri))
    k = size / max(img.shape[:2])
    out = np.zeros((size, size, 3), np.uint8)
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)[:size, :size]
    out[:small.shape[0], :small.shape[1]] = small
    return out


def save(floor_map, stamp):
    out = c.DATA_DIR / "map"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stamp}_map.png"
    cv2.imencode(".png", floor_map.cropped()[0])[1].tofile(str(path))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--replay", help="папка със записани кадри")
    p.add_argument("--record", action="store_true")
    p.add_argument("--rear", action="store_true", help="ползвай и задната камера")
    p.add_argument("--front-only", action="store_true", help="replay без задните кадри")
    p.add_argument("--video", action="store_true", help="запиши видео камера | изглед от колата | карта")
    p.add_argument("--near", type=float, default=40, help="ползвай пода от толкова см")
    p.add_argument("--far", type=float, default=120, help="... до толкова см (по-далечното е размазано)")
    p.add_argument("--width", type=float, default=100)
    p.add_argument("--scale", type=float, default=4, help="пиксели на см")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--no-correct", dest="mode", action="store_const", const="first")
    mode.add_argument("--no-tags", dest="mode", action="store_const", const="none")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--objects", action="store_true", help="хора и играчки не влизат в картата")
    p.add_argument("--seconds", type=float, default=0)
    a = p.parse_args()
    a.mode = a.mode or "tags"

    use_rear = a.rear or bool(a.replay and (Path(a.replay) / "rear").exists() and not a.front_only)
    builder = MapBuilder(a.camera, use_rear, a.mode, a.near, a.far, a.width, a.scale, a.objects)
    rig, localizer, fusion = builder.rig, builder.localizer, builder.fusion

    stamp_name = time.strftime("%Y%m%d_%H%M%S")
    record_dir = video = None
    if a.record and not a.replay:
        record_dir = c.DATA_DIR / "odom" / stamp_name
        record_dir.mkdir(parents=True)
        if use_rear:
            (record_dir / "rear").mkdir()
        print(f"Записвам кадрите в {record_dir}")

    source = replay_source(a.replay, use_rear) if a.replay else \
        live_source(a.camera, rig["rear_camera"] if use_rear else None)
    start, frames, work, floor_map = None, 0, [], None
    for stamp, frame, rear in source:
        start = stamp if start is None else start
        if record_dir:
            save_frame(record_dir, stamp, frame)
            if rear is not None:
                save_frame(record_dir / "rear", stamp, rear)
        t0 = time.perf_counter()
        pose = builder.process(frame, rear, stamp)
        floor_map, path = builder.floor_map, builder.path
        frames += 1
        work.append((time.perf_counter() - t0) * 1000)

        if not a.headless or a.video:
            cam = cv2.resize(frame, (622, 350))
            panel = np.zeros((700, 622 + 490 + 700, 3), np.uint8)
            panel[:350, :622] = cam
            if floor_map is not None:
                car = floor_map.car_view(pose)
                car = cv2.resize(car, (int(car.shape[1] * 700 / car.shape[0]), 700))[:, :490]
                panel[:, 622:622 + car.shape[1]] = car
                panel[:, -700:] = world_view(floor_map, pose, path)
            else:
                label(panel, "show 2 markers to start", row=12)
        if a.video:
            if video is None:
                out = c.DATA_DIR / "map"
                out.mkdir(parents=True, exist_ok=True)
                video = cv2.VideoWriter(str(out / f"{stamp_name}_demo.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"), 15, (panel.shape[1], panel.shape[0]))
            video.write(panel)
        if not a.headless:
            cv2.imshow("static map: camera | car view | world", cv2.resize(panel, None, fx=0.8, fy=0.8))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                builder.clear()
                floor_map = None
            if key in (ord("s"), ord("S")) and floor_map is not None:
                print(f"Записано: {save(floor_map, time.strftime('%Y%m%d_%H%M%S'))}")
        if a.seconds and stamp - start >= a.seconds:
            break
    source.close()
    cv2.destroyAllWindows()
    if video is not None:
        video.release()

    floor_map, path = builder.floor_map, builder.path
    if floor_map is None:
        print("Няма карта: камерата трябва да види поне два маркера (или ползвай --no-tags).")
        return
    if use_rear:
        print(f"Одометрия от задната камера: {fusion.odom.rear_frames} кадъра")
    print(f"{frames} кадъра, {np.median(work):.0f} ms на кадър (медиана), залепени {len(path)}")
    print(f"Картата: {save(floor_map, stamp_name)}")
    if a.mode != "none":
        errors = marker_errors(floor_map, localizer)
        if errors:
            print("Маркерите в картата спрямо рулетката: "
                  + ", ".join(f"{i}: {e:.1f} см" for i, e in sorted(errors.items())))


if __name__ == "__main__":
    main()
