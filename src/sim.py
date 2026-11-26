"""Тестов стенд без количка: записът играе ролята на камерите, а изгледът е "дрон" отзад.

  .venv/Scripts/python.exe src/sim.py data/odom/<папка>
  .venv/Scripts/python.exe src/sim.py data/odom/<папка> --video --headless

Първо записът се изпълнява като на живо (картата расте). После (или с M по всяко време)
караш виртуалното камионче по построената карта.

--objects: хора и играчки се следят и се рисуват отделно (не влизат в картата).
--obstacles: шкафове, кошчета и т.н. се издигат в 3D, а не се разливат по пода.
--pixels: същите предмети с истинските пиксели вместо кубчета (surfels.py, иска видеокарта).

SPACE пауза | M запис <-> ръчно каране | W A S D каране (латиница)
I запълване: черно -> telea -> ns (-> diffusion с --diffusion) | C оцветяване на измисленото вкл./изкл.
R/F камерата нагоре/надолу | T/G по-далече/по-близо | ESC изход

Лилавият оттенък е запълнен (измислен) под, тъмното с решетка е невидяно, останалото е
истински видяно. --fill black изключва запълването (виж inpaint.py).
Всичко се приема за плосък под, затова предметите с височина изглеждат разтегнати.
"""
import argparse
import time

import cv2
import numpy as np

import common as c
from calibrate import label
from inpaint import BackgroundFiller
from motion import rot
from static_map import MapBuilder, replay_source

VIEW_W, VIEW_H, SIDE_W = 960, 540, 320
FOCAL = 0.75 * VIEW_W          # около 67 градуса хоризонтално
AHEAD_CM = 20.0                # накъде гледа виртуалната камера: толкова пред колата
SMOOTH = 0.25                  # колко бързо дронът догонва колата
DRIVE_CM, TURN_DEG = 3.0, 5.0  # на едно натискане
FILLS = ["black", "telea", "ns"]

# Камиончето в координати на колата, см: (x1, x2, y1, y2, z1, z2, цвят BGR).
# Размерите са приблизителни; началото е под предната камера.
CAR = [
    *[(x, x + 3, y, y + 6, 0, 6, (40, 40, 40)) for x in (-11, 8) for y in (-22, -5)],
    (-8, 8, -24, 2, 4, 9, (40, 160, 60)),     # шаси
    (-8, 8, -7, 2, 9, 16, (60, 190, 90)),     # кабина
    (-8, 8, -22, -9, 9, 11, (170, 170, 170)),  # лаптопът
]
FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


class ChaseCamera:
    def __init__(self, back_cm=80.0, height_cm=75.0):
        self.back, self.height = back_cm, height_cm
        self.pose = None

    def follow(self, car_pose, smooth=SMOOTH):
        if self.pose is None:
            self.pose = car_pose.copy()
        diff = car_pose - self.pose
        diff[2] = (diff[2] + np.pi) % (2 * np.pi) - np.pi
        self.pose = self.pose + smooth * diff

    def matrices(self):
        """R, позиция C и вътрешна матрица K на виртуалната камера в света (z нагоре)."""
        forward = rot(self.pose[2]) @ (0.0, 1.0)
        C = np.array([*(self.pose[:2] - forward * self.back), self.height])
        target = np.array([*(self.pose[:2] + forward * AHEAD_CM), 0.0])
        f = (target - C) / np.linalg.norm(target - C)
        r = np.cross(f, (0.0, 0.0, 1.0))
        r /= np.linalg.norm(r)
        R = np.array([r, np.cross(f, r), f])  # надясно, надолу, напред
        K = np.array([[FOCAL, 0, VIEW_W / 2], [0, FOCAL, VIEW_H / 2], [0, 0, 1.0]])
        return R, C, K


def draw_grid(img, cam, step=20.0, reach=300.0):
    R, C, K = cam.matrices()
    cx, cy = np.round(cam.pose[:2] / step) * step
    for axis in (0, 1):
        for offset in np.arange(-reach, reach + 1, step):
            t = np.arange(-reach, reach + 1, step)
            line = np.zeros((len(t), 3))
            line[:, axis] = t + (cx, cy)[axis]
            line[:, 1 - axis] = offset + (cx, cy)[1 - axis]
            pts = (line - C) @ R.T
            px = pts @ K.T
            px = px[:, :2] / np.maximum(px[:, 2:], 1e-6)
            for i in range(len(t) - 1):
                if pts[i, 2] > 5 and pts[i + 1, 2] > 5:  # пред камерата
                    cv2.line(img, px[i].astype(int), px[i + 1].astype(int), (55, 55, 55), 1, cv2.LINE_AA)


def render_floor(floor_map, cam, layer=None):
    """Картата е равнина, затова всяка гледна точка към нея е една хомография.

    layer: (запълнено изображение, маска) от inpaint.py; истинският под винаги е отгоре.
    """
    R, C, K = cam.matrices()
    to_img = K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])  # (x, y, 1) на пода -> пиксел
    out = np.full((VIEW_H, VIEW_W, 3), 28, np.uint8)
    draw_grid(out, cam)
    if floor_map is not None:
        s = floor_map.s
        to_world = np.array([[1 / s, 0, floor_map.origin[0] + 0.5 / s],  # пиксел от картата -> см
                             [0, -1 / s, floor_map.origin[1] - 0.5 / s],
                             [0, 0, 1.0]])
        H = to_img @ to_world
        if layer is not None:
            filled = cv2.warpPerspective(layer[0], H, (VIEW_W, VIEW_H), flags=cv2.INTER_LINEAR)
            where = cv2.warpPerspective(layer[1], H, (VIEW_W, VIEW_H), flags=cv2.INTER_NEAREST)
            cv2.copyTo(filled, where, out)
        color = cv2.warpPerspective(floor_map.image, H, (VIEW_W, VIEW_H), flags=cv2.INTER_LINEAR)
        seen = cv2.warpPerspective(np.isfinite(floor_map.dist).astype(np.uint8), H, (VIEW_W, VIEW_H),
                                   flags=cv2.INTER_NEAREST)
        cv2.copyTo(color, seen, out)
    horizon = int(VIEW_H / 2 - FOCAL * -R[2, 2] / np.hypot(R[2, 0], R[2, 1]))  # над него няма под
    if horizon > 0:
        out[:horizon] = 15
    return out


def draw_boxes(img, cam, boxes):
    """boxes: [(8 ъгъла в света, цвят)]; стените се рисуват от далечните към близките."""
    R, C, K = cam.matrices()
    faces = []
    for corners, color in boxes:
        pts = (corners - C) @ R.T
        if pts[:, 2].min() < 1:
            continue
        px = pts @ K.T
        px = (px[:, :2] / px[:, 2:]).astype(np.int32)
        center = corners.mean(0)
        for i, face in enumerate(FACES):
            mid = corners[list(face)].mean(0)
            if (mid - center) @ (C - mid) <= 0:
                continue  # задна страна
            shade = (0.7, 0.8, 0.9, 0.6, 0.45, 1.0)[i]
            faces.append((np.linalg.norm(mid - C), px[list(face)], [int(v * shade) for v in color]))
    for _, px, color in sorted(faces, key=lambda f: -f[0]):  # далечните първи
        cv2.fillConvexPoly(img, px, color, cv2.LINE_AA)
        cv2.polylines(img, [px], True, (0, 0, 0), 1, cv2.LINE_AA)


def box_corners(x1, x2, y1, y2, z1, z2):
    return np.array([(x, y, z) for x in (x1, x2) for y in (y1, y2) for z in (z1, z2)], np.float64)


def truck_boxes(car_pose):
    boxes = []
    for *size, color in CAR:
        corners = box_corners(*size)
        corners[:, :2] = corners[:, :2] @ rot(car_pose[2]).T + car_pose[:2]
        boxes.append((corners, color))
    return boxes


def draw_objects(img, cam, tracker):
    """Следените обекти: червена кутия, докато се виждат; само контур - ако са по памет."""
    R, C, K = cam.matrices()
    for track, age, moving in tracker.visible():
        half, (x, y) = track["width"] / 2, track["pos"]
        corners = box_corners(x - half, x + half, y - half, y + half, 0, track["height"])
        pts = (corners - C) @ R.T
        if pts[:, 2].min() < 1:
            continue
        px = pts @ K.T
        px = (px[:, :2] / px[:, 2:]).astype(np.int32)
        if age < 0.3:
            solid = img.copy()
            draw_boxes(solid, cam, [(corners, (40, 40, 230))])
            cv2.addWeighted(solid, 0.45, img, 0.55, 0, img)  # да не закрива пода зад себе си
            text = f"{track['name']} {'moving' if moving else ''}"
        else:
            for face in FACES:
                cv2.polylines(img, [px[list(face)]], True, (90, 90, 200), 1, cv2.LINE_AA)
            text = f"{track['name']}? last seen {age:.0f}s"
        top = px[1::2].min(0)
        cv2.putText(img, text, (int(top[0]), int(top[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 200, 255), 1, cv2.LINE_AA)


def top_view(builder, pose, size=(SIDE_W, VIEW_H - 2 * 180)):
    out = np.zeros((size[1], size[0], 3), np.uint8)
    if builder.floor_map is None:
        return out
    img, (x0, y0) = builder.floor_map.cropped()
    to_px = lambda p: (builder.floor_map.world_to_map(p) - (x0, y0)).astype(np.int32)
    if len(builder.path) > 1:
        cv2.polylines(img, [to_px(np.array(builder.path))], False, (0, 255, 0), 3)
    tri = np.array([(0, 10), (-7, -7), (7, -7)]) @ rot(pose[2]).T + pose[:2]
    cv2.fillPoly(img, [to_px(tri)], (0, 200, 255))
    for track, age, _ in builder.tracker.visible():
        cv2.circle(img, to_px(track["pos"]), 8, (40, 40, 230), -1 if age < 0.3 else 2)
    k = min(size[0] / img.shape[1], size[1] / img.shape[0])
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)[:size[1], :size[0]]
    out[:small.shape[0], :small.shape[1]] = small
    return out


def draw_scene(view, builder, cam, car_pose):
    """Предмети и камионче в правилен ред: зад камиончето, то самото, пред него."""
    # по-близкото до камерата се рисува последно
    draw_objects(view, cam, builder.tracker)
    if builder.obstacles is None:
        draw_boxes(view, cam, truck_boxes(car_pose))
        return
    if builder.surfels is None:
        from depth3d import draw_voxels as draw
        xyz, colors = builder.obstacles.voxels()
    else:
        from surfels import draw_points as draw
        xyz, colors = builder.surfels.cloud(every=8)
    matrices = cam.matrices()
    R, C, _ = matrices
    truck = float((np.array([*car_pose[:2], 8.0]) - C) @ R[2])
    draw(view, matrices, FOCAL, xyz, colors, near_cm=truck)
    draw_boxes(view, cam, truck_boxes(car_pose))
    # твърде близките до дрона биха закрили всичко
    draw(view, matrices, FOCAL, xyz, colors, near_cm=0.55 * truck, far_cm=truck - 15)


def compose_screen(builder, cam, car_pose, front, rear, text, layer=None):
    screen = np.zeros((VIEW_H, VIEW_W + SIDE_W, 3), np.uint8)
    if car_pose is not None:
        view = render_floor(builder.floor_map, cam, layer)
        draw_scene(view, builder, cam, car_pose)
        screen[:, :VIEW_W] = view
        screen[360:, VIEW_W:] = top_view(builder, car_pose)
    label(screen, text)
    for row, (image, name) in enumerate(((front, "front camera"), (rear, "rear camera"))):
        if image is not None:
            found = builder.found_objects[row]
            if found:
                image = image.copy()
                for _, _, mask in found:
                    outline = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.drawContours(image, outline, -1, (0, 0, 255), 6)
            small = cv2.resize(image, (SIDE_W, 180))
            cv2.putText(small, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            screen[row * 180:(row + 1) * 180, VIEW_W:] = small
    return screen


def drive(pose, key):
    """Нова поза след клавиш W/A/S/D; theta > 0 е наляво."""
    pose = pose.copy()
    if key in "ws":
        pose[:2] += rot(pose[2]) @ (0.0, DRIVE_CM if key == "w" else -DRIVE_CM)
    elif key in "ad":
        pose[2] += np.radians(TURN_DEG if key == "a" else -TURN_DEG)
    return pose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="папка със записани кадри")
    p.add_argument("--front-only", action="store_true")
    p.add_argument("--no-tags", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--objects", action="store_true", help="динамичен слой (YOLO)")
    p.add_argument("--obstacles", action="store_true", help="обемни предмети (дълбочина, depth3d.py)")
    p.add_argument("--pixels", action="store_true", help="предметите с истински пиксели (surfels.py)")
    p.add_argument("--fill", choices=FILLS, default="telea")
    p.add_argument("--diffusion", action="store_true", help="дорисуване с SD-Turbo на заден план")
    p.add_argument("--headless", action="store_true")
    a = p.parse_args()

    from pathlib import Path
    use_rear = (Path(a.folder) / "rear").exists() and not a.front_only
    builder = MapBuilder(use_rear=use_rear, mode="none" if a.no_tags else "tags", objects=a.objects,
                         obstacles=a.obstacles, pixels=a.pixels)
    cam = ChaseCamera()
    fill_mode, mark = a.fill, True
    classic = BackgroundFiller()
    fills, diffusion = list(FILLS), None
    if a.diffusion:
        from diffuse import Diffuser, DiffusionFiller
        print("Зареждам модела...")
        diffusion = DiffusionFiller(Diffuser())
        fills.append("diffusion")
        fill_mode = "diffusion"
    source = replay_source(a.folder, use_rear)
    name = time.strftime("%Y%m%d_%H%M%S")
    video = None
    if a.video:
        out = c.DATA_DIR / "map"
        out.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(out / f"{name}_sim.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                15, (VIEW_W + SIDE_W, VIEW_H))

    manual = paused = finished = False
    ticks = 0
    car = front = rear = last_stamp = None
    while True:
        started = time.perf_counter()
        wait = 0.03
        if not manual and not paused and not finished:
            item = next(source, None)
            if item is None:
                finished = True
                if a.headless or builder.pose is None:
                    break
                manual = True
            else:
                stamp, front, rear = item
                pose = builder.process(front, rear, stamp)
                car = pose if pose is not None else car
                if last_stamp is not None:
                    wait = min(stamp - last_stamp, 0.2)  # темпото на записа
                last_stamp = stamp

        if car is not None:
            cam.follow(car, SMOOTH if not manual else 0.4)
        text = ("manual: W A S D | M replay" if manual else "paused" if paused else "replay | M drive") \
            if car is not None else "show 2 markers to start"
        layer = None
        if fill_mode == "diffusion":
            diffusion.focus = None if car is None else car[:2].copy()
            ticks += 1
            if not a.headless or ticks % 8 == 0:  # във видеото чакаме плочката, затова по-рядко
                diffusion.update(builder.floor_map, wait=a.headless, tiles=1 if a.headless else 3)
            layer = diffusion.layer(builder.floor_map, mark)
        elif fill_mode != "black":
            classic.method = fill_mode
            classic.update(builder.floor_map, wait=a.headless)
            layer = classic.layer(builder.floor_map, mark)
        screen = compose_screen(builder, cam, car, None if manual else front, None if manual else rear,
                                f"{text} | fill: {fill_mode}", layer)
        if video is not None:
            video.write(screen)
        if a.headless:
            continue

        cv2.imshow("sim", screen)
        left = wait - (time.perf_counter() - started)
        code = cv2.waitKey(max(1, int(left * 1000)))
        key = chr(code & 0xFF).lower() if code >= 0 else ""
        if code & 0xFF == 27 or cv2.getWindowProperty("sim", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == " ":
            paused = not paused
        elif key == "i":
            fill_mode = fills[(fills.index(fill_mode) + 1) % len(fills)]
        elif key == "c":
            mark = not mark
        elif key == "m" and car is not None:
            manual = not manual
            if not manual:
                car = builder.pose if builder.pose is not None else car
        elif manual and key and key in "wasd":
            car = drive(car, key)
        elif key and key in "rf":
            cam.height = float(np.clip(cam.height + (10 if key == "r" else -10), 25, 250))
        elif key and key in "tg":
            cam.back = float(np.clip(cam.back + (10 if key == "t" else -10), 10, 200))

    cv2.destroyAllWindows()
    for title, filler in (("OpenCV", classic), ("SD-Turbo, на плочка", diffusion)):
        if filler is not None and filler.ms:
            print(f"Запълване {title}: {np.median(filler.ms):.0f} ms (медиана, на заден план), {len(filler.ms)} пъти")
    if video is not None:
        video.release()
        print(f"Видео: {c.DATA_DIR / 'map' / f'{name}_sim.mp4'}")
    if builder.floor_map is None:
        print("Няма карта: камерата трябва да види поне два маркера (или ползвай --no-tags).")


if __name__ == "__main__":
    main()
