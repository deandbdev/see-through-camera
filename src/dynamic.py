"""Стъпка 8: динамичен слой. Малък YOLO-seg открива хора, животни и играчки.

Пикселите им НЕ влизат в статичната карта, а самите обекти се следят по пода и се рисуват
отделно. Ползва се с --objects в static_map.py и sim.py.

Разпознава по клас, не по движение: стоящ човек също не влиза в картата. Моделът (COCO) не
познава повечето предмети и понякога бърка (кофа = "person").
"""
import os

import cv2
import numpy as np

import common as c

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # без самоволно инсталиране на пакети

MODEL = c.DATA_DIR.parent / "models" / "yolo11n-seg.pt"
DYNAMIC = ("person", "cat", "dog", "bird", "horse", "sheep", "teddy bear", "sports ball",
           "car", "truck", "bus", "bicycle", "motorcycle", "skateboard")
MASK_CONF = 0.25     # за картата: по-добре да изрежем повече, отколкото да полепим човек
TRACK_CONF = 0.40    # за показване като обект: по-малко лъжливи
GROW_PX = 15         # маската се разширява (ръбове, сянка при краката)
MATCH_CM = 35.0      # в този радиус новото засичане е същият обект
CONFIRM = 3          # толкова засичания, преди да го покажем
FORGET_S = 3.0       # колко секунди пазим невиждан обект ("последно видян")
MOVING_CM_S = 8.0


class Detector:
    def __init__(self):
        from ultralytics import YOLO, settings

        settings.update({"sync": False})  # без изпращане на статистика
        if not MODEL.exists():
            raise SystemExit(f"Няма модел {MODEL}.")
        self.model = YOLO(str(MODEL))
        self.classes = [i for i, n in self.model.names.items() if n in DYNAMIC]

    def detect(self, frame):
        """-> [(клас, увереност, маска uint8 с размера на кадъра)]"""
        r = self.model.predict(frame, conf=MASK_CONF, classes=self.classes, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        return [(self.model.names[int(k)], float(p), m.astype(np.uint8) * 255)
                for k, p, m in zip(r.boxes.cls, r.boxes.conf, masks) if m.any()]


def exclusion(objects):
    """Обща маска на всички динамични пиксели (разширена) или None."""
    if not objects:
        return None
    mask = np.maximum.reduce([m for _, _, m in objects])
    return cv2.dilate(mask, np.ones((GROW_PX, GROW_PX), np.uint8))


def locate(mask, proj):
    """Къде стъпва обектът: (x, y спрямо камерата, ширина, височина, отрязан ли е отдолу) или None."""
    ys, xs = np.nonzero(mask)
    bottom, top = ys.max(), ys.min()
    low = xs[ys >= bottom - 8]
    foot_px = (float(np.median(low)), float(bottom))
    pts, hits = proj.pixels_to_local([foot_px, (xs.min(), bottom), (xs.max(), bottom)])
    if not hits.all():
        return None
    foot, distance = pts[0], float(np.linalg.norm(pts[0]))
    if distance > 250:
        return None
    width = float(np.clip(np.linalg.norm(pts[2] - pts[1]), 3, 60))
    height = float(np.clip(proj.height_at((float(np.median(xs[ys <= top + 8])), float(top)), distance), 3, 100))
    return foot[0], foot[1], width, height, bool(bottom >= mask.shape[0] - 3)


class Tracker:
    """Следи обектите в координатите на света: най-близкото ново засичане е същият обект."""

    def __init__(self):
        self.tracks, self.next_id, self.now = [], 1, 0.0

    def update(self, detections, now):
        """detections: [(клас, увереност, x, y, ширина, височина, отрязан)] в света."""
        self.now = now
        free = list(self.tracks)
        for name, conf, x, y, width, height, cut in sorted(detections, key=lambda d: -d[1]):
            if conf < TRACK_CONF:
                continue
            pos = np.array((x, y))
            near = [t for t in free if np.linalg.norm(t["pos"] - pos) < MATCH_CM]
            if near:
                t = min(near, key=lambda t: np.linalg.norm(t["pos"] - pos))
                free.remove(t)
                t["pos"] = 0.5 * t["pos"] + 0.5 * pos  # изглаждане на шума
                t["width"] = 0.7 * t["width"] + 0.3 * width
                t["height"] = 0.7 * t["height"] + 0.3 * height
            else:
                if any(np.linalg.norm(t["pos"] - pos) < MATCH_CM for t in self.tracks):
                    continue  # две кутии на един и същ обект
                t = dict(id=self.next_id, pos=pos, width=width, height=height, hits=0, history=[])
                self.next_id += 1
                self.tracks.append(t)
            t.update(name=name, seen=now, cut=cut, hits=t["hits"] + 1)
            t["history"] = [h for h in t["history"] if now - h[0] <= 1.0] + [(now, t["pos"].copy())]
        self.tracks = [t for t in self.tracks if now - t["seen"] <= FORGET_S]

    def visible(self):
        """[(следа, секунди откакто е видяна, движи ли се)] - само потвърдените."""
        out = []
        for t in self.tracks:
            if t["hits"] < CONFIRM:
                continue
            (t0, p0), (t1, p1) = t["history"][0], t["history"][-1]
            moving = t1 - t0 > 0.4 and np.linalg.norm(p1 - p0) / (t1 - t0) > MOVING_CM_S
            out.append((t, self.now - t["seen"], bool(moving)))
        return out
