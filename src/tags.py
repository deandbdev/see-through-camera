"""ArUco маркери на пода -> абсолютна поза на колата.

Свят (см): координатите са тези на центровете в calib/markers.json (измерени с рулетка).
yaw (завъртане на листа, градуси, наляво > 0) може да е null: научава се, щом камерата види
поне два маркера едновременно, тогава позата идва само от измерените центрове.
"""
import json

import cv2
import numpy as np

import common as c
from motion import rigid_fit, rot

MAP_PATH = c.CALIB_DIR / "markers.json"
DEFAULT_MAP = {
    "size_cm": 12.0,
    "markers": {
        "0": {"x": 0, "y": 0, "yaw": None},
        "1": {"x": 50, "y": 0, "yaw": None},
        "2": {"x": 0, "y": 60, "yaw": None},
        "3": {"x": 50, "y": 60, "yaw": None},
    },
}
SIZE_TOLERANCE = 0.15  # измерената от камерата страна срещу истинската
MIN_YAW_SAMPLES = 15


class TagLocalizer:
    def __init__(self, projector, path=MAP_PATH):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(DEFAULT_MAP, indent=2))
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        self.path, self.data = path, data
        self.proj = projector
        self.size = float(data["size_cm"])
        self.markers = {int(k): v for k, v in data["markers"].items()}
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        self.yaw_samples = {}
        self.distances = {}  # (i, j) -> разстояния между центрове, мерени от камерата
        self.side_ratios = []  # измерена / истинска страна: проверка на калибрацията на пода
        h = self.size / 2
        # редът на ъглите в OpenCV: горе-ляво, горе-дясно, долу-дясно, долу-ляво
        self.square = np.array([(-h, h), (h, h), (h, -h), (-h, -h)])

    def detect(self, gray):
        """{id: (4 пиксела, 4 точки на пода спрямо колата)} за годните маркери от картата."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = {}
        if ids is None:
            return found
        for quad, marker_id in zip(corners, ids.ravel()):
            marker_id = int(marker_id)
            if marker_id not in self.markers:
                continue
            px = quad.reshape(4, 2)
            floor, hits = self.proj.pixels_to_local(px)
            if not hits.all():
                continue
            side = np.linalg.norm(floor - np.roll(floor, -1, axis=0), axis=1).mean()
            self.side_ratios.append(float(side / self.size))
            if abs(side / self.size - 1) <= SIZE_TOLERANCE:
                found[marker_id] = (px, floor)
        return found

    def world_corners(self, marker_id):
        m = self.markers[marker_id]
        return self.square @ rot(np.radians(m["yaw"])).T + (m["x"], m["y"])

    def locate(self, found):
        """Поза (x, y, theta) на колата в света или None.

        Нужни са маркер с известен yaw или поне два маркера едновременно.
        """
        known = [i for i in found if self.markers[i]["yaw"] is not None]
        for i in found:
            for j in found:
                if i < j:
                    d = np.linalg.norm(found[i][1].mean(0) - found[j][1].mean(0))
                    self.distances.setdefault((i, j), []).append(float(d))
        if len(found) >= 2 and len(known) < len(found):
            # има маркер с неизвестно завъртане: позата идва от центровете
            theta, t = rigid_fit(np.array([found[i][1].mean(0) for i in found]),
                                 np.array([(self.markers[i]["x"], self.markers[i]["y"]) for i in found], float))
        elif known:
            theta, t = rigid_fit(np.concatenate([found[i][1] for i in known]),
                                 np.concatenate([self.world_corners(i) for i in known]))
        else:
            return None
        self._learn_yaw(found, theta)
        return np.array([t[0], t[1], theta])

    def _learn_yaw(self, found, theta):
        for i, (_, floor) in found.items():
            if self.markers[i]["yaw"] is not None:
                continue
            world = (floor - floor.mean(0)) @ rot(theta).T
            yaw, _ = rigid_fit(self.square, world)
            samples = self.yaw_samples.setdefault(i, [])
            samples.append(yaw)
            if len(samples) >= MIN_YAW_SAMPLES:
                mean = np.arctan2(np.sin(samples).mean(), np.cos(samples).mean())
                self.markers[i]["yaw"] = round(float(np.degrees(mean)), 2)
                print(f"Маркер {i}: научено завъртане {self.markers[i]['yaw']:+.1f}°")

    def distance_report(self):
        """Редове: разстояние между центрове по рулетка срещу по камера."""
        lines = []
        for (i, j), values in sorted(self.distances.items()):
            a, b = self.markers[i], self.markers[j]
            true = float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))
            got = float(np.median(values))
            lines.append(f"  маркери {i}-{j}: рулетка {true:.1f} см, камера {got:.1f} см "
                         f"({(got - true) / true * 100:+.1f}%, {len(values)} кадъра)")
        return lines
