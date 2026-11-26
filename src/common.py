"""Общи неща за калибрацията: камера, дъска, пиксели <-> сантиметри на пода."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "calib"
DATA_DIR = ROOT / "data"

PATTERN = (9, 6)      # вътрешни ъгли
SQUARE_CM = 2.5       # страна на квадрата
SIZE = (1280, 720)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Камера {index} не дава кадри (заета ли е от друга програма?)")
    if (frame.shape[1], frame.shape[0]) != SIZE:
        raise SystemExit(f"Камера {index} дава {frame.shape[1]}x{frame.shape[0]}, а не {SIZE[0]}x{SIZE[1]}")
    return cap


def board_points_cm(square_cm=SQUARE_CM):
    """Ъглите в координатите на дъската, (N, 2), ред по ред."""
    cols, rows = PATTERN
    return np.array([(c, r) for r in range(rows) for c in range(cols)], np.float32) * square_cm


def find_corners(gray, fast=False):
    """Връща (N, 2) пиксели или None. fast=True е за live преглед."""
    if fast:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, PATTERN, flags=flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    else:
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 2).astype(np.float32) if found else None


def calibrate_intrinsics(views, square_cm=SQUARE_CM):
    """views: списък от (N, 2) пиксели. Връща (rms, K, dist, грешка за всеки кадър)."""
    obj = np.zeros((len(board_points_cm()), 3), np.float32)
    obj[:, :2] = board_points_cm(square_cm)
    out = cv2.calibrateCameraExtended([obj] * len(views), views, SIZE, None, None)
    rms, K, dist, per_view = out[0], out[1], out[2], out[7]
    return rms, K, dist, per_view.ravel()


def undistort_points(pts, K, dist):
    return cv2.undistortPoints(np.asarray(pts, np.float32).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)


def floor_homography(corners, K, dist, square_cm=SQUARE_CM):
    """Хомография: неизкривени пиксели -> сантиметри на пода (в системата на дъската)."""
    H, _ = cv2.findHomography(undistort_points(corners, K, dist), board_points_cm(square_cm))
    return H


def pixels_to_floor(pts, K, dist, H):
    """Сурови (изкривени) пиксели -> сантиметри на пода."""
    und = undistort_points(pts, K, dist)
    return cv2.perspectiveTransform(und.reshape(-1, 1, 2), H).reshape(-1, 2)


def board_errors(floor_pts, square_cm=SQUARE_CM):
    """Измерени срещу истински разстояния върху дъска, проектирана на пода.

    Връща списък от (име, истина см, измерено см).
    """
    g = floor_pts.reshape(PATTERN[1], PATTERN[0], 2)
    cols, rows = PATTERN
    width, height = (cols - 1) * square_cm, (rows - 1) * square_cm
    mean_len = lambda v: float(np.linalg.norm(v, axis=-1).mean())
    return [
        ("квадрат", square_cm, mean_len(np.concatenate([np.diff(g, axis=1).reshape(-1, 2),
                                                        np.diff(g, axis=0).reshape(-1, 2)]))),
        ("ширина", width, mean_len(g[:, -1] - g[:, 0])),
        ("височина", height, mean_len(g[-1] - g[0])),
        ("диагонал", float(np.hypot(width, height)), mean_len(np.array([g[-1, -1] - g[0, 0],
                                                                       g[-1, 0] - g[0, -1]]))),
    ]
