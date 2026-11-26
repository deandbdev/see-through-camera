"""Стъпка 1: проверка на Python, OpenCV (ArUco), PyTorch/CUDA и камерите.

Пускане:  py -3.10 src/check_env.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "check_env"
MAX_INDEX = 4        # проверяваме камери с индекс 0..3
MEASURE_SECONDS = 3  # колко дълго мерим реалните FPS


def check_opencv():
    print(f"Python  {sys.version.split()[0]}  ({sys.executable})")
    print(f"NumPy   {np.__version__}")
    print(f"OpenCV  {cv2.__version__}")
    print(f"  ArUco detector: {'да' if hasattr(cv2, 'aruco') and hasattr(cv2.aruco, 'ArucoDetector') else 'НЕ'}")


def check_torch():
    try:
        import torch
    except ImportError as e:
        print(f"PyTorch: липсва ({e})")
        return
    print(f"PyTorch {torch.__version__}")
    if not torch.cuda.is_available():
        print("  CUDA: НЕ")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"  CUDA: да, {props.name}, {props.total_memory / 1024**3:.1f} GB")
    # Реално смятане на GPU, включително fp16
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    print(f"  fp16 умножение на матрици: {'ок' if torch.isfinite(x @ x).all() else 'ГРЕШКА'}")
    try:
        import torchvision
        from torchvision.ops import nms
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
        nms(boxes, torch.tensor([1.0], device="cuda"), 0.5)
        print(f"  torchvision {torchvision.__version__}: GPU операции ок")
    except Exception as e:
        print(f"  torchvision: ГРЕШКА: {str(e).splitlines()[0]}")


def check_camera(index):
    # MSMF, не DSHOW: с DSHOW вградената камера дава само 10 FPS на 720p
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    try:
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        for _ in range(15):  # загряване: авто-експозиция
            cap.read()

        frames, frame = 0, None
        start = time.perf_counter()
        while time.perf_counter() - start < MEASURE_SECONDS:
            ok, f = cap.read()
            if ok:
                frames, frame = frames + 1, f
        fps = frames / (time.perf_counter() - start)

        if frame is None:
            print(f"  камера {index}: отваря се, но НЕ дава кадри")
            return False

        h, w = frame.shape[:2]
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"camera{index}.jpg"
        cv2.imencode(".jpg", frame)[1].tofile(str(path))
        print(f"  камера {index}: {w}x{h}, {fps:.1f} FPS, средна яркост {frame.mean():.0f}/255 -> {path}")
        return True
    finally:
        cap.release()


if __name__ == "__main__":
    check_opencv()
    check_torch()
    print("Камери:")
    found = [i for i in range(MAX_INDEX) if check_camera(i)]
    print(f"Работещи камери: {found or 'няма'}")
