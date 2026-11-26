"""Синхронно четене от няколко камери.

Уеб камерите не се тригерират хардуерно: всяка се чете в отделна нишка, кадрите се
пазят с времето на получаване и се избира двойката с най-близки времена.
"""
import collections
import threading
import time

import common as c


class CameraThread:
    def __init__(self, index, history=4):
        self.index = index
        self.cap = c.open_camera(index)
        self.frames = collections.deque(maxlen=history)  # (време, кадър)
        self.count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self.frames.append((time.perf_counter(), frame))
                self.count += 1

    def snapshot(self):
        with self._lock:
            return list(self.frames)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()


def synced_pair(a, b):
    """Скорошна двойка с минимална разлика: (кадър_a, кадър_b, разлика в сек) или None."""
    fa, fb = a.snapshot(), b.snapshot()
    if not fa or not fb:
        return None
    # водеща е по-изоставащата камера: най-новият ѝ кадър търси най-близкия от другата
    if fa[-1][0] <= fb[-1][0]:
        ta, tb = fa[-1], min(fb, key=lambda f: abs(f[0] - fa[-1][0]))
    else:
        ta, tb = min(fa, key=lambda f: abs(f[0] - fb[-1][0])), fb[-1]
    return ta[1], tb[1], abs(ta[0] - tb[0])
