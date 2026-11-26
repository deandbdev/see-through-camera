"""Стъпка 9: запълване на невидяния под (черно -> класически inpainting от OpenCV).

  .venv/Scripts/python.exe src/inpaint.py data/odom/<папка>              # измерва грешката
  .venv/Scripts/python.exe src/inpaint.py data/odom/<папка> --diffusion  # и на SD-Turbo

Измерване: скриваме парчета истински видян под, запълваме ги и сравняваме с истината.
"inside" = дупка, заобиколена отвсякъде; "edge" = продължение отвъд края на видяното.
Две числа: грешка в яркостта (по-малко = по-близо до истината, но мъглата печели от това) и
текстура (колко резки ръбове има спрямо истината: 1 = колкото в нея, 0 = гладко петно).
Записва data/metrics/inpaint_<време>.csv и .png с примери.

Запълва се само до REACH_CM от видяното: по-навътре би било чиста измислица.
"""
import argparse
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import common as c

REACH_CM = 25.0     # докъде извън видяното си позволяваме да запълваме
WORK_PPC = 2.0      # пиксели на см, на които смятаме (по-малко = по-бързо)
RADIUS = 4          # на cv2.inpaint, в работни пиксели
METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}
TINT = np.array((255, 80, 200), np.float32)  # измисленото се оцветява леко лилаво
TINT_SHARE = 0.3


def fill(image, seen, px_per_cm, method="telea", reach_cm=REACH_CM):
    """image и seen (bool) на цялата карта -> (запълнени пиксели, маска кои са измислени)."""
    out = np.zeros_like(image)
    invented = np.zeros(seen.shape, bool)
    if not seen.any():
        return out, invented
    ys, xs = np.nonzero(seen)
    m = int(reach_cm * px_per_cm) + 20
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, seen.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, seen.shape[1])
    img, known = image[y0:y1, x0:x1], seen[y0:y1, x0:x1]

    k = WORK_PPC / px_per_cm
    size = (max(1, round(img.shape[1] * k)), max(1, round(img.shape[0] * k)))
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # известен е само пиксел, изцяло от видяно: иначе черното се просмуква
    small_known = cv2.resize(known.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) == 255
    if not small_known.any():
        return out, invented
    away = cv2.distanceTransform((~small_known).astype(np.uint8), cv2.DIST_L2, 3)
    reach = reach_cm * WORK_PPC
    # В маската е ВСИЧКО невидяно: иначе inpaint смята черното отвън за истинско и го
    # засмуква навътре. Показваме после само това до reach.
    filled = cv2.inpaint(small, (~small_known).astype(np.uint8) * 255, RADIUS, METHODS[method])

    full = (img.shape[1], img.shape[0])
    out[y0:y1, x0:x1] = cv2.resize(filled, full, interpolation=cv2.INTER_LINEAR)
    wanted = cv2.resize(((~small_known) & (away <= reach)).astype(np.uint8), full,
                        interpolation=cv2.INTER_NEAREST) > 0
    invented[y0:y1, x0:x1] = wanted & ~known
    return out, invented


def tint(image):
    return cv2.addWeighted(image, 1 - TINT_SHARE, np.full_like(image, TINT.astype(np.uint8)), TINT_SHARE, 0)


class BackgroundFiller:
    """Смята в отделна нишка, а показването ползва последния готов резултат."""

    def __init__(self, method="telea"):
        self.method = method
        self.result = None  # (карта, метод, запълнено, запълнено с оттенък, маска uint8)
        self.ms = []
        self._thread = None

    def _work(self, key, image, seen, px_per_cm, method):
        t0 = time.perf_counter()
        filled, invented = fill(image, seen, px_per_cm, method)
        self.result = (key, method, filled, tint(filled), invented.astype(np.uint8))
        self.ms.append((time.perf_counter() - t0) * 1000)

    def update(self, floor_map, wait=False):
        """Пуска ново смятане, ако предишното е свършило. wait: чакай го (за видео)."""
        if floor_map is None or (self._thread and self._thread.is_alive()):
            return
        args = (floor_map, floor_map.image.copy(), np.isfinite(floor_map.dist), floor_map.s, self.method)
        if wait:
            self._work(*args)
        else:
            self._thread = threading.Thread(target=self._work, args=args, daemon=True)
            self._thread.start()

    def layer(self, floor_map, mark=True):
        """(изображение, маска) за тази карта или None."""
        if self.result is None or self.result[0] is not floor_map or self.result[1] != self.method:
            return None
        return self.result[3 if mark else 2], self.result[4]


def evaluate(floor_map, trials=60, seed=0, extra=None, centres=None):
    """Грешка (0-255) и текстура върху скрит истински под, по вид и размер на дупката.

    extra: {име: fn(скрито, известно) -> изображение} - още методи върху същите дупки.
    """
    rng = np.random.default_rng(seed)
    image, seen, s = floor_map.image, np.isfinite(floor_map.dist), floor_map.s
    rows, examples = [], []
    extra = extra or {}
    names = ("black", "mean", *METHODS, *extra)
    half_window = int(32 * s)  # прозорец 64 см, колкото плочката на дифузията

    def sharpness(img, where):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dx, dy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        return float(np.hypot(dx, dy)[where].mean())
    for kind in ("inside", "edge"):
        for hole_cm in (5, 10, 20):
            half = int(hole_cm * s / 2)
            # дупката и малко околност трябва да са изцяло видени
            core = cv2.erode(seen.astype(np.uint8), np.ones((2 * half + int(6 * s) + 1,) * 2, np.uint8))
            # и прозорецът около нея трябва да се побира в картата
            core[:half_window], core[-half_window:], core[:, :half_window], core[:, -half_window:] = 0, 0, 0, 0
            if centres is not None:  # дупки само на тези места
                core[~centres] = 0
            ys, xs = np.nonzero(core)
            if not len(ys):
                continue
            errors = {name: [] for name in names}
            texture = {name: [] for name in names}
            for n, i in enumerate(rng.choice(len(ys), min(trials, len(ys)), replace=False)):
                y, x = ys[i], xs[i]
                y0, x0 = y - half_window, x - half_window
                win = (slice(y0, y + half_window), slice(x0, x + half_window))
                img, known = image[win], seen[win].copy()
                hy, hx = y - y0, x - x0
                hole = np.zeros(known.shape, bool)
                hole[hy - half:hy + half, hx - half:hx + half] = True
                if kind == "inside":
                    known &= ~hole
                else:  # все едно не сме виждали нищо отвъд долния ръб на дупката (напред = нагоре)
                    known[:hy + half, :] = False
                shown = np.where(known[..., None], img, 0).astype(np.uint8)
                outputs = {"black": np.zeros_like(img),
                           "mean": np.full_like(img, img[known].mean(0).astype(np.uint8))}
                for name in METHODS:
                    outputs[name] = fill(shown, known, s, name, reach_cm=max(hole_cm, 5) + 5)[0]
                for name, method in extra.items():
                    outputs[name] = method(shown, known)
                # гледаме само пиксели, за които има истина
                scored = hole & seen[win]
                truth = img[scored].astype(np.float32)
                # без самия ръб на дупката: скокът там не е текстура
                interior = cv2.erode(scored.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                reference = max(sharpness(img, interior), 1e-6)
                for name, result in outputs.items():
                    errors[name].append(np.abs(result[scored].astype(np.float32) - truth).mean())
                    pasted = np.where(scored[..., None], result, img).astype(np.uint8)
                    texture[name].append(sharpness(pasted, interior) / reference)
                if n == 0:
                    tiles = [img, shown] + [np.where(hole[..., None], outputs[m], shown)
                                            for m in ("telea", "ns", *extra)]
                    examples.append((f"{kind} {hole_cm} cm", [cv2.resize(t.astype(np.uint8), (240, 240)) for t in tiles]))
            for name, e in errors.items():
                rows.append((kind, hole_cm, name, len(e), round(float(np.mean(e)), 1),
                             round(float(np.median(texture[name])), 2)))
    return rows, examples


def main():
    from static_map import MapBuilder, replay_source

    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--diffusion", action="store_true")
    p.add_argument("--trials", type=int, default=0, help="дупки от всеки вид (60; с дифузия 20)")
    a = p.parse_args()

    extra = {}
    if a.diffusion:
        from diffuse import Diffuser
        extra["diffusion"] = Diffuser().generate
    names = ("black", "mean", *METHODS, *extra)

    use_rear = (Path(a.folder) / "rear").exists()
    builder = MapBuilder(use_rear=use_rear)
    for _, frame, rear in replay_source(a.folder, use_rear):
        builder.process(frame, rear)
    if builder.floor_map is None:
        raise SystemExit("Няма карта от този запис.")
    rows, examples = evaluate(builder.floor_map, a.trials or (20 if a.diffusion else 60), extra=extra)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = c.DATA_DIR / "metrics" / f"inpaint_{stamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(("recording", "kind", "hole_cm", "method", "n", "mean_abs_error", "texture"))
        w.writerows((Path(a.folder).name, *row) for row in rows)

    for column, title in ((4, "Средна грешка в яркостта (0 = идеално):"), (5, "Текстура (1 = колкото истината):")):
        print(title)
        print(f"{'вид':8s}{'см':>4s}  " + "".join(f"{m:>11s}" for m in names))
        for i in range(0, len(rows), len(names)):
            group = rows[i:i + len(names)]
            print(f"{group[0][0]:8s}{group[0][1]:4d}  " + "".join(f"{r[column]:11.2f}" for r in group))

    sheet = []
    for name, tiles in examples:
        row = np.hstack(tiles)
        cv2.putText(row, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        sheet.append(row)
    columns = ("truth", "hidden", "telea", "ns", *extra)
    header = np.zeros((28, 240 * len(columns), 3), np.uint8)
    for i, text in enumerate(columns):
        cv2.putText(header, text, (i * 240 + 90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imencode(".png", np.vstack([header, *sheet]))[1].tofile(str(path.with_suffix(".png")))
    print(f"Записано: {path} (и .png с примери)")


if __name__ == "__main__":
    main()
