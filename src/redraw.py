"""Опит A: генеративен модел "прерисува" нашия дрон изглед, и мерим колко лъже.

  .venv/Scripts/python.exe src/redraw.py --preview     # един кадър с трите сили
  .venv/Scripts/python.exe src/redraw.py               # видео и метрики

SD-Turbo получава нашия кадър (с малко от предишния си изход, за да трепти по-малко), добавя
шум и го "изчиства". Колкото по-голяма е силата, толкова повече си измисля.
Мерим по ArUco маркерите: четат ли се още и къде на пода се оказват (спрямо рулетката).
Това НЕ е методът на Tesla: моделът не гледа камерите, а нашата картина.
"""
import argparse
import csv
import time

import cv2
import numpy as np

import common as c
import sim
from diffuse import Diffuser
from static_map import MapBuilder, replay_source

RECORDING = "20261125_134157"
FIRST, LAST, MAP_FROM = 130, 560, 118      # отрязък без хора в кадър
W, H = 768, 432
VIDEO_FPS = 10
STRENGTHS = (("weak", 0.3), ("medium", 0.5), ("strong", 0.7))
STEPS = 2
PREVIOUS = 0.3                             # дял от предишния изход в началната картина
PROMPT = "realistic photo of a kitchen floor and white cabinets seen from above and behind, sharp, detailed"
BACKGROUND = (24, 20, 20)


class Redrawer:
    def __init__(self):
        self.m = Diffuser(prompt=PROMPT)
        torch = self.m.torch
        gen = torch.Generator("cpu").manual_seed(1)
        # един и същ шум за всички кадри: по-малко трептене
        self.noise = torch.randn((1, 4, H // 8, W // 8), generator=gen).to("cuda", torch.float16)

    def __call__(self, image, strength):
        torch = self.m.torch
        with torch.no_grad():
            x = torch.from_numpy(image[:, :, ::-1].copy()).to("cuda", torch.float16)
            x = x.permute(2, 0, 1)[None] / 127.5 - 1
            latent = self.m.vae.encode(x).latent_dist.mean * self.m.vae.config.scaling_factor
            for t in np.linspace(int(999 * strength), 0, STEPS + 1)[:-1].round().astype(int):
                a = self.m.alphas[int(t)].item()
                noisy = a ** 0.5 * latent + (1 - a) ** 0.5 * self.noise
                eps = self.m.unet(noisy, int(t), encoder_hidden_states=self.m.text).sample
                latent = (noisy - (1 - a) ** 0.5 * eps) / a ** 0.5
            out = self.m.vae.decode(latent / self.m.vae.config.scaling_factor).sample
            out = ((out[0].permute(1, 2, 0).float() + 1) * 127.5).clamp(0, 255)
        return out.cpu().numpy()[:, :, ::-1].astype(np.uint8).copy()


def find_markers(image, camera, localizer):
    """Маркерите в дрон изгледа -> {id: (x, y) на пода}; виртуалната камера е известна."""
    R, C, K = camera.matrices()
    to_floor = np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], -R @ C]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = localizer.detector.detectMarkers(gray)
    found = {}
    for quad, marker_id in zip(corners, [] if ids is None else ids.ravel()):
        p = to_floor @ np.r_[quad.reshape(4, 2).mean(0), 1.0]
        found[int(marker_id)] = p[:2] / p[2]
    return found


def source_frames():
    """(кадър от нашия дрон изглед, виртуалната камера, локализаторът) на VIDEO_FPS."""
    sim.VIEW_W, sim.VIEW_H, sim.FOCAL, sim.AHEAD_CM = W, H, 0.75 * W, 30.0
    plain = np.full((H, W, 3), BACKGROUND, np.uint8)
    sim.draw_grid = lambda img, cam: img.__setitem__(slice(None), plain)
    builder = MapBuilder(obstacles=True)
    camera = sim.ChaseCamera(75.0, 95.0)
    next_time = None
    for i, (stamp, frame, _) in enumerate(replay_source(c.DATA_DIR / "odom" / RECORDING, False)):
        if i >= LAST:
            break
        if i == MAP_FROM:
            builder.clear()
        pose = builder.process(frame, None, stamp)
        if pose is None:
            continue
        camera.follow(pose, 0.15)
        if i < FIRST:
            continue
        next_time = stamp if next_time is None else next_time
        if stamp < next_time:
            continue
        next_time += 1.0 / VIDEO_FPS
        view = sim.render_floor(builder.floor_map, camera)
        sim.draw_scene(view, builder, camera, pose)
        yield view, camera, builder.localizer


def titled(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 12 + 16, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()

    model = Redrawer()
    out_dir = c.DATA_DIR / "map"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    video = None
    previous, before_source = {}, None
    names = ("ours", *(n for n, _ in STRENGTHS))
    stats = {n: dict(seen=0, kept=0, errors=[], false=0, invented=[], change=[], flicker=[]) for n in names}
    frames = 0
    for n, (view, camera, localizer) in enumerate(source_frames()):
        if a.preview and n != 40:
            continue
        frames += 1
        truth = {i: np.array((m["x"], m["y"])) for i, m in localizer.markers.items()}
        empty = (np.abs(view.astype(int) - BACKGROUND).sum(-1) < 6)  # където ние нямаме нищо
        images = {"ours": view}
        for name, strength in STRENGTHS:
            start = view if name not in previous else cv2.addWeighted(view, 1 - PREVIOUS, previous[name], PREVIOUS, 0)
            images[name] = model(start, strength)
        baseline = find_markers(view, camera, localizer)
        for name, image in images.items():
            s = stats[name]
            found = find_markers(image, camera, localizer)
            s["seen"] += len(baseline)
            for marker_id, position in found.items():
                if marker_id in baseline:
                    s["kept"] += 1
                    s["errors"].append(float(np.linalg.norm(position - truth[marker_id])))
                elif marker_id not in truth:
                    s["false"] += 1
            if name != "ours":
                s["invented"].append(float((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[empty] > 55).mean()))
                s["change"].append(float(np.abs(image.astype(int) - view.astype(int))[~empty].mean()))
            if name in previous and before_source is not None:
                s["flicker"].append(float(np.abs(image.astype(int) - previous[name].astype(int)).mean()))
        previous, before_source = images, view

        top = np.hstack([titled(images["ours"], "our view (measured geometry)"), titled(images["weak"], "AI redraw 0.3")])
        bottom = np.hstack([titled(images["medium"], "AI redraw 0.5"), titled(images["strong"], "AI redraw 0.7")])
        sheet = np.vstack([top, bottom])
        if a.preview:
            cv2.imencode(".png", sheet)[1].tofile(str(out_dir / "redraw_preview.png"))
            print("Записано: data/map/redraw_preview.png")
            return
        if video is None:
            video = cv2.VideoWriter(str(out_dir / f"{stamp}_redraw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    VIDEO_FPS, (2 * W, 2 * H))
        video.write(sheet)
        if frames in (20, 90, 170):
            cv2.imencode(".png", sheet)[1].tofile(str(out_dir / f"{stamp}_redraw_{frames}.png"))
    video.release()

    path = c.DATA_DIR / "metrics" / f"redraw_{stamp}.csv"
    rows = []
    for name in names:
        s = stats[name]
        mean = lambda key: round(float(np.mean(s[key])), 1) if s[key] else ""
        rows.append((name, frames, s["seen"], s["kept"], round(100 * s["kept"] / max(1, s["seen"])),
                     round(float(np.median(s["errors"])), 1) if s["errors"] else "",
                     round(float(np.percentile(s["errors"], 90)), 1) if s["errors"] else "", s["false"],
                     round(100 * float(np.mean(s["invented"])), 1) if s["invented"] else "",
                     mean("change"), mean("flicker")))
    header = ("version", "frames", "markers_in_ours", "markers_still_read", "read_percent", "position_error_median_cm",
              "position_error_p90_cm", "false_ids", "empty_area_filled_percent", "pixel_change", "flicker")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    for row in rows:
        print(dict(zip(header, row)))
    print(f"Видео: {out_dir / f'{stamp}_redraw.mp4'}")
    print(f"Метрики: {path}")


if __name__ == "__main__":
    main()
