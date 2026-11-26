"""Обучаваме малката мрежа (nanofix.py) върху наши данни: рендер -> истински кадър.

  .venv/Scripts/python.exe src/nf_train.py data    # прави двойките
  .venv/Scripts/python.exe src/nf_train.py train   # обучава (спира да почине, ако картата прегрее)
  .venv/Scripts/python.exe src/nf_train.py eval    # мери върху записа, който не е участвал

Двойка: кадрите на записа се делят на три групи по блокове; за всяка група картата (под и точки)
се строи само от другите две и се рисува гледната точка на скритите кадри. Целта е истинският кадър,
не изходът на друг модел. Теглата се обучават от нулата (не се ползва тяхното nf.pt).
Мрежата научава ТАЗИ кухня; на други места вероятно ще се проваля.
"""
import argparse
import csv
import json
import subprocess
import time

import cv2
import numpy as np

import common as c

TRAIN = ("20261125_134822", "20261125_135428", "20261125_135744")
TEST = "20261125_134157"
SIZE = (768, 432)
BACKGROUND = (24, 20, 20)
FOLDS = 3
DATA = c.DATA_DIR / "nf_train"
WEIGHTS = c.DATA_DIR.parent / "models" / "neural-fixer" / "nf_kitchen.pt"
STEPS, BATCH, CROP = 4000, 8, (384, 256)
HOT_C, REST_S = 87, 30     # над тази температура: кратка почивка (не чакаме да изстине напълно)


def make_pairs(recording):
    import sim
    import surfels
    from difix_eval import Viewpoint
    from mapper import FloorMap
    from static_map import REAR_PENALTY_CM, MapBuilder, replay_source

    folder = c.DATA_DIR / "odom" / recording
    use_rear = (folder / "rear").exists()
    builder = MapBuilder(use_rear=use_rear, obstacles=True)
    keyframes = []
    original = builder.obstacles.observe

    def record(frame, camera, pose, moving=None):
        above = original(frame, camera, pose, moving)
        keyframes.append((frame.copy(), camera, pose.copy(), above.copy(), builder.obstacles.last_depth.copy()))
        return above

    builder.obstacles.observe = record
    for stamp, frame, rear in replay_source(folder, use_rear):
        builder.process(frame, rear, stamp)
    if builder.floor_map is None or len(keyframes) < 3 * surfels.HOLD:
        print(f"{recording}: няма карта, пропускам")
        return 0

    plain = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, np.uint8)
    sim.VIEW_W, sim.VIEW_H = SIZE
    sim.draw_grid = lambda img, cam: img.__setitem__(slice(None), plain)
    counts, fold_of = [0, 0], []
    for _, camera, *_ in keyframes:
        fold_of.append((counts[camera] // surfels.HOLD) % FOLDS)
        counts[camera] += 1

    out = DATA / recording
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for fold in range(FOLDS):
        floor = FloorMap(builder.view_proj, center=tuple(keyframes[0][2][:2]))
        floor.origin = builder.floor_map.origin.copy()
        points = surfels.Surfels(builder.odom_projs)
        for (frame, camera, pose, above, depth), f in zip(keyframes, fold_of):
            if f == fold:
                continue
            points.observe(frame, depth, camera, pose)
            points.cloud(every=8)
            floor.add(frame, pose, None if camera == 0 else builder.rear_proj,
                      0.0 if camera == 0 else REAR_PENALTY_CM, above)
        xyz, color = points.cloud()
        for i, ((frame, camera, pose, _, _), f) in enumerate(zip(keyframes, fold_of)):
            if f != fold:
                continue
            proj = builder.odom_projs[camera]
            view = Viewpoint(*surfels.camera_matrices(proj, pose, SIZE[0] / c.SIZE[0]))
            sim.FOCAL = float(view.K[1, 1])
            image = sim.render_floor(floor, view)
            surfels.draw_points(image, view.matrices(), sim.FOCAL, xyz, color)
            if (np.abs(image.astype(int) - BACKGROUND).sum(-1) > 6).mean() < 0.15:
                continue  # почти нищо за поправяне
            truth = cv2.resize(cv2.undistort(frame, proj.K, proj.dist), SIZE, interpolation=cv2.INTER_AREA)
            for suffix, picture in (("in", image), ("gt", truth)):
                cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
                    str(out / f"{i:05d}_{suffix}.jpg"))
            saved += 1
    print(f"{recording}: {saved} двойки от {len(keyframes)} ключови кадъра")
    return saved


def load_pairs(recordings):
    pairs = []
    for recording in recordings:
        for path in sorted((DATA / recording).glob("*_in.jpg")):
            both = [cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
                    for p in (path, path.with_name(path.name.replace("_in", "_gt")))]
            pairs.append(np.concatenate([x[:, :, ::-1] for x in both], 2))  # H, W, (вход RGB, истина RGB)
    return pairs


def temperature():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def train():
    import torch
    import torch.nn.functional as F

    from nanofix import _Net

    pairs = load_pairs(TRAIN)
    if not pairs:
        raise SystemExit("Няма двойки: първо nf_train.py data")
    data = torch.from_numpy(np.stack(pairs))  # остава в обикновената памет
    print(f"{len(data)} двойки за обучение")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = _Net()
    torch.nn.init.zeros_(net.out.weight)  # започваме от "не променяй нищо"
    torch.nn.init.zeros_(net.out.bias)
    net = net.to("cuda").train()
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, STEPS, eta_min=3e-5)
    scaler = torch.cuda.amp.GradScaler()
    background = torch.tensor(BACKGROUND[::-1], device="cuda").view(1, 3, 1, 1) / 255
    cw, ch = CROP
    started, rested, losses = time.time(), 0.0, []
    for step in range(1, STEPS + 1):
        picks = rng.integers(0, len(data), BATCH)
        crops = []
        for k in picks:
            y, x = rng.integers(0, SIZE[1] - ch + 1), rng.integers(0, SIZE[0] - cw + 1)
            crops.append(data[k, y:y + ch, x:x + cw])
        batch = torch.stack(crops).to("cuda").permute(0, 3, 1, 2).float() / 255
        raw, truth = batch[:, :3], batch[:, 3:]
        content = (raw - background).abs().sum(1, keepdim=True) > 12 / 255
        target = torch.where(content, truth, raw)  # където нямаме нищо, не искаме измисляне
        with torch.cuda.amp.autocast():
            nothing = torch.zeros_like(raw)
            out = net(raw, nothing, nothing[:, :1])
            loss = F.l1_loss(out, target)
            # и ръбовете да съвпадат, иначе средното прави мъгла
            loss = loss + 0.5 * (F.l1_loss(out.diff(dim=-1), target.diff(dim=-1)) +
                                 F.l1_loss(out.diff(dim=-2), target.diff(dim=-2)))
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        schedule.step()
        losses.append(loss.item())
        if step % 50 == 0:
            heat = temperature()
            print(f"стъпка {step}/{STEPS}: загуба {np.mean(losses[-200:]):.4f}, {heat} °C, "
                  f"{(time.time() - started) / 60:.1f} мин", flush=True)
            if heat >= HOT_C:
                time.sleep(REST_S)
                rested += REST_S
        if step % 250 == 0:  # да не се губи всичко при спиране
            WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"net": net.state_dict(), "cfg": {}, "step": step}, WEIGHTS)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "cfg": {}, "step": STEPS}, WEIGHTS)
    print(f"Записано: {WEIGHTS}; {(time.time() - started) / 60:.1f} мин, от тях почивка {rested / 60:.1f}")


def evaluate():
    from difix_eval import CACHE
    from nanofix import NanoFix

    models = {"ours": None, "kitchen": NanoFix(WEIGHTS), "theirs": NanoFix()}
    pairs = load_pairs([TEST])
    if not pairs:
        raise SystemExit("Няма тестови двойки: първо nf_train.py data")
    rows, outputs = {}, {name: [] for name in models}
    for name, model in models.items():
        errors = []
        for pair in pairs:
            raw, truth = pair[:, :, 2::-1].copy(), pair[:, :, :2:-1].copy()  # обратно в BGR
            out = raw if model is None else model(raw)
            content = np.abs(raw.astype(int) - BACKGROUND).sum(-1) > 12
            errors.append(float(np.abs(out[content].astype(int) - truth[content].astype(int)).mean()))
            outputs[name].append(out)
        rows[name] = dict(version=name, test_pairs=len(pairs), mean_abs_error=round(float(np.mean(errors)), 1))

    # дрон изглед: маркерите (няма истински кадър от тази гледна точка)
    data = np.load(CACHE)
    marks = json.loads((c.CALIB_DIR / "markers.json").read_text(encoding="utf-8-sig"))["markers"]
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    def read(image, m):
        R, C, K = m[0][:, :3], m[0][:, 3], m[1][:, :3]
        to_floor = np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], -R @ C]))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        found = {}
        for quad, i in zip(corners, [] if ids is None else ids.ravel()):
            p = to_floor @ np.r_[quad.reshape(4, 2).mean(0), 1.0]
            found[int(i)] = p[:2] / p[2]
        return found

    drone = {name: [] for name in models}
    total = 0
    for image, m in zip(data["drone"], data["matrices"]):
        baseline = read(image, m)
        total += len(baseline)
        for name, model in models.items():
            out = image if model is None else model(image)
            drone[name].append(out)
            found = read(out, m)
            rows[name].setdefault("read", 0)
            rows[name]["read"] += sum(i in baseline for i in found)
            rows[name].setdefault("errors", []).extend(
                float(np.linalg.norm(p - (marks[str(i)]["x"], marks[str(i)]["y"])))
                for i, p in found.items() if i in baseline)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = c.DATA_DIR / "metrics" / f"nf_kitchen_{stamp}.csv"
    table = [(r["version"], r["test_pairs"], r["mean_abs_error"], total, r["read"],
              round(float(np.median(r["errors"])), 1) if r["errors"] else "") for r in rows.values()]
    header = ("version", "test_pairs", "mean_abs_error", "drone_markers", "markers_read", "position_error_median_cm")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    for row in table:
        print(dict(zip(header, row)))

    blank = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    sheet = []
    for k in (len(pairs) // 5, len(pairs) // 2, 4 * len(pairs) // 5):
        truth = pairs[k][:, :, :2:-1]
        sheet.append(np.hstack([truth, outputs["ours"][k], outputs["kitchen"][k], outputs["theirs"][k]]))
    for k in (4, 12):
        views = [cv2.resize(drone[name][k], SIZE, interpolation=cv2.INTER_AREA) for name in models]
        sheet.append(np.hstack([blank, *views]))
    for i, text in enumerate(("real hidden frame", "our render", "trained on our data", "their weights")):
        cv2.putText(sheet[0], text, (i * SIZE[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    sheet = cv2.resize(np.vstack(sheet), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imencode(".png", sheet)[1].tofile(str(path.with_suffix(".png")))
    print(f"Записано: {path} и .png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("data", "train", "eval"))
    what = p.parse_args().what
    if what == "data":
        for recording in (*TRAIN, TEST):
            make_pairs(recording)
    elif what == "train":
        train()
    else:
        evaluate()


if __name__ == "__main__":
    main()
