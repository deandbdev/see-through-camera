"""Малка мрежа, която дорисува невидяния под (изглед отгоре). Дорисуваното е ПРЕДПОЛОЖЕНИЕ.

  .venv/Scripts/python.exe src/fill_train.py data    # карти на пода от всички записи
  .venv/Scripts/python.exe src/fill_train.py train   # обучава (почива, ако картата прегрее)
  .venv/Scripts/python.exe src/fill_train.py eval    # мери върху ивица, която обучението не е виждало

Вход: подът с дупките, къде е известен, и предишният резултат на мрежата (за да не трепти).
Учи се само от истински под: местата на маркерите, предметите над пода и ПРОВЕРОЧНАТА ИВИЦА
(x > TEST_X см) не участват. Мрежата научава този под; на друг ще трябва ново обучение.
"""
import argparse
import csv
import subprocess
import time

import cv2
import numpy as np

import common as c
from motion import rot

DATA = c.DATA_DIR / "floor_fill"
WEIGHTS = c.DATA_DIR.parent / "models" / "floor-fill" / "floor_kitchen.pt"
PPC = 2.0                 # пиксели на см, на които работи мрежата
HIDE_CM = 22.0            # около маркер
TEST_X = 62.0             # вдясно от това (свят, см) е проверката
CHANNELS = (32, 64, 128, 192, 256)
STEPS, BATCH, CROP = 8000, 12, 192   # 192 px = 96 см
HOT_C, REST_S = 87, 30
TEST_RUN = "20261125_135744"


def clean_map(floor_map, markers):
    """-> (изображение, годен под) на PPC: без местата на маркерите."""
    s = floor_map.s
    seen = np.isfinite(floor_map.dist)
    hidden = np.zeros(seen.shape, np.uint8)
    for m in markers.values():
        px = floor_map.world_to_map((m["x"], m["y"]))
        cv2.circle(hidden, px.astype(int), int(HIDE_CM * s), 1, -1)
    valid = seen & (hidden == 0)
    # шахматната дъска за калибрация, ако още лежи на пода
    found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(floor_map.image, cv2.COLOR_BGR2GRAY), (9, 6))
    if found:
        box = cv2.minAreaRect(corners.astype(np.float32))
        board = np.zeros_like(hidden)
        grown = (box[0], (box[1][0] + 16 * s, box[1][1] + 16 * s), box[2])
        cv2.fillConvexPoly(board, cv2.boxPoints(grown).astype(np.int32), 1)
        valid &= board == 0
        print("   скрита шахматна дъска")
    k = PPC / s
    image = cv2.resize(floor_map.image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    valid = cv2.resize(valid.astype(np.uint8) * 255, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) == 255
    return image, valid


def make_data():
    from static_map import MapBuilder, replay_source

    DATA.mkdir(parents=True, exist_ok=True)
    for folder in sorted((c.DATA_DIR / "odom").iterdir()):
        use_rear = (folder / "rear").exists()
        builder = MapBuilder(use_rear=use_rear, obstacles=True)
        for stamp, frame, rear in replay_source(folder, use_rear):
            builder.process(frame, rear, stamp)
        if builder.floor_map is None:
            print(f"{folder.name}: няма карта")
            continue
        image, valid = clean_map(builder.floor_map, builder.localizer.markers)
        if not valid.any():
            continue
        ys, xs = np.nonzero(valid)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        corner = builder.floor_map.origin + (x0 / PPC, -y0 / PPC)  # свят: горе-ляво на изрязаното
        np.savez_compressed(DATA / f"{folder.name}.npz", image=image[y0:y1, x0:x1], valid=valid[y0:y1, x0:x1],
                            corner=corner)
        picture = np.where(valid[..., None], image, 0)[y0:y1, x0:x1].astype(np.uint8)
        split = int((TEST_X - corner[0]) * PPC)
        if 0 < split < picture.shape[1]:
            cv2.line(picture, (split, 0), (split, picture.shape[0]), (0, 0, 255), 1)
        cv2.imencode(".png", picture)[1].tofile(str(DATA / f"{folder.name}.png"))
        world_x = corner[0] + (np.arange(x1 - x0) + 0.5) / PPC
        inside = valid[y0:y1, x0:x1]
        print(f"{folder.name}: {inside.sum() / PPC ** 2 / 1e4:.2f} м² под, от тях в проверочната ивица "
              f"{inside[:, world_x > TEST_X].sum() / PPC ** 2 / 1e4:.2f} м²")


def load_maps():
    """Картите за обучение: проверочната ивица е обявена за негодна и е изтрита."""
    maps = []
    pad = CROP
    for path in sorted(DATA.glob("*.npz")):
        d = np.load(path)
        image, valid, corner = d["image"].copy(), d["valid"].copy(), d["corner"]
        world_x = corner[0] + (np.arange(image.shape[1]) + 0.5) / PPC
        valid[:, world_x > TEST_X] = False
        image[~valid] = 0
        if valid.sum() < 2000:
            continue
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        valid = np.pad(valid, pad)
        maps.append((image, valid.astype(np.uint8) * 255))
    return maps


def sample(maps, weights, rng):
    """Случайно завъртян и мащабиран квадрат от пода: (изображение, годно)."""
    while True:
        image, valid = maps[rng.choice(len(maps), p=weights)]
        ys, xs = np.nonzero(valid[::4, ::4])
        i = rng.integers(len(ys))
        centre = (float(xs[i] * 4 + rng.integers(-30, 31)), float(ys[i] * 4 + rng.integers(-30, 31)))
        M = cv2.getRotationMatrix2D(centre, rng.uniform(0, 360), rng.uniform(0.8, 1.3))
        M[:, 2] += (CROP / 2 - centre[0], CROP / 2 - centre[1])
        patch = cv2.warpAffine(image, M, (CROP, CROP), flags=cv2.INTER_LINEAR)
        ok = cv2.warpAffine(valid, M, (CROP, CROP), flags=cv2.INTER_LINEAR) == 255
        if ok.mean() < 0.3:
            continue
        if rng.random() < 0.5:
            patch, ok = patch[:, ::-1], ok[:, ::-1]
        gain = rng.uniform(0.85, 1.15) * rng.uniform(0.95, 1.05, 3)
        patch = np.clip(patch * gain, 0, 255).astype(np.uint8)
        return patch, ok.copy()


def holes(rng):
    """Две вложени дупки (голяма, по-малка): все едно колата открива още под."""
    big = np.zeros((CROP, CROP), np.uint8)
    if rng.random() < 0.7:  # всичко отвъд една права: край на видяното
        angle = rng.uniform(0, 2 * np.pi)
        normal = np.array((np.cos(angle), np.sin(angle)))
        offset = rng.uniform(-0.15, 0.3) * CROP
        yy, xx = np.indices(big.shape)
        side = (xx - CROP / 2) * normal[0] + (yy - CROP / 2) * normal[1]
        big[side > offset] = 1
        small = (side > offset + rng.uniform(4, 30)).astype(np.uint8)
    else:
        small = np.zeros_like(big)
    for _ in range(rng.integers(0 if big.any() else 1, 4)):  # и затворени дупки
        w, h = rng.uniform(8, 60, 2)
        box = cv2.boxPoints(((rng.uniform(30, CROP - 30), rng.uniform(30, CROP - 30)), (w, h), rng.uniform(0, 180)))
        cv2.fillConvexPoly(big, box.astype(np.int32), 1)
        inner = box.mean(0) + (box - box.mean(0)) * rng.uniform(0.4, 0.9)
        cv2.fillConvexPoly(small, inner.astype(np.int32), 1)
    return big > 0, small > 0


def build_net():
    import torch
    import torch.nn.functional as F

    from nanofix import _Net

    class FillNet(_Net):
        def forward(self, image, known, prev, prev_ok):
            """image в [0,1] (B,3,H,W), known и prev_ok (B,1,H,W). Връща целия под."""
            count = known.sum((2, 3), keepdim=True).clamp(min=1)
            mean = (image * known).sum((2, 3), keepdim=True) / count
            base = image * known + mean * (1 - known)  # започваме от средния цвят
            x = torch.cat([base - mean, known, (prev - mean) * prev_ok, prev_ok], 1)
            skips = []
            for i, encoder in enumerate(self.enc):
                x = encoder(x)
                if i < len(self.enc) - 1:
                    skips.append(x)
                    x = F.avg_pool2d(x, 2)
            x = self.mid(x)
            for decoder in self.dec:
                skip = skips.pop()
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = decoder(torch.cat([x, skip], 1))
            return image * known + (base + self.out(x)).clamp(0, 1) * (1 - known)

    net = FillNet(ch=CHANNELS, cin=8)
    torch.nn.init.zeros_(net.out.weight)
    torch.nn.init.zeros_(net.out.bias)
    return net


def temperature():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def train(steps):
    import torch

    maps = load_maps()
    if not maps:
        raise SystemExit("Няма карти: първо fill_train.py data")
    weights = np.array([m[1].sum() for m in maps], float)
    weights /= weights.sum()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = build_net().to("cuda").train()
    print(f"{len(maps)} карти, {sum(p.numel() for p in net.parameters()) / 1e6:.1f} млн. параметъра", flush=True)
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps, eta_min=3e-5)
    scaler = torch.cuda.amp.GradScaler()
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    def loss_of(out, truth, where):
        """Само там, където сме скрили истински под: и яркостта, и ръбовете."""
        total = ((out - truth).abs() * where).sum() / (where.sum() * 3 + 1)
        for dim in (-1, -2):
            pair = where.narrow(dim, 1, where.shape[dim] - 1) * where.narrow(dim, 0, where.shape[dim] - 1)
            total = total + 1.0 * ((out.diff(dim=dim) - truth.diff(dim=dim)).abs() * pair).sum() / (pair.sum() * 3 + 1)
        return total

    started, rested, losses = time.time(), 0.0, []
    for step in range(1, steps + 1):
        images, oks, bigs, smalls = [], [], [], []
        for _ in range(BATCH):
            image, ok = sample(maps, weights, rng)
            big, small = holes(rng)
            images.append(image[:, :, ::-1].copy())
            oks.append(ok)
            bigs.append(big)
            smalls.append(small)
        truth = torch.from_numpy(np.stack(images)).to("cuda").permute(0, 3, 1, 2).float() / 255
        ok, big, small = (torch.from_numpy(np.stack(x))[:, None].to("cuda").float() for x in (oks, bigs, smalls))
        first_known, known = ok * (1 - big), ok * (1 - small)
        two = rng.random() < 0.5
        nothing, none = torch.zeros_like(truth), torch.zeros_like(known)
        if two:  # в отделен autocast: иначе кешира теглата без градиенти за следващото преминаване
            with torch.no_grad(), torch.cuda.amp.autocast():
                prev = net(truth * first_known, first_known, nothing, none).float()
        with torch.cuda.amp.autocast():
            if two:
                prev_ok = 1 - first_known
                out = net(truth * known, known, prev, prev_ok)
                where = ok * (1 - known)
                still = (1 - known) * prev_ok
                loss = loss_of(out, truth, where) + 0.2 * ((out - prev).abs() * still).sum() / (still.sum() * 3 + 1)
            else:
                known = first_known
                out = net(truth * known, known, nothing, none)
                loss = loss_of(out, truth, ok * (1 - known))
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        schedule.step()
        losses.append(loss.item())
        if step % 100 == 0:
            heat = temperature()
            print(f"стъпка {step}/{steps}: загуба {np.mean(losses[-300:]):.4f}, {heat} °C, "
                  f"{(time.time() - started) / 60:.1f} мин", flush=True)
            if heat >= HOT_C:
                time.sleep(REST_S)
                rested += REST_S
        if step % 500 == 0 or step == steps:
            torch.save({"net": net.state_dict(), "step": step}, WEIGHTS)
    print(f"Записано: {WEIGHTS}; {(time.time() - started) / 60:.1f} мин, от тях почивка {rested / 60:.1f}")


class FloorFill:
    """Дорисуване с обучената мрежа. Изображенията са на PPC пиксела на см, BGR."""

    def __init__(self, path=WEIGHTS):
        import torch

        self.torch = torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        net = build_net()
        net.load_state_dict(checkpoint["net"])
        self.net = net.to("cuda").half().eval()

    def __call__(self, image, known, prev=None, prev_ok=None):
        torch = self.torch
        h, w = known.shape
        H, W = -(-h // 16) * 16, -(-w // 16) * 16

        def tensor(a):
            a = np.pad(a, ((0, H - h), (0, W - w)) + ((0, 0),) * (a.ndim - 2))
            t = torch.from_numpy(a).to("cuda").half()
            return t.permute(2, 0, 1)[None] if a.ndim == 3 else t[None, None]

        rgb = tensor(image[:, :, ::-1].astype(np.float32) / 255)
        k = tensor(known.astype(np.float32))
        if prev is None:
            p, p_ok = torch.zeros_like(rgb), torch.zeros_like(k)
        else:
            p, p_ok = tensor(prev[:, :, ::-1].astype(np.float32) / 255), tensor(prev_ok.astype(np.float32))
        with torch.no_grad():
            out = self.net(rgb * k, k, p, p_ok)
        return (out[0].permute(1, 2, 0)[:h, :w] * 255).round().byte().cpu().numpy()[:, :, ::-1].copy()


def evaluate(diffusion):
    from inpaint import evaluate as holes_test
    from mapper import FloorMap

    d = np.load(DATA / f"{TEST_RUN}.npz")
    image, valid, corner = d["image"], d["valid"], d["corner"]
    # същата карта, увеличена до 4 пиксела на см, каквито ползват останалите методи
    floor = FloorMap.__new__(FloorMap)
    floor.s = 4.0
    floor.image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    seen = cv2.resize(valid.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) > 0
    floor.image[~seen] = 0
    floor.dist = np.where(seen, 0.0, np.inf).astype(np.float32)
    world_x = corner[0] + (np.arange(floor.image.shape[1]) + 0.5) / floor.s
    strip = np.tile(world_x > TEST_X + 10, (floor.image.shape[0], 1))
    pad = 140  # прозорецът на проверката е 64 см и трябва да се побира
    floor.image = cv2.copyMakeBorder(floor.image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    floor.dist = np.pad(floor.dist, pad, constant_values=np.inf)
    strip = np.pad(strip, pad)

    net = FloorFill()

    def ours(shown, known):
        k = PPC / floor.s
        size = (round(shown.shape[1] * k), round(shown.shape[0] * k))
        small = cv2.resize(shown, size, interpolation=cv2.INTER_AREA)
        small_known = cv2.resize(known.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) == 255
        out = net(small, small_known)
        big = cv2.resize(out, (shown.shape[1], shown.shape[0]), interpolation=cv2.INTER_CUBIC)
        return np.where(known[..., None], shown, big).astype(np.uint8)

    extra = {"net": ours}
    if diffusion:
        from diffuse import Diffuser
        extra["diffusion"] = Diffuser().generate
    rows, examples = holes_test(floor, 20, extra=extra, centres=strip)
    names = ("black", "mean", "telea", "ns", *extra)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = c.DATA_DIR / "metrics" / f"floor_fill_{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(("kind", "hole_cm", "method", "n", "mean_abs_error", "texture"))
        w.writerows(rows)
    for column, title in ((4, "Средна грешка в яркостта (0 = идеално):"), (5, "Текстура (1 = колкото истината):")):
        print(title)
        print(f"{'вид':8s}{'см':>4s}  " + "".join(f"{m:>11s}" for m in names))
        for i in range(0, len(rows), len(names)):
            group = rows[i:i + len(names)]
            print(f"{group[0][0]:8s}{group[0][1]:4d}  " + "".join(f"{r[column]:11.2f}" for r in group))
    columns = ("truth", "hidden", "telea", "ns", *extra)
    header = np.zeros((28, 240 * len(columns), 3), np.uint8)
    for i, text in enumerate(columns):
        cv2.putText(header, text, (i * 240 + 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    sheet = []
    for name, tiles in examples:
        row = np.hstack(tiles)
        cv2.putText(row, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        sheet.append(row)
    cv2.imencode(".png", np.vstack([header, *sheet]))[1].tofile(str(path.with_suffix(".png")))
    print(f"Записано: {path} (и .png)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("data", "train", "eval"))
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--diffusion", action="store_true")
    a = p.parse_args()
    if a.what == "data":
        make_data()
    elif a.what == "train":
        train(a.steps)
    else:
        evaluate(a.diffusion)


if __name__ == "__main__":
    main()
