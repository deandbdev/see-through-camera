"""Стъпка 9, трето ниво: дорисуване с SD-Turbo (fp16, локално, models/sd-turbo).

SD-Turbo не е inpainting модел, затова: img2img от груб (telea) старт, а след всяка стъпка
истински известното се връща на мястото му (blended latent diffusion). Картата се обхожда
на плочки; вече дорисуваното служи за контекст на съседните, за да няма шевове.

Дорисуваното се пази в отделен слой и НИКОГА не се записва в картата на видяното.
"""
import threading
import time

import cv2
import numpy as np

import common as c
from inpaint import REACH_CM, fill, tint

MODEL_DIR = c.DATA_DIR.parent / "models" / "sd-turbo"
SIDE = 512                 # на толкова пиксела работи моделът
WINDOW_CM = 64.0           # колко под побира една плочка (8 px/cm: ивиците остават видими)
STRIDE_CM = 40.0
MIN_CONTEXT = 0.12         # под такъв дял известно в плочката моделът няма от какво да тръгне
PROMPT = "top-down photo of a floor, sharp, realistic, even lighting"
# Компромис: 0.7/3 греши по-малко по яркост, но прави гладко сиво в големите празнини;
# 0.97/4 рисува правдоподобен под, който обаче е по-далеч от истината пиксел по пиксел.
STRENGTH, STEPS = 0.97, 4


class Diffuser:
    def __init__(self, prompt=PROMPT, strength=STRENGTH, steps=STEPS, seed=1):
        import torch
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        if not MODEL_DIR.exists():
            raise SystemExit(f"Няма модел в {MODEL_DIR}.")
        if not torch.cuda.is_available():
            raise SystemExit("PyTorch не вижда CUDA, а без видеокарта дифузията е твърде бавна.")
        self.torch, self.strength, self.steps = torch, strength, steps
        load = dict(torch_dtype=torch.float16, variant="fp16", local_files_only=True)
        self.vae = AutoencoderKL.from_pretrained(MODEL_DIR / "vae", **load).to("cuda").eval()
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_DIR / "unet", **load).to("cuda").eval()
        schedule = DDPMScheduler.from_pretrained(MODEL_DIR / "scheduler", local_files_only=True)
        if schedule.config.prediction_type != "epsilon":
            raise SystemExit(f"Неочакван тип предсказване: {schedule.config.prediction_type}")
        self.alphas = schedule.alphas_cumprod
        # текстът е един и същ, затова енкодерът не стои във видеопаметта
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR / "tokenizer", local_files_only=True)
        encoder = CLIPTextModel.from_pretrained(MODEL_DIR / "text_encoder", **load)
        ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            self.text = encoder.to("cuda")(ids.to("cuda"))[0]
        del encoder
        torch.cuda.empty_cache()
        # един и същ шум за всяка плочка: еднакъв вход дава еднакъв изход (без трептене)
        gen = torch.Generator("cpu").manual_seed(seed)
        self.noise = torch.randn((1, 4, SIDE // 8, SIDE // 8), generator=gen).to("cuda", torch.float16)

    def generate(self, image, known):
        """image (BGR, квадрат), known (bool) -> същият размер, с дорисувано извън known."""
        torch = self.torch
        side = image.shape[0]
        big = cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)
        big_known = cv2.resize(known.astype(np.uint8) * 255, (SIDE, SIDE), interpolation=cv2.INTER_AREA) == 255
        # груб старт
        small = cv2.resize(big, (128, 128), interpolation=cv2.INTER_AREA)
        small_known = cv2.resize(big_known.astype(np.uint8) * 255, (128, 128),
                                 interpolation=cv2.INTER_AREA) == 255
        rough = cv2.inpaint(small, (~small_known).astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
        start = np.where(big_known[..., None], big, cv2.resize(rough, (SIDE, SIDE)))

        with torch.no_grad():
            x = torch.from_numpy(start[:, :, ::-1].copy()).to("cuda", torch.float16)
            x = x.permute(2, 0, 1)[None] / 127.5 - 1
            base = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor
            unknown = torch.from_numpy((~big_known).astype(np.float32)).to("cuda")[None, None]
            mask = torch.nn.functional.max_pool2d(unknown, 8).to(torch.float16)

            latent = base
            top = int(999 * self.strength)
            for t in np.linspace(top, 0, self.steps + 1)[:-1].round().astype(int):
                a = self.alphas[int(t)].item()
                noisy = a ** 0.5 * latent + (1 - a) ** 0.5 * self.noise
                eps = self.unet(noisy, int(t), encoder_hidden_states=self.text).sample
                guess = (noisy - (1 - a) ** 0.5 * eps) / a ** 0.5
                latent = mask * guess + (1 - mask) * base
            out = self.vae.decode(latent / self.vae.config.scaling_factor).sample
            out = ((out[0].permute(1, 2, 0).float() + 1) * 127.5).clamp(0, 255).cpu().numpy()[:, :, ::-1]

        if big_known.any():  # изравняваме цвета по това, което знаем
            out = out + (big[big_known].astype(np.float32) - out[big_known]).mean(0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.resize(out, (side, side), interpolation=cv2.INTER_AREA)


class DiffusionFiller:
    """Като BackgroundFiller: смята в отделна нишка, плочка по плочка, и пази готовото.

    Докато плочка не е готова, на нейно място се вижда telea.
    """

    method = "diffusion"

    def __init__(self, diffuser, reach_cm=REACH_CM):
        self.diffuser, self.reach = diffuser, reach_cm
        self.ms = []
        self.focus = None          # (x, y) на колата: плочките около нея са първи
        self._thread = None
        self._reset(None)

    def _reset(self, floor_map):
        self.map = floor_map
        self.result = None  # (запълнено, с оттенък, маска uint8)
        if floor_map is not None:
            self.image = np.zeros_like(floor_map.image)          # слоят на дорисуваното
            self.done = np.zeros(floor_map.image.shape[:2], bool)

    def _next_tile(self, seen, wanted):
        s = self.map.s
        side, stride = int(WINDOW_CM * s), int(STRIDE_CM * s)
        todo = wanted & ~self.done
        if not todo.any():
            return None
        ys, xs = np.nonzero(todo)
        best = None
        focus = None if self.focus is None else self.map.world_to_map(self.focus)
        margin = (side - stride) // 2
        for gy, gx in {(y // stride, x // stride) for y, x in zip(ys[::37], xs[::37])}:
            y0 = int(np.clip(gy * stride - margin, 0, seen.shape[0] - side))
            x0 = int(np.clip(gx * stride - margin, 0, seen.shape[1] - side))
            win = (slice(y0, y0 + side), slice(x0, x0 + side))
            context = (seen[win] | self.done[win]).mean()
            if context < MIN_CONTEXT or todo[win].sum() < 50:
                continue
            center = np.array((x0 + side / 2, y0 + side / 2))
            score = -np.linalg.norm(center - focus) if focus is not None else context
            if best is None or score > best[0]:
                best = (score, win)
        return None if best is None else best[1]

    def _step(self, image, seen):
        """Дорисува една плочка. False, ако няма какво."""
        s = self.map.s
        near = cv2.distanceTransform((~seen).astype(np.uint8), cv2.DIST_L2, 3) <= self.reach * s
        wanted = near & ~seen
        win = self._next_tile(seen, wanted)
        if win is None:
            return False
        t0 = time.perf_counter()
        known = seen[win] | self.done[win]
        tile = np.where(seen[win][..., None], image[win], self.image[win])
        out = self.diffuser.generate(tile, known)
        new = wanted[win] & ~self.done[win]
        self.image[win][new] = out[new]
        self.done[win][new] = True
        self.ms.append((time.perf_counter() - t0) * 1000)
        return True

    def _publish(self, image, seen):
        filled, invented = fill(image, seen, self.map.s)
        use = self.done & ~seen
        filled[use] = self.image[use]
        invented |= use
        self.result = (filled, tint(filled), invented.astype(np.uint8))

    def _work(self, image, seen, tiles):
        count = 0
        self._publish(image, seen)
        while count < tiles and self._step(image, seen):
            count += 1
            self._publish(image, seen)

    def update(self, floor_map, wait=False, tiles=3):
        """wait: чакай (за видео). tiles: най-много толкова плочки с тази снимка на картата."""
        if floor_map is None or (self._thread and self._thread.is_alive()):
            return
        if floor_map is not self.map:
            self._reset(floor_map)
        args = (floor_map.image.copy(), np.isfinite(floor_map.dist), tiles)
        if wait:
            self._work(*args)
        else:
            self._thread = threading.Thread(target=self._work, args=args, daemon=True)
            self._thread.start()

    def layer(self, floor_map, mark=True):
        if self.result is None or self.map is not floor_map:
            return None
        return self.result[1 if mark else 0], self.result[2]
