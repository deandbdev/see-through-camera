"""Difix (nvidia/difix): "изчиства" рендер от 3D реконструкция с една стъпка. Само за изследване.

Моделът е с лиценз на NVIDIA за некомерсиална употреба. Зареждаме само тежестите; файлът
autoencoder_kl.py от хранилището му не се изпълнява (и не е нужен пакетът peft).
Настройките са от nv-tlabs/Difix3D: подсказка "remove degradation", една стъпка при t = 199,
без добавен шум и без guidance. Текстовата част е взета от SD-Turbo (приемаме, че е същата).
"""
import json

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

import common as c

MODELS = c.DATA_DIR.parent / "models"
SIZE = (1024, 576)       # размерът, с който моделът е правен
TIMESTEP = 199
LORA_SCALE = 2.0         # alpha / rank: peft по подразбиране alpha=8, rank=4


def _build_vae(dtype, lora_scale):
    from diffusers import AutoencoderKL

    folder = MODELS / "difix" / "vae"
    config = json.loads((folder / "config.json").read_text())
    keys = ("in_channels", "out_channels", "down_block_types", "up_block_types", "block_out_channels",
            "layers_per_block", "act_fn", "latent_channels", "norm_num_groups", "sample_size", "scaling_factor")
    vae = AutoencoderKL(**{k: config[k] for k in keys})
    # връзките "накуп" от енкодера към декодера, които обикновеният VAE няма
    for i, (channels_in, channels_out) in enumerate(((512, 512), (256, 512), (128, 512), (128, 256)), 1):
        setattr(vae.decoder, f"skip_conv_{i}", torch.nn.Conv2d(channels_in, channels_out, 1, bias=False))

    state = load_file(str(folder / "diffusion_pytorch_model.safetensors"))
    merged = {}
    for key, value in state.items():
        if ".lora_" in key:
            continue
        name = key.replace(".base_layer", "")
        if ".base_layer.weight" in key:  # добавяме LoRA поправката директно в тежестите
            prefix = key.split(".base_layer")[0]
            a, b = state[f"{prefix}.lora_A.vae_skip.weight"], state[f"{prefix}.lora_B.vae_skip.weight"]
            delta = torch.einsum("or,r...->o...", b.flatten(1), a)
            value = value + lora_scale * delta.reshape(value.shape)
        merged[name] = value
    vae.load_state_dict(merged, strict=True)
    return vae.to("cuda", dtype).eval()


class Difix:
    def __init__(self, dtype=torch.float16, lora_scale=LORA_SCALE):
        from diffusers import DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        folder, turbo = MODELS / "difix", MODELS / "sd-turbo"
        if not (folder / "unet" / "diffusion_pytorch_model.safetensors").exists():
            raise SystemExit(f"Няма Difix в {folder}.")
        self.dtype = dtype
        self.vae = _build_vae(dtype, lora_scale)
        self.unet = UNet2DConditionModel.from_pretrained(folder / "unet", torch_dtype=dtype,
                                                         local_files_only=True).to("cuda").eval()
        schedule = DDPMScheduler.from_pretrained(folder / "scheduler", local_files_only=True)
        self.alpha = schedule.alphas_cumprod[TIMESTEP].item()
        tokenizer = CLIPTokenizer.from_pretrained(folder / "tokenizer", local_files_only=True)
        encoder = CLIPTextModel.from_pretrained(turbo / "text_encoder", torch_dtype=torch.float16, variant="fp16",
                                                local_files_only=True)
        ids = tokenizer("remove degradation", padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            self.text = encoder.to("cuda")(ids.to("cuda"))[0].to(dtype)
        del encoder
        torch.cuda.empty_cache()

    def _encode(self, x):
        enc, skips = self.vae.encoder, []
        h = enc.conv_in(x)
        for block in enc.down_blocks:
            skips.append(h)
            h = block(h)
        h = enc.conv_out(enc.conv_act(enc.conv_norm_out(enc.mid_block(h))))
        mean = self.vae.quant_conv(h).chunk(2, dim=1)[0]  # без случайност: средното
        return mean * self.vae.config.scaling_factor, skips

    def _decode(self, z, skips):
        dec = self.vae.decoder
        h = dec.mid_block(dec.conv_in(self.vae.post_quant_conv(z / self.vae.config.scaling_factor)))
        convs = (dec.skip_conv_1, dec.skip_conv_2, dec.skip_conv_3, dec.skip_conv_4)
        for conv, skip, block in zip(convs, skips[::-1], dec.up_blocks):
            h = block(h + conv(skip))
        return dec.conv_out(dec.conv_act(dec.conv_norm_out(h)))

    def __call__(self, image, clean=True):
        """BGR изображение -> BGR със същия размер. clean=False: само кодиране и декодиране."""
        h, w = image.shape[:2]
        size = (w - w % 8, h - h % 8)
        x = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        with torch.no_grad():
            x = torch.from_numpy(x[:, :, ::-1].copy()).to("cuda", self.dtype).permute(2, 0, 1)[None] / 127.5 - 1
            z, skips = self._encode(x)
            if clean:
                eps = self.unet(z, TIMESTEP, encoder_hidden_states=self.text).sample
                z = (z - (1 - self.alpha) ** 0.5 * eps) / self.alpha ** 0.5
            out = self._decode(z, skips)
            out = ((out[0].permute(1, 2, 0).float() + 1) * 127.5).clamp(0, 255)
        if torch.isnan(out).any():
            raise FloatingPointError("NaN в изхода: тази точност не стига за модела")
        out = out.cpu().numpy()[:, :, ::-1].astype(np.uint8)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
