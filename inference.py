# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <dataset_path> <model_path>

import math
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torch import Tensor, from_numpy
from torchvision.transforms.functional import pad, pil_to_tensor

from network import Generator


def get_image_paths(dataset_path: Path, n: int | None = None):
    if n is not None:
        return list(dataset_path.rglob("*.png"))[:n]
    return list(dataset_path.rglob("*.png"))


def rgb_to_lab(img_rgb: Tensor):
    img_lab = rgb2lab(img_rgb.permute(1, 2, 0).numpy()).astype("float32")
    img_lab = from_numpy(img_lab).permute(2, 0, 1)
    L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1
    return L, ab


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(512, 384),
            A.ToTensorV2(),
        ],
        additional_targets={"rgb_image": "image"},
        strict=True,
        seed=42,
    )


def inference_single(
    model: Generator, input_img: Image.Image, device: torch.device
) -> Image.Image:
    input_img = pil_to_tensor(input_img) / 255
    input_img = get_val_transforms()(image=input_img.permute((1, 2, 0)).numpy())[
        "image"
    ]
    if input_img.shape[0] == 1:
        input_img = input_img.repeat(3, 1, 1)

    generated_image = (
        model(input_img.unsqueeze(0).to(device))[0].permute((1, 2, 0)).detach().numpy()
    )
    generated_image = (generated_image * 255).astype(np.uint8)
    return Image.fromarray(generated_image)


def get_pyramid_blend_mask(tile_h, tile_w):
    """Generates a 2D pyramid blend mask."""
    ramp_y = 1 - torch.abs(torch.linspace(-1, 1, tile_h))
    ramp_x = 1 - torch.abs(torch.linspace(-1, 1, tile_w))

    mask_2d = torch.outer(ramp_y, ramp_x)

    return mask_2d.unsqueeze(0)


def lab_to_rgb_np(L: Tensor, ab: Tensor) -> np.array:
    L = (
        L + 1
    ) * 50.0  # reverse the transformation to range -1 and 1 done in dataset __getitem__
    ab = ab * 110.0
    lab = torch.cat((L, ab), dim=0).permute(1, 2, 0).cpu().detach().numpy()
    rgb_np = lab2rgb(lab)
    return rgb_np


def inference_tiled(
    model: Generator,
    input_img: Image.Image,
    device: torch.device,
    tile_h: int = 512,
    tile_w: int = 384,
    overlap: int = 128,
) -> Image.Image:
    input_img_tensor = pil_to_tensor(input_img) / 255

    img_h, img_w = input_img_tensor.shape[1], input_img_tensor.shape[2]
    padded_w = (tile_w - overlap) * math.ceil((img_w / (tile_w - overlap)))
    padded_h = (tile_h - overlap) * math.ceil((img_h / (tile_h - overlap)))
    padded = pad(
        input_img,
        ((padded_w - img_w) // 2, (padded_h - img_h) // 2),
        fill=0,
        padding_mode="constant",
    )

    # 3 x H x W
    output_canvas = torch.zeros(
        (3, padded_h, padded_w), dtype=torch.float64, device=input_img_tensor.device
    )

    # 1 x H x W
    output_canvas_weight = torch.zeros(
        (1, padded_h, padded_w), dtype=torch.float64, device=input_img_tensor.device
    )

    blend_mask = get_pyramid_blend_mask(tile_h, tile_w).to(device)

    for i in range(0, padded_h, tile_h - overlap):
        for j in range(0, padded_w, tile_w - overlap):
            crop_box = (j, i, min(j + tile_w, padded_w), min(i + tile_h, padded_h))
            tile = padded.crop(crop_box)
            original_tile_w, original_tile_h = tile.size

            padding = (0, 0, tile_w - original_tile_w, tile_h - original_tile_h)
            padded_tile = pad(tile, padding, fill=0, padding_mode="constant")
            # C x H x W
            tile_tensor = pil_to_tensor(padded_tile).to(device) / 255
            if tile_tensor.shape[0] == 1:
                tile_tensor = tile_tensor.repeat(3, 1, 1)

            generated_tile = (
                model(tile_tensor.unsqueeze(0).to(device))[0].detach().cpu().numpy()
            )

            cropped_mask = blend_mask[:, :original_tile_h, :original_tile_w].numpy()
            generated_tile = generated_tile[:, :original_tile_h, :original_tile_w]
            _, generated_h, generated_w = generated_tile.shape
            output_canvas[:, i : i + generated_h, j : j + generated_w] += (
                generated_tile * cropped_mask
            )
            output_canvas_weight[:, i : i + generated_h, j : j + generated_w] += (
                cropped_mask
            )

    # Normalize the output canvas by the weights
    output_canvas /= torch.clamp(output_canvas_weight, min=1e-6)

    out = (output_canvas * 255).numpy().astype(np.uint8)

    out_img = Image.fromarray(np.transpose(out, (1, 2, 0)), mode="RGB").crop(
        (
            (padded_w - img_w) // 2,
            (padded_h - img_h) // 2,
            (padded_w + img_w) // 2,
            (padded_h + img_h) // 2,
        )
    )
    return out_img
