import random
import torch
from typing import Tuple
from loguru import logger


def get_random_patch(input_tensor: torch.Tensor, target_tensor: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    pd, ph, pw = patch_size
    _, D, H, W = input_tensor.shape

    if D < pd or H < ph or W < pw:
        logger.error(f"Volume trop petit pour un patch de taille {patch_size} (volume: {(D, H, W)})")
        raise ValueError("Patch size is too large for the given volume.")

    margin_d, margin_h, margin_w = pd // 2, ph // 2, pw // 2
    center_d = random.randint(margin_d, D - margin_d - 1)
    center_h = random.randint(margin_h, H - margin_h - 1)
    center_w = random.randint(margin_w, W - margin_w - 1)

    d0 = center_d - margin_d
    h0 = center_h - margin_h
    w0 = center_w - margin_w

    
    input_patch = input_tensor[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    target_patch = target_tensor[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    print(input_tensor.shape, target_tensor.shape)
    print(target_patch.shape, input_patch.shape)

    return input_patch, target_patch
