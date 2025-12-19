# inference/data/mask_mapper.py
"""
Simplified MaskMapper for IMAGE-ONLY pipeline.

This mapper is intentionally minimal and robust:
- If `None` is provided, it returns `None` (no-mask workflow).
- If a mask is provided (PIL / numpy / torch.Tensor / filepath), it converts it
  to a torch.FloatTensor on CPU with shape (1, 1, H, W) — i.e. batch=1, obj=1.
- If the mask already appears to have multiple object channels, it keeps them
  but ensures dtype=float32 and shape (1, C, H, W).

Goal: be a no-op for most image-only runs (masks optional), but still accept
masks if the user provides them.
"""

from typing import Optional, Union
import os
from os import path

import numpy as np
from PIL import Image
import torch


class MaskMapper:
    def __init__(self, device: Optional[torch.device] = None):
        """
        device: optional torch.device where mask tensors should be placed.
                If None, tensors remain on CPU; downstream code may .to(device).
        """
        self.device = device

    def _load_image_to_numpy(self, src: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Load PIL / path / numpy into HxW or HxWxC numpy float in [0,1]."""
        if isinstance(src, np.ndarray):
            arr = src
        elif isinstance(src, Image.Image):
            arr = np.array(src)
        elif isinstance(src, str) and path.exists(src):
            with Image.open(src) as im:
                arr = np.array(im)
        else:
            raise ValueError("Unsupported mask input type for MaskMapper: " + str(type(src)))

        # If grayscale image, result will be HxW; if color, HxWxC
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr

    def _np_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to torch.FloatTensor of shape (1, C, H, W).
        - If arr shape is (H, W) -> becomes (1, 1, H, W)
        - If arr shape is (H, W, C) -> becomes (1, C, H, W)
        """
        if arr.ndim == 2:
            arr = arr[None, :, :]               # (1, H, W)
        elif arr.ndim == 3:
            # Some masks might be HxWx1 or HxWxN
            arr = arr.transpose(2, 0, 1)       # (C, H, W)
        else:
            raise ValueError("Unexpected numpy mask shape: " + str(arr.shape))

        # Ensure float32
        t = torch.from_numpy(arr).float()
        # Ensure shape (1, C, H, W)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t

    def map(self, mask: Union[None, str, Image.Image, np.ndarray, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Convert various mask representations into a torch tensor (1, C, H, W) or return None.

        Accepted mask inputs:
          - None -> returns None
          - PIL.Image.Image -> converted
          - numpy.ndarray -> converted
          - torch.Tensor -> normalized and possibly reshaped
          - filepath (string) pointing to an image -> loaded

        Returned tensor is dtype float32 in range [0,1]. If self.device was provided,
        it will be moved to that device.
        """
        if mask is None:
            return None

        # Already a torch tensor
        if isinstance(mask, torch.Tensor):
            t = mask.float()
            # Accept shapes: (H,W), (C,H,W), (B,C,H,W)
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)   # -> (1,1,H,W)
            elif t.dim() == 3:
                # assume (C,H,W) -> (1,C,H,W)
                t = t.unsqueeze(0)
            elif t.dim() == 4:
                # already (B,C,H,W) — if B>1 we keep it as-is
                pass
            else:
                raise ValueError("Unsupported torch mask ndim: " + str(t.dim()))

            # Normalize if values seem 0/255
            if t.max() > 1.1:
                t = t / 255.0

            if self.device is not None:
                t = t.to(self.device)
            return t

        # PIL / numpy / path
        arr = self._load_image_to_numpy(mask)
        t = self._np_to_tensor(arr)

        # If mask is binary (0/1), ensure float in [0,1]
        if t.max() > 1.1:
            t = t / 255.0

        if self.device is not None:
            t = t.to(self.device)
        return t

    # convenience alias
    __call__ = map
