# --------------------------- dataset/range_transform.py ---------------------------
"""
Image-only version of range transforms.
Safely handles:
    - torch tensors of shape (C,H,W) or (B,C,H,W)
    - numpy arrays in (H,W,C) or (C,H,W) or (B,H,W,C) or (B,C,H,W)
    - LAB-to-RGB or L-only transforms

No video, no time dimension, no multi-object masks.
"""

import torch
import numpy as np

# ---------------------------
# Utility: Ensure tensor
# ---------------------------

def _to_tensor(x):
    """
    Convert numpy / torch input to torch.Tensor (float32).
    Leaves torch.Tensor unchanged except to ensure float dtype.
    """
    if isinstance(x, torch.Tensor):
        return x.float()
    x = np.asarray(x)
    t = torch.from_numpy(x).float()
    return t

# ---------------------------
# LAB inverse transform (L+ab → normalized)
# ---------------------------

def inv_lll2rgb_trans(x):
    """
    Accepts:
      - torch.Tensor with shape (C,H,W) or (B,C,H,W)
      - numpy array with shape (C,H,W) or (B,C,H,W)

    Returns:
      - torch.Tensor or numpy-ready array in LAB format expected by skimage.color.lab2rgb
        (if input had batch dim >1, batch preserved; otherwise returns (C,H,W))
    Notes:
      - Input channels expected to be in range [-1,1] (L and AB).
      - This converts:
          L: [-1,1] -> [0,100]
          AB: [-1,1] -> approx [-128,128]
    """
    t = _to_tensor(x)  # float tensor

    # Normalize shape to (B,C,H,W)
    added_batch = False
    if t.dim() == 3:
        t = t.unsqueeze(0)  # (1,C,H,W)
        added_batch = True
    elif t.dim() == 4:
        pass
    else:
        raise RuntimeError(f"Unexpected input shape for inv_lll2rgb_trans: {t.shape}")

    # Prepare output container
    out = torch.zeros_like(t)

    # L channel in [-1,1] -> [0,100]
    out[:, 0:1, :, :] = (t[:, 0:1, :, :] + 1.0) * 50.0

    # If AB present, convert
    if t.size(1) >= 3:
        out[:, 1:2, :, :] = t[:, 1:2, :, :] * 128.0
        out[:, 2:3, :, :] = t[:, 2:3, :, :] * 128.0
    elif t.size(1) == 2:
        # If model returns 2 channels (A,B), expecting shape (B,2,H,W):
        out = torch.zeros((t.size(0), 3, t.size(2), t.size(3)), device=t.device, dtype=t.dtype)
        out[:, 0:1, :, :] = (t[:, 0:1, :, :] + 1.0) * 50.0
        out[:, 1:3, :, :] = t[:, 1:3, :, :] * 128.0

    if added_batch:
        return out.squeeze(0)  # (C,H,W)
    return out  # (B,C,H,W)

# ---------------------------
# Identity inverse image transform
# (Used as a placeholder when decoding RGB directly)
# ---------------------------

def inv_im_trans(x):
    """
    Safe reverse transform:
    - Accepts torch.Tensor in (C,H,W) or (B,C,H,W) -> returns numpy array (H,W,C) or (B,H,W,C)
    - Accepts numpy array in (H,W,C) or (C,H,W) or batched variants -> returns (H,W,C) or (B,H,W,C)
    - Values are clipped to [0,1]
    """

    # Convert to numpy-like array
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    # Handle batched and unbatched forms
    if arr.ndim == 4:
        # (B,C,H,W) -> (B,H,W,C)
        arr = np.transpose(arr, (0, 2, 3, 1))
        arr = np.clip(arr, 0.0, 1.0)
        return arr
    elif arr.ndim == 3:
        # Could be (C,H,W) or (H,W,C). Detect which by channel dimension size.
        c0, c1, c2 = arr.shape
        # Heuristic: if first dim <=4 and last dim likely >4 (width), assume (C,H,W)
        if c0 <= 4 and c2 > 4:
            # (C,H,W) -> (H,W,C)
            arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr, 0.0, 1.0)
            return arr
        else:
            # likely (H,W,C)
            arr = np.clip(arr, 0.0, 1.0)
            return arr
    else:
        raise RuntimeError(f"Unexpected shape for inv_im_trans: {arr.shape}")
