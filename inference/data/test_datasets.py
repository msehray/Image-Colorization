# inference/data/test_datasets.py
"""
Image-only test datasets for ColorMNet inference.

This file replaces the original video-oriented dataset classes with a
lightweight image-sequence loader suitable for:
    input root:  <d16_batch_path>/<run_id>/*.png
    ref root:    <ref_path>/<run_id>/ref.png

It provides:
- SimpleImageSequence: a minimal sequence object that supports iteration via DataLoader
- get_image_sequence_readers(d16_batch_path) -> list of sequence readers
"""

import os
from os import path
from typing import List
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Single sequence dataset: loads images from a single folder and yields items
# ---------------------------------------------------------------------------
class SimpleImageSequence(Dataset):
    """
    A dataset for a single sequence folder (image-only).

    It yields dicts compatible with the minimal expectations in test_app.py:
        {
            "rgb": torch.Tensor C,H,W (float32, [0,1]),
            "frame": frame_filename,
            "shape": (H, W),
            "need_resize": False,
            "save": True
        }
    """

    def __init__(self, folder: str):
        self.folder = folder
        # accept common extensions
        self.frames = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(self.frames) == 0:
            raise RuntimeError(f"No image frames found in folder: {folder}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        fname = self.frames[idx]
        fp = path.join(self.folder, fname)
        img = Image.open(fp).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # H,W,C in [0,1]
        # convert to C,H,W torch tensor
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        return {
            "rgb": t,
            "frame": fname,
            "shape": t.shape[1:],   # H, W
            "need_resize": False,
            "save": True
        }


# ---------------------------------------------------------------------------
# Helper to return a list of sequence readers (compatible with original meta_loader usage)
# ---------------------------------------------------------------------------
def get_image_sequence_readers(d16_batch_path: str) -> List[SimpleImageSequence]:
    """
    Scans d16_batch_path for run folders and returns a list of SimpleImageSequence objects.
    d16_batch_path is expected to contain one or more subfolders (run_xxx).
    """
    if not path.isdir(d16_batch_path):
        raise RuntimeError(f"d16_batch_path does not exist or is not a directory: {d16_batch_path}")

    subdirs = sorted([
        d for d in os.listdir(d16_batch_path)
        if path.isdir(path.join(d16_batch_path, d))
    ])
    if len(subdirs) == 0:
        raise RuntimeError(f"No subfolders found in d16_batch_path: {d16_batch_path}")

    seqs = []
    for s in subdirs:
        folder = path.join(d16_batch_path, s)
        try:
            seq = SimpleImageSequence(folder)
            # Attach vid_name attribute to mimic original API
            seq.vid_name = s
            seq.root_folder = folder
            seqs.append(seq)
        except Exception as e:
            # Skip empty or problematic folders but warn
            print(f"[WARN] Skipping folder {folder}: {e}")
            continue
    return seqs


# ---------------------------------------------------------------------------
# Convenience function for backward-compatibility name used in old code
# ---------------------------------------------------------------------------
def DAVISTestDataset_221128_TransColorization_batch(d16_batch_path, imset=None, size=-1, args=None):
    """
    Backwards-compatible factory replacement.
    Original code used this class name; we return the list of readers (meta_loader)
    and it should be consumed similarly:
        meta_dataset = DAVISTestDataset_...(...)
        meta_loader = meta_dataset.get_datasets()  # original behavior

    To keep usage easy, we return an object with get_datasets() method.
    """
    class _Wrapper:
        def __init__(self, root):
            self.root = root
            self.seqs = get_image_sequence_readers(root)

        def get_datasets(self):
            return self.seqs

        def __len__(self):
            return len(self.seqs)

    return _Wrapper(d16_batch_path)


# If someone imports specific utilities, expose get_image_sequence_readers
__all__ = [
    "SimpleImageSequence",
    "get_image_sequence_readers",
    "DAVISTestDataset_221128_TransColorization_batch",
]
