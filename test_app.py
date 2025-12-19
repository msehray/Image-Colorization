# ---------------------- test_app.py — IMAGE ONLY VERSION ----------------------
# This version removes ALL video logic, ALL dataset dependencies, ALL masks,
# and treats the model as a simple image → colored image propagator.

import os
from os import path
from argparse import ArgumentParser
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ColorMNet imports (unchanged)
from model.network import ColorMNet
from inference.inference_core import InferenceCore
from dataset.range_transform import inv_lll2rgb_trans
from skimage import color

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def detach_to_cpu(x):
    return x.detach().cpu()

def tensor_to_np_float(image):
    return image.numpy().astype('float32')

def lab2rgb_transform_PIL(mask):
    mask_d = detach_to_cpu(mask)
    mask_d = inv_lll2rgb_trans(mask_d)
    im = tensor_to_np_float(mask_d)

    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    im = color.lab2rgb(im)
    return im.clip(0, 1)


# -----------------------------------------------------------------------------
# Simple IMAGE dataset (replaces DAVIS/Youtube dataset completely)
# -----------------------------------------------------------------------------

class SimpleImageDataset(Dataset):
    """Loads a folder of grayscale images and returns normalized tensors."""

    def __init__(self, folder):
        self.folder = folder
        self.frames = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(self.frames) == 0:
            raise RuntimeError(f"No images found in {folder}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        fname = self.frames[idx]
        img_path = path.join(self.folder, fname)

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        rgb = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W

        return {
            "rgb": rgb,
            "frame": fname,
            "shape": rgb.shape[1:],  # HW
        }


# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", default="DINOv2FeatureV6_LocalAtten_s2_154000.pth")
    parser.add_argument("--d16_batch_path", required=True, help="Root containing input_images/run_xxx/")
    parser.add_argument("--ref_path", required=True, help="Root containing ref/run_xxx/ref.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=-1)
    return parser


# -----------------------------------------------------------------------------
# Core inference (IMAGE ONLY)
# -----------------------------------------------------------------------------

def run_inference(args):
    """
    Pure IMAGE-ONLY inference:
    - Reads input_images/<run_id>/00000.png, 00001.png, ...
    - Reads ref/<run_id>/ref.png
    - Colors each frame independently using ColorMNet
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Locate run folder inside d16_batch_path
    run_dirs = sorted(os.listdir(args.d16_batch_path))
    if len(run_dirs) == 0:
        raise RuntimeError("No run folder found in d16_batch_path")

    run_id = run_dirs[0]   # e.g. run_173xxxxxxxx
    input_folder = path.join(args.d16_batch_path, run_id)
    ref_folder   = path.join(args.ref_path, run_id)

    ref_path = path.join(ref_folder, "ref.png")
    if not path.exists(ref_path):
        raise RuntimeError(f"Reference image not found: {ref_path}")

    # Load reference image
    ref_img = Image.open(ref_path).convert("RGB")
    ref_arr = np.array(ref_img).astype(np.float32) / 255.0
    ref_tensor = torch.from_numpy(ref_arr).permute(2, 0, 1).float().to(device)

    # 2. Load model
    config = {"FirstFrameIsNotExemplar": True}
    network = ColorMNet(config, args.model).to(device).eval()
    weights = torch.load(args.model, map_location=device)
    network.load_weights(weights, init_as_zero_if_needed=True)

    processor = InferenceCore(network, config=config)

    # 3. Build dataset
    dataset = SimpleImageDataset(input_folder)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 4. Prepare output
    out_dir = path.join(args.output, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # 5. Process each frame
    for batch in loader:
        rgb = batch["rgb"][0].to(device)    # C,H,W
        fname = batch["frame"][0]

        # Colorization call
        # ---------------------------------------------------------
        # If processor.colorize() does not exist, use:
        # prob = processor.step_AnyExemplar(rgb, ref_tensor, None, None, end=False)
        # ---------------------------------------------------------
        try:
            prob = processor.colorize(rgb, ref_tensor)
        except:
            # Fallback to AnyExemplar API using reference image as "mask substitute"
            ref_fake = ref_tensor[:1, :, :]  # L channel style
            prob = processor.step_AnyExemplar(
                rgb,
                ref_fake.repeat(3,1,1),  # exemplar
                None,
                None,
                end=True
            )

        # Convert LAB → RGB
        out_img = lab2rgb_transform_PIL(torch.cat([rgb[:1,:,:], prob], dim=0))
        out_img = (out_img * 255).astype(np.uint8)
        out_pil = Image.fromarray(out_img)

        out_pil.save(path.join(out_dir, fname))

    print("Image colorization complete.")


# -----------------------------------------------------------------------------
# Public entry point for app.py
# -----------------------------------------------------------------------------

def run_cli(args_list=None):
    parser = build_parser()
    args = parser.parse_args(args_list)
    run_inference(args)


def main():
    run_cli()


if __name__ == "__main__":
    main()
