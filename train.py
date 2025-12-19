# train.py — IMAGE-ONLY training script for ColorMNet
# -----------------------------------------------------------------------------
# Usage example:
#   python train.py --data_root ./dataset/images --refs_root ./dataset/refs \
#                   --model_out ./saves/checkpoint.pth --epochs 10 --batch_size 4
#
# Notes:
# - Expects data_root to contain image files (jpg/png). Each image is treated as a
#   training sample. If refs_root is provided, a random reference is sampled per
#   batch; otherwise the reference defaults to the same image (self-reference).
# - By default, input_mode='l' : input is grayscale L-channel (replicated to 3
#   channels) — typical for colorization. If input_mode='rgb', full RGB is used.
# - Loss: L1 on AB channels normalized to [-1,1] (matching the model's tanh output).
# - This is a simple training loop meant for fine-tuning / experiments. The real
#   ColorMNet training used advanced schedules and memory management which we
#   omit here for simplicity.
# -----------------------------------------------------------------------------

import os
from os import path
import random
from argparse import ArgumentParser
from glob import glob
import time

import numpy as np
from PIL import Image
from skimage import color

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from model.network import ColorMNet  # uses modified image-only network

# -------------------------
# Dataset
# -------------------------
class SimpleColorizationDataset(Dataset):
    """
    dataset_root: folder with images (rgb)
    refs_root: optional folder with reference images; if None, uses same image
    input_mode: 'l' or 'rgb'
    """
    def __init__(self, dataset_root, refs_root=None, input_mode="l"):
        self.dataset_root = dataset_root
        self.refs_root = refs_root
        self.input_mode = input_mode
        self.files = sorted([
            p for p in glob(path.join(dataset_root, "*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {dataset_root}")
        if refs_root is not None:
            self.refs = sorted([
                p for p in glob(path.join(refs_root, "*"))
                if p.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if len(self.refs) == 0:
                raise RuntimeError(f"No reference images found in {refs_root}")
        else:
            self.refs = None

    def __len__(self):
        return len(self.files)

    def _load_rgb(self, fp):
        img = Image.open(fp).convert("RGB")
        arr = np.array(img).astype("float32") / 255.0  # H,W,C
        return arr

    def __getitem__(self, idx):
        fp = self.files[idx]
        rgb = self._load_rgb(fp)  # HWC

        # convert to LAB to obtain target AB
        lab = color.rgb2lab(rgb)  # float in L [0,100], a,b approx [-128,127]
        L = lab[:, :, 0]  # H,W
        A = lab[:, :, 1]
        B = lab[:, :, 2]

        # prepare input_rgb depending on input_mode
        if self.input_mode == "l":
            # normalize L to [0,1] by dividing by 100 and replicate to 3 channels
            l_norm = (L / 100.0).astype("float32")
            input_rgb = np.stack([l_norm, l_norm, l_norm], axis=0)  # C,H,W
        else:
            # use original rgb in [0,1]
            input_rgb = np.transpose(rgb, (2, 0, 1)).astype("float32")  # C,H,W

        # target AB normalized to [-1,1]
        A_norm = (A / 128.0).astype("float32")  # H,W
        B_norm = (B / 128.0).astype("float32")
        target_ab = np.stack([A_norm, B_norm], axis=0)  # 2,H,W

        # exemplar (reference) selection
        if self.refs is None:
            ref_rgb = rgb  # use same image as reference
        else:
            ref_fp = random.choice(self.refs)
            ref_rgb = self._load_rgb(ref_fp)

        ref_tensor = torch.from_numpy(np.transpose(ref_rgb, (2, 0, 1))).float()  # C,H,W
        input_tensor = torch.from_numpy(input_rgb).float()
        target_tensor = torch.from_numpy(target_ab).float()

        return {
            "input": input_tensor,
            "ref": ref_tensor,
            "target_ab": target_tensor
        }

# -------------------------
# Training utilities
# -------------------------
def save_checkpoint(net, optimizer, epoch, out_path):
    os.makedirs(path.dirname(out_path), exist_ok=True)
    state = {
        "epoch": epoch,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, out_path)

# -------------------------
# Training loop
# -------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataset / dataloader
    ds = SimpleColorizationDataset(args.data_root, refs_root=args.refs_root, input_mode=args.input_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # model
    config = {}
    net = ColorMNet(config, model_path=args.pretrained if args.pretrained else None, map_location=device if args.pretrained else None)
    net.to(device)
    net.train()

    # optimizer + loss
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    criterion = nn.L1Loss()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for batch in loader:
            input_t = batch["input"].to(device)      # B,C,H,W (input rgb or grayscale replicated)
            ref_t = batch["ref"].to(device)          # B,C,H,W
            target_ab = batch["target_ab"].to(device)  # B,2,H,W (normalized to [-1,1])

            optimizer.zero_grad()
            # forward: Our ColorMNet.forward expects B,C,H,W inputs. It returns prob in [-1,1]
            pred = net(input_t, exemplar=ref_t)  # expect [B,2,H,W] or compatible

            # If model returns multiple outputs, try to select the prob-like one
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            # Ensure shapes align
            if pred.shape != target_ab.shape:
                # try to resize/interpolate pred to match target
                pred = nn.functional.interpolate(pred, size=target_ab.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(pred, target_ab)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                print(f"Epoch {epoch} Step {global_step} Loss {loss.item():.6f}")

        t1 = time.time()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} finished — avg_loss: {avg_loss:.6f} time: {(t1-t0):.1f}s")

        # save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            out_path = path.join(args.model_out, f"checkpoint_epoch{epoch}.pth")
            save_checkpoint(net, optimizer, epoch, out_path)
            print(f"Saved checkpoint: {out_path}")

    print("Training finished.")

# -------------------------
# CLI
# -------------------------
def build_parser():
    p = ArgumentParser()
    p.add_argument("--data_root", required=True, help="Folder of training images (RGB)")
    p.add_argument("--refs_root", default=None, help="Optional folder of reference images (RGB). If not provided, self-reference is used.")
    p.add_argument("--pretrained", default=None, help="Optional pretrained model path to load (state_dict).")
    p.add_argument("--model_out", default="./saves", help="Folder to save checkpoints.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--input_mode", choices=["l", "rgb"], default="l", help="l: use grayscale L replicated to 3-ch; rgb: use original RGB as input")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    # Ensure model_out exists (folder path is used as base)
    os.makedirs(args.model_out, exist_ok=True)
    train(args)
