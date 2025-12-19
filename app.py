# app.py — Gradio front-end adapted for IMAGE-only colorization
import os
import sys
import shutil
import urllib.request
from os import path
import io
from contextlib import redirect_stdout, redirect_stderr
import time
import zipfile

import gradio as gr
from PIL import Image
import cv2
import torch  # used for cuda device set / sync / empty_cache

# ----------------- BASIC INFO -----------------
CHECKPOINT_URL = "https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth"
CHECKPOINT_LOCAL = "DINOv2FeatureV6_LocalAtten_s2_154000.pth"

TITLE = "ColorMNet — 图像着色 / Image Colorization (Local GPU)"
DESC = """
Upload one or more **B&W images** and a **reference image** (colored), then click “Start Coloring”.  
This app runs on a local GPU and calls `test_app.run_cli(args_list)` in-process.  
Temporary workspace layout:
- input images: `_colormnet_tmp/input_images/<run_stem>/00000.png ...`
- reference: `_colormnet_tmp/ref/<run_stem>/ref.png`
- outputs: `_colormnet_tmp/output/<run_stem>/*.png`
"""

# ----------------- TEMP WORKDIR -----------------
TEMP_ROOT = path.join(os.getcwd(), "_colormnet_tmp")
INPUT_DIR = "input_images"
REF_DIR = "ref"
OUTPUT_DIR = "output"

def reset_temp_root():
    """Clear and re-create temp workspace per run."""
    if path.isdir(TEMP_ROOT):
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
    os.makedirs(TEMP_ROOT, exist_ok=True)
    for sub in (INPUT_DIR, REF_DIR, OUTPUT_DIR):
        os.makedirs(path.join(TEMP_ROOT, sub), exist_ok=True)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# ----------------- CHECKPOINT (optional) -----------------
def ensure_checkpoint():
    """Pre-download checkpoint to avoid timeouts on first inference (optional)."""
    try:
        if not path.exists(CHECKPOINT_LOCAL):
            print(f"[INFO] Downloading checkpoint from: {CHECKPOINT_URL}")
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_LOCAL)
            print("[INFO] Checkpoint downloaded:", CHECKPOINT_LOCAL)
    except Exception as e:
        print(f"[WARN] Pre-download of checkpoint failed: {e}")

# ----------------- Helper: write uploaded files -----------------
def save_uploaded_images(files, target_dir):
    """
    Save a list of uploaded files (each entry may be a dict or path) into target_dir
    as 00000.png, 00001.png, ...
    Returns number of images saved.
    """
    ensure_dir(target_dir)
    idx = 0
    for f in files:
        # gr.Files returns dicts with 'name' or a local path string depending on context
        if isinstance(f, dict) and "name" in f:
            src = f["name"]
            # attempt to read and re-save as png to normalize format
            try:
                img = Image.open(src).convert("RGB")
                out_path = path.join(target_dir, f"{idx:05d}.png")
                img.save(out_path)
                idx += 1
                continue
            except Exception:
                # fallback: copy raw file
                shutil.copy2(src, path.join(target_dir, f"{idx:05d}" + path.splitext(src)[1]))
                idx += 1
                continue
        elif isinstance(f, str) and path.exists(f):
            try:
                img = Image.open(f).convert("RGB")
                out_path = path.join(target_dir, f"{idx:05d}.png")
                img.save(out_path)
                idx += 1
                continue
            except Exception:
                shutil.copy2(f, path.join(target_dir, f"{idx:05d}" + path.splitext(f)[1]))
                idx += 1
                continue
        else:
            # attempt to handle PIL Image
            try:
                if isinstance(f, Image.Image):
                    out_path = path.join(target_dir, f"{idx:05d}.png")
                    f.save(out_path)
                    idx += 1
                    continue
            except Exception:
                pass
        # if unknown type, skip
    return idx

def zip_output_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for file in sorted(files):
                full = path.join(root, file)
                arcname = path.relpath(full, folder_path)
                zf.write(full, arcname=arcname)

# ----------------- CLI MAPPING (unchanged) -----------------
CONFIG_TO_CLI = {
    "FirstFrameIsNotExemplar": "--FirstFrameIsNotExemplar",  # bool
    "dataset": "--dataset",
    "split": "--split",
    "save_all": "--save_all",                                # bool
    "benchmark": "--benchmark",                              # bool
    "disable_long_term": "--disable_long_term",              # bool
    "max_mid_term_frames": "--max_mid_term_frames",
    "min_mid_term_frames": "--min_mid_term_frames",
    "max_long_term_elements": "--max_long_term_elements",
    "num_prototypes": "--num_prototypes",
    "top_k": "--top_k",
    "mem_every": "--mem_every",
    "deep_update_every": "--deep_update_every",
    "save_scores": "--save_scores",                          # bool
    "flip": "--flip",                                        # bool
    "size": "--size",
    "reverse": "--reverse",                                  # bool
}

def build_args_list_for_test(d16_batch_path: str,
                             out_path: str,
                             ref_root: str,
                             cfg: dict):
    args = [
        "--d16_batch_path", d16_batch_path,
        "--ref_path", ref_root,
        "--output", out_path,
    ]
    for k, v in cfg.items():
        if k not in CONFIG_TO_CLI:
            continue
        flag = CONFIG_TO_CLI[k]
        if isinstance(v, bool):
            if v:
                args.append(flag)
        elif v is None:
            continue
        else:
            args.extend([flag, str(v)])
    return args

# ----------------- GRADIO HANDLER (Local GPU) -----------------
def gradio_infer(
    debug_shapes,
    gpu_id,                   # UI: GPU ID
    input_images,             # list of uploaded image files (gr.Files)
    ref_image,                # single reference image (PIL)
    first_not_exemplar, dataset, split, save_all, benchmark,
    disable_long_term, max_mid, min_mid, max_long,
    num_proto, top_k, mem_every, deep_update,
    save_scores, flip, size, reverse
):
    # set visible CUDA devices before anything triggers CUDA init
    if gpu_id is None:
        gpu_id = 0
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    except Exception as e:
        print(f"[WARN] set_device failed or CUDA not available: {e}")

    # basic checks
    if not input_images or len(input_images) == 0:
        return None, "请上传一张或多张黑白输入图像 / Please upload one or more B&W input images."
    if ref_image is None:
        return None, "请上传参考图像 / Please upload a reference image."

    # reset temp workspace
    reset_temp_root()

    # create a unique run stem (timestamp)
    run_stem = f"run_{int(time.time())}"
    input_root = path.join(TEMP_ROOT, INPUT_DIR)     # _colormnet_tmp/input_images
    ref_root   = path.join(TEMP_ROOT, REF_DIR)       # _colormnet_tmp/ref
    output_root= path.join(TEMP_ROOT, OUTPUT_DIR)    # _colormnet_tmp/output
    input_images_dir = path.join(input_root, run_stem)
    ref_dir = path.join(ref_root, run_stem)
    out_frames_dir = path.join(output_root, run_stem)
    for d in (input_root, ref_root, output_root, input_images_dir, ref_dir, out_frames_dir):
        ensure_dir(d)

    # save uploaded input images into input_images/<run_stem>/00000.png ...
    try:
        saved_count = save_uploaded_images(input_images, input_images_dir)
        if saved_count == 0:
            return None, "未能保存任意输入图像 / Failed to save any input images."
    except Exception as e:
        return None, f"保存输入图像失败 / Failed to save input images:\n{e}"

    # save reference image as ref/<run_stem>/ref.png
    ref_png_path = path.join(ref_dir, "ref.png")
    if isinstance(ref_image, Image.Image):
        try:
            ref_image.save(ref_png_path)
        except Exception as e:
            return None, f"保存参考图像失败 / Failed to save reference image:\n{e}"
    elif isinstance(ref_image, str):
        try:
            shutil.copy2(ref_image, ref_png_path)
        except Exception as e:
            return None, f"复制参考图像失败 / Failed to copy reference image:\n{e}"
    else:
        return None, "无法读取参考图像输入 / Failed to read reference image."

    # user config (same defaults as original)
    default_config = {
        "FirstFrameIsNotExemplar": True,
        "dataset": "D16_batch",
        "split": "val",
        "save_all": True,
        "benchmark": False,
        "disable_long_term": False,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "save_scores": False,
        "flip": False,
        "size": -1,
        "reverse": False,
    }
    user_config = {
        "FirstFrameIsNotExemplar": bool(first_not_exemplar) if first_not_exemplar is not None else default_config["FirstFrameIsNotExemplar"],
        "dataset": str(dataset) if dataset else default_config["dataset"],
        "split": str(split) if split else default_config["split"],
        "save_all": bool(save_all) if save_all is not None else default_config["save_all"],
        "benchmark": bool(benchmark) if benchmark is not None else default_config["benchmark"],
        "disable_long_term": bool(disable_long_term) if disable_long_term is not None else default_config["disable_long_term"],
        "max_mid_term_frames": int(max_mid) if max_mid is not None else default_config["max_mid_term_frames"],
        "min_mid_term_frames": int(min_mid) if min_mid is not None else default_config["min_mid_term_frames"],
        "max_long_term_elements": int(max_long) if max_long is not None else default_config["max_long_term_elements"],
        "num_prototypes": int(num_proto) if num_proto is not None else default_config["num_prototypes"],
        "top_k": int(top_k) if top_k is not None else default_config["top_k"],
        "mem_every": int(mem_every) if mem_every is not None else default_config["mem_every"],
        "deep_update_every": int(deep_update) if deep_update is not None else default_config["deep_update_every"],
        "save_scores": bool(save_scores) if save_scores is not None else default_config["save_scores"],
        "flip": bool(flip) if flip is not None else default_config["flip"],
        "size": int(size) if size is not None else default_config["size"],
        "reverse": bool(reverse) if reverse is not None else default_config["reverse"],
    }

    # optional pre-download checkpoint
    ensure_checkpoint()

    # import test_app and call run_cli (same as original)
    try:
        import test_app as test
    except Exception as e:
        return None, f"导入 test.py 失败 / Failed to import test.py：\n{e}"

    args_list = build_args_list_for_test(
        d16_batch_path=input_root,   # root of input images (test_app should find input_images/<run_stem>)
        out_path=output_root,        # test_app will write to output/<run_stem>/*.png
        ref_root=ref_root,           # ref/<run_stem>/ref.png
        cfg=user_config
    )

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            entry = getattr(test, "run_cli", None)
            if entry is None or not callable(entry):
                raise RuntimeError("test.py 未提供可调用的 run_cli(args_list) 接口。")
            entry(args_list)
        log = f"GPU_ID={gpu_id} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} \n" \
              f"Args: {' '.join(args_list)}\n\n{buf.getvalue()}"
    except Exception as e:
        log = f"GPU_ID={gpu_id} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} \n" \
              f"Args: {' '.join(args_list)}\n\n{buf.getvalue()}\n\nERROR: {e}"
        return None, log

    # try to release GPU memory
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # after test_app, collect output images from output/<run_stem>/
    if not path.isdir(out_frames_dir):
        return None, f"未找到输出帧目录 / Output frame dir not found：{out_frames_dir}\n\n{log}"

    # create zip for download
    zip_path = path.abspath(path.join(TEMP_ROOT, f"output_{run_stem}.zip"))
    try:
        zip_output_folder(out_frames_dir, zip_path)
    except Exception as e:
        return None, f"打包输出失败 / Failed to zip output:\n{e}\n\n{log}"

    return zip_path, f"完成 ✅ / Done ✅\n\n{log}"

# ----------------- UI (Gradio) -----------------
with gr.Blocks() as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESC)

    with gr.Row():
        gpu_id = gr.Number(label="GPU ID (e.g., 0 for cuda:0)", value=0, precision=0)
        debug_shapes = gr.Checkbox(label="Debug Logs", value=False)

    with gr.Row():
        inp_images = gr.Files(label="输入黑白图像（可多选） / Input B&W image(s)", file_count="multiple")
        inp_ref = gr.Image(label="参考图像（RGB） / Reference Image (RGB)", type="pil")

    with gr.Accordion("Advanced Settings (passed to test_app)", open=False):
        with gr.Row():
            first_not_exemplar = gr.Checkbox(label="FirstFrameIsNotExemplar (--FirstFrameIsNotExemplar)", value=True)
            reverse = gr.Checkbox(label="reverse (--reverse)", value=False)
            dataset = gr.Textbox(label="dataset (--dataset)", value="D16_batch")
            split = gr.Textbox(label="split (--split)", value="val")
            save_all = gr.Checkbox(label="save_all (--save_all)", value=True)
            benchmark = gr.Checkbox(label="benchmark (--benchmark)", value=False)
        with gr.Row():
            disable_long_term = gr.Checkbox(label="disable_long_term (--disable_long_term)", value=False)
            max_mid = gr.Number(label="max_mid_term_frames (--max_mid_term_frames)", value=10, precision=0)
            min_mid = gr.Number(label="min_mid_term_frames (--min_mid_term_frames)", value=5, precision=0)
            max_long = gr.Number(label="max_long_term_elements (--max_long_term_elements)", value=10000, precision=0)
            num_proto = gr.Number(label="num_prototypes (--num_prototypes)", value=128, precision=0)
        with gr.Row():
            top_k = gr.Number(label="top_k (--top_k)", value=30, precision=0)
            mem_every = gr.Number(label="mem_every (--mem_every)", value=5, precision=0)
            deep_update = gr.Number(label="deep_update_every (--deep_update_every)", value=-1, precision=0)
            save_scores = gr.Checkbox(label="save_scores (--save_scores)", value=False)
            flip = gr.Checkbox(label="flip (--flip)", value=False)
            size = gr.Number(label="size (--size)", value=-1, precision=0)

    run_btn = gr.Button("开始着色 / Start Coloring")
    with gr.Row():
        out_file = gr.File(label="输出：压缩的着色图像 / Output (zip of colorized images)")
        status = gr.Textbox(label="状态 / 日志输出 / Status & Logs", interactive=False, lines=16)

    run_btn.click(
        fn=gradio_infer,
        inputs=[
            debug_shapes,
            gpu_id,
            inp_images, inp_ref,
            first_not_exemplar, dataset, split, save_all, benchmark,
            disable_long_term, max_mid, min_mid, max_long,
            num_proto, top_k, mem_every, deep_update,
            save_scores, flip, size, reverse
        ],
        outputs=[out_file, status]
    )

if __name__ == "__main__":
    try:
        ensure_checkpoint()
    except Exception as e:
        print(f"[WARN] Pre-download failed: {e}")
    demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860, share=False)
