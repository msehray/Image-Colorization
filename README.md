# Video Colorization

**A Deep Spatial-Temporal Feature Propagation Network for Video Colorization**

Author: Mayank

## Demo

Try the online demo: [![google colab logo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s19qGVrcVopJho1NPNlB8veZsyXPDvml)

## Installation

```bash
# Create conda environment
conda create -n colormnet python=3.8 -y
conda activate colormnet

# Install PyTorch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install py-thin-plate-spline
git clone https://github.com/cheind/py-thin-plate-spline.git
cd py-thin-plate-spline && pip install -e . && cd ..

# Install Pytorch-Correlation-extension
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension && python setup.py install && cd ..

# Install other dependencies
pip install -r requirements.txt
```

## Pretrained Models

Download the pretrained model and put it in `./saves` directory (create the folder if it doesn't exist).

[Download Model](https://github.com/yyang181/colormnet/releases/download/v0.1/DINOv2FeatureV6_LocalAtten_s2_154000.pth)

## Usage

### Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
# Add --FirstFrameIsNotExemplar if the reference frame is not the first input image
```

**Note for Windows users**: If you encounter multiprocessor Runtime errors in the data loader, check [this solution](https://github.com/yyang181/colormnet/issues/5#issuecomment-2339263103).

### Run Gradio Demo

```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```

## Training

### Dataset Structure

Organize your training and validation data as follows:

```
data_root/
├── 001/
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── 002/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── ...
```

### Training Command

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --master_port 25205 \
    --nproc_per_node=1 \
    train.py \
    --exp_id DINOv2FeatureV6_LocalAtten_DAVISVidevo \
    --davis_root /path/to/your/training/data/ \
    --validation_root /path/to/your/validation/data \
    --savepath ./wandb_save_dir
```

## Evaluation

```bash
pip install lpips && python evaluation_matrics/evaluation.py
```

## License

This project is licensed under BY-NC-SA 4.0. See [LICENSES.md](LICENSES.md) for details.
