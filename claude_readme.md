# WanTraining - Wan Video Model LoRA Training Guide

This repository contains code for training LoRA (Low-Rank Adaptation) models for the Wan AI video generation models. This guide covers setup, dataset preparation, training, and model usage.

## Table of Contents
- [Installation](#installation)
- [Model Download](#model-download)
- [Dataset Requirements](#dataset-requirements)
- [Training Process](#training-process)
- [Command Arguments](#command-arguments)
- [Training Examples](#training-examples)
- [Checkpoints](#checkpoints)
- [Using Trained Models](#using-trained-models)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/spacepxl/WanTraining
   cd WanTraining
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install PyTorch with CUDA support**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Model Download

Before training, you need to download the base models:

```bash
python train_wan_lora.py --download_model --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B
```

This downloads:
- The UMT5 text encoder
- VAE model
- Base diffusion model
- CLIP vision model (for I2V models only)

Available base models:
- `Wan-AI/Wan2.1-T2V-1.3B` (Smaller text-to-video model)
- `Wan-AI/Wan2.1-T2V-14B` (Larger text-to-video model)
- `Wan-AI/Wan2.1-I2V-14B-480P` (Image-to-video model)

## Dataset Requirements

The training code supports both image and video datasets with the following structure:

```
dataset/
├── train/
│   ├── video1.mp4
│   ├── video1.txt  (Caption for video1)
│   ├── image1.jpg
│   └── image1.txt  (Caption for image1)
└── val/  (or validation/ or test/)
    ├── video2.mp4
    ├── video2.txt
    └── ...
```

### Supported Media Formats
- Videos: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`
- Images: `.jpg`, `.png`

### Caption Files
Each media file should have a corresponding `.txt` file with the same name containing the prompt text.

### Generating Embeddings
Before training, you need to preprocess the dataset to generate embeddings:

```bash
python train_wan_lora.py --cache_embeddings --dataset /path/to/your/dataset
```

This creates `.safetensors` files alongside your media and caption files, containing pre-computed text embeddings.

## Training Process

Basic training command:

```bash
python train_wan_lora.py --dataset /path/to/your/dataset --output_dir ./outputs
```

### What Happens During Training

1. The script loads the required models (text encoder, VAE, diffusion model)
2. It creates a LoRA adapter for the specified layers of the diffusion model
3. Training processes batches of data:
   - Videos/images are encoded to latent space
   - Noise is added according to a random timestep
   - The model predicts the noise and is trained to minimize error
4. Checkpoints are saved at regular intervals
5. Training progress is logged to TensorBoard

## Command Arguments

### Basic Arguments
- `--dataset` - Path to your dataset directory
- `--output_dir` - Directory for saving results (default: `./outputs`)
- `--pretrained_model_name_or_path` - Model to fine-tune (default: `Wan-AI/Wan2.1-T2V-1.3B`)
- `--max_train_steps` - Total training steps (default: 1000)

### LoRA Configuration
- `--lora_rank` - Dimension of LoRA matrices (default: 16)
- `--lora_alpha` - Alpha parameter for LoRA (default: same as rank)
- `--lora_target` - Layers to target with LoRA (default: `attn`, options: `attn`, `all-linear`)
- `--learning_rate` - Base learning rate (default: 1e-5)

### Dataset Options
- `--token_limit` - Transformer sequence length limit (default: 10000)
- `--base_res` - Base resolution bucket (default: 624, options: 624, 960)
- `--max_frame_stride` - Maximum frame stride for sampling videos (default: 2)

### Control LoRA Training
- `--control_lora` - Enable training as a control LoRA
- `--control_type` - Type of control signal (default: `tile`)
- `--control_inject_noise` - Add noise to control latents (default: 0.0)

### Checkpointing and Validation
- `--checkpointing_steps` - Save checkpoint frequency (default: 100)
- `--val_steps` - Validation frequency (default: 100)
- `--val_samples` - Maximum samples for validation (default: 4)
- `--seed` - Random seed (default: 42)
- `--val_seed` - Separate validation seed (default: None)

### Advanced Options
- `--gradient_checkpointing` - Enable gradient checkpointing to save memory
- `--distill_cfg` - CFG scale for distillation (default: 0.0)
- `--fuse_lora` - LoRA checkpoint to fuse before training
- `--init_lora` - LoRA checkpoint to use as initialization

## Training Examples

### Basic Text-to-Video LoRA Training
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --max_train_steps 2000
```

### Training with Higher Resolution
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --base_res 960
```

### Training Control LoRA for Video Upscaling
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --control_lora --control_type tile
```

### Training with Memory Optimization
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --gradient_checkpointing
```

## Checkpoints

Checkpoints are saved in the specified output directory with the following structure:

```
./outputs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/
│   ├── wan-lora-00000100.safetensors
│   ├── wan-lora-00000200.safetensors
│   └── ...
├── command_args.json  (Record of training arguments)
└── events.out.tfevents.*  (TensorBoard logs)
```

## Using Trained Models

You can test your trained LoRA using the provided script:

```bash
python test_wan_control_lora.py --lora_path ./outputs/YYYY-MM-DD_HH-MM-SS/checkpoints/wan-lora-00000100.safetensors --prompt "Your prompt text"
```

For control LoRA testing, provide an input video:

```bash
python test_wan_control_lora.py --lora_path ./outputs/control_lora/checkpoints/wan-lora-00000100.safetensors --input_video ./test.mp4
```

For integrating with Comfy UI:

```bash
python convert_lora_to_comfy.py --input_lora ./outputs/YYYY-MM-DD_HH-MM-SS/checkpoints/wan-lora-00000100.safetensors --output_lora ./comfy_lora.safetensors
```

---

*This readme was generated to provide more detailed documentation for the WanTraining codebase.*