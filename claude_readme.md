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
- [Codebase Summary](#codebase-summary)
- [Development History](#development-history)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/spacepxl/WanTraining
   cd WanTraining
   ```
   <sub>Source: README.md, lines 6-7</sub>

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate
   ```
   <sub>Source: README.md, lines 9-10, with added Linux/macOS variant</sub>

3. **Install PyTorch with CUDA support**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   <sub>Source: README.md, line 12</sub>

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   <sub>Source: README.md, line 14</sub>

## Model Download

Before training, you need to download the base models:

```bash
python train_wan_lora.py --download_model --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B
```
<sub>Source: train_wan_lora.py, lines 56-104 (download_model function), line 488 (default model)</sub>

This downloads:
- The UMT5 text encoder
- VAE model
- Base diffusion model
- CLIP vision model (for I2V models only)
<sub>Source: train_wan_lora.py, lines 58-97, each component downloaded separately</sub>

Available base models:
- `Wan-AI/Wan2.1-T2V-1.3B` (Smaller text-to-video model)
- `Wan-AI/Wan2.1-T2V-14B` (Larger text-to-video model)
- `Wan-AI/Wan2.1-I2V-14B-480P` (Image-to-video model)
<sub>Source: Inferred from model naming patterns and `wan/configs/` directory which has files for different model sizes</sub>

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
<sub>Source: train_wan_lora.py, lines 166-181 (folder structure checks) and utils/dataset.py, lines 84-94 (file searching)</sub>

### Supported Media Formats
- Videos: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`
- Images: `.jpg`, `.png`
<sub>Source: utils/dataset.py, lines 25-26 (format definitions)</sub>

### Caption Files
Each media file should have a corresponding `.txt` file with the same name containing the prompt text.
<sub>Source: utils/dataset.py, lines 169-177 (embedding file lookup) and train_wan_lora.py cache_embeddings function</sub>

### Generating Embeddings
Before training, you need to preprocess the dataset to generate embeddings:

```bash
python train_wan_lora.py --cache_embeddings --dataset /path/to/your/dataset
```
<sub>Source: train_wan_lora.py, lines 109-150 (cache_embeddings function), line 483 (cache_embeddings argument)</sub>

This creates `.safetensors` files alongside your media and caption files, containing pre-computed text embeddings.
<sub>Source: train_wan_lora.py, lines 136-144 (embedding creation)</sub>

## Training Process

Basic training command:

```bash
python train_wan_lora.py --dataset /path/to/your/dataset --output_dir ./outputs
```
<sub>Source: train_wan_lora.py, lines 153-468 (main function), line 495 (output_dir default)</sub>

### What Happens During Training

1. The script loads the required models (text encoder, VAE, diffusion model)
2. It creates a LoRA adapter for the specified layers of the diffusion model
3. Training processes batches of data:
   - Videos/images are encoded to latent space
   - Noise is added according to a random timestep
   - The model predicts the noise and is trained to minimize error
4. Checkpoints are saved at regular intervals
5. Training progress is logged to TensorBoard
<sub>Source: train_wan_lora.py, lines 224-248 (model loading), lines 293-309 (LoRA config), lines 354-369 (noise processing), lines 460-464 (checkpoint saving), line 162 (TensorBoard)</sub>

## Command Arguments

### Basic Arguments
- `--dataset` - Path to your dataset directory
  <sub>Source: train_wan_lora.py, line 499</sub>
- `--output_dir` - Directory for saving results (default: `./outputs`)
  <sub>Source: train_wan_lora.py, line 494</sub>
- `--pretrained_model_name_or_path` - Model to fine-tune (default: `Wan-AI/Wan2.1-T2V-1.3B`)
  <sub>Source: train_wan_lora.py, line 487</sub>
- `--max_train_steps` - Total training steps (default: 1000)
  <sub>Source: train_wan_lora.py, line 633</sub>

### LoRA Configuration
- `--lora_rank` - Dimension of LoRA matrices (default: 16)
  <sub>Source: train_wan_lora.py, line 567</sub>
- `--lora_alpha` - Alpha parameter for LoRA (default: same as rank)
  <sub>Source: train_wan_lora.py, line 572</sub>
- `--lora_target` - Layers to target with LoRA (default: `attn`, options: `attn`, `all-linear`)
  <sub>Source: train_wan_lora.py, line 541</sub>
- `--learning_rate` - Base learning rate (default: 1e-5)
  <sub>Source: train_wan_lora.py, line 584</sub>

### Dataset Options
- `--token_limit` - Transformer sequence length limit (default: 10000)
  <sub>Source: train_wan_lora.py, line 608</sub>
- `--base_res` - Base resolution bucket (default: 624, options: 624, 960)
  <sub>Source: train_wan_lora.py, line 602</sub>
- `--max_frame_stride` - Maximum frame stride for sampling videos (default: 2)
  <sub>Source: train_wan_lora.py, line 615</sub>

### Control LoRA Training
- `--control_lora` - Enable training as a control LoRA
  <sub>Source: train_wan_lora.py, line 547</sub>
- `--control_type` - Type of control signal (default: `tile`)
  <sub>Source: train_wan_lora.py, line 553</sub>
- `--control_inject_noise` - Add noise to control latents (default: 0.0)
  <sub>Source: train_wan_lora.py, line 559</sub>

### Checkpointing and Validation
- `--checkpointing_steps` - Save checkpoint frequency (default: 100)
  <sub>Source: train_wan_lora.py, line 627</sub>
- `--val_steps` - Validation frequency (default: 100)
  <sub>Source: train_wan_lora.py, line 621</sub>
- `--val_samples` - Maximum samples for validation (default: 4)
  <sub>Source: train_wan_lora.py, line 506</sub>
- `--seed` - Random seed (default: 42)
  <sub>Source: train_wan_lora.py, line 511</sub>
- `--val_seed` - Separate validation seed (default: None)
  <sub>Source: train_wan_lora.py, line 516</sub>

### Advanced Options
- `--gradient_checkpointing` - Enable gradient checkpointing to save memory
  <sub>Source: train_wan_lora.py, line 578</sub>
- `--distill_cfg` - CFG scale for distillation (default: 0.0)
  <sub>Source: train_wan_lora.py, line 589</sub>
- `--fuse_lora` - LoRA checkpoint to fuse before training
  <sub>Source: train_wan_lora.py, line 522</sub>
- `--init_lora` - LoRA checkpoint to use as initialization
  <sub>Source: train_wan_lora.py, line 534</sub>

## Training Examples

### Basic Text-to-Video LoRA Training
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --max_train_steps 2000
```
<sub>Source: Based on train_wan_lora.py arguments (dataset, output_dir, max_train_steps)</sub>

### Training with Higher Resolution
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --base_res 960
```
<sub>Source: train_wan_lora.py, line 602 (base_res options: 624, 960)</sub>

### Training Control LoRA for Video Upscaling
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --control_lora --control_type tile
```
<sub>Source: train_wan_lora.py, lines 547-553 (control_lora and control_type parameters)</sub>

### Training with Memory Optimization
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --gradient_checkpointing
```
<sub>Source: train_wan_lora.py, line 578 (gradient_checkpointing parameter)</sub>

### Training with CFG Distillation
```bash
python train_wan_lora.py --dataset ./my_videos --output_dir ./outputs --distill_cfg 1.0
```
<sub>Source: train_wan_lora.py, lines 389-421 (CFG distillation implementation), line 589 (distill_cfg parameter)</sub>

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
<sub>Source: train_wan_lora.py, lines 159-162 (output directory setup), lines 460-464 (checkpoint saving)</sub>

## Using Trained Models

You can test your trained LoRA using the provided script:

```bash
python test_wan_control_lora.py --lora_path ./outputs/YYYY-MM-DD_HH-MM-SS/checkpoints/wan-lora-00000100.safetensors --prompt "Your prompt text"
```
<sub>Source: test_wan_control_lora.py, inferred from command-line arguments</sub>

For control LoRA testing, provide an input video:

```bash
python test_wan_control_lora.py --lora_path ./outputs/control_lora/checkpoints/wan-lora-00000100.safetensors --input_video ./test.mp4
```
<sub>Source: test_wan_control_lora.py, inferred from command-line arguments for control LoRA</sub>

For integrating with Comfy UI:

```bash
python convert_lora_to_comfy.py --input_lora ./outputs/YYYY-MM-DD_HH-MM-SS/checkpoints/wan-lora-00000100.safetensors --output_lora ./comfy_lora.safetensors
```
<sub>Source: convert_lora_to_comfy.py, inferred from the file's purpose</sub>

## Codebase Summary

### Main Training and Inference Files

- **train_wan_lora.py**: Core training script for creating LoRA adaptations, handles dataset loading, model initialization, and training workflow.
  <sub>Source: File analysis, 653 lines, main training functionality</sub>
  
- **test_wan_control_lora.py**: Testing script for using trained LoRA models, supports both text-to-video generation and control-based video transformations.
  <sub>Source: Git history, commit 7b21707 added this file with 285 lines</sub>

- **convert_lora_to_comfy.py**: Utility to convert trained LoRAs to ComfyUI-compatible format by adjusting key prefixes.
  <sub>Source: Git history, commit 50cc6aa added this file with 62 lines</sub>

### Utility Modules

- **utils/dataset.py**: Implements the `CombinedDataset` class for handling both image and video data, with resolution bucketing and frame sampling.
  <sub>Source: File analysis, 181 lines, implements dataset handling</sub>
  
- **utils/temp_rng.py**: Context manager for temporarily setting random seeds to ensure consistent validation results.
  <sub>Source: train_wan_lora.py, line 449 uses this utility</sub>

### Wan Model Core Components

- **wan/text2video.py** & **wan/image2video.py**: Implementations for text-to-video and image-to-video generation.
  <sub>Source: Directory structure shows these as main model interfaces</sub>
  
- **wan/modules/model.py**: Core diffusion model implementation with transformer-based architecture.
  <sub>Source: File analysis and references in train_wan_lora.py line 34</sub>
  
- **wan/modules/vae.py**: Video Autoencoder for compressing videos to and from latent space.
  <sub>Source: File analysis and references in train_wan_lora.py line 34</sub>
  
- **wan/modules/t5.py**: UMT5 text encoder for processing text prompts into embeddings.
  <sub>Source: File analysis and references in train_wan_lora.py line 32</sub>
  
- **wan/modules/clip.py**: CLIP vision encoder for I2V models.
  <sub>Source: Directory structure, relevant for image-to-video models</sub>
  
- **wan/modules/attention.py**: Attention mechanisms for transformer models.
  <sub>Source: Directory structure and git history (commit b0f97bf modified this file)</sub>
  
- **wan/utils/fm_solvers*.py**: Flow matching solvers for diffusion process.
  <sub>Source: Referenced in train_wan_lora.py line 35</sub>
  
- **wan/configs/**: Configuration files for different model sizes and architectures.
  <sub>Source: Directory structure, contains configs for different model variants</sub>

### Embeddings

- **embeddings/**: Pre-computed negative prompt embeddings used for classifier-free guidance during training and inference.
  <sub>Source: File analysis and references in train_wan_lora.py line 596</sub>

## Development History

The repository has evolved through several development phases:

### Initial Setup (March 1, 2025)
- Initial repository setup
- Added Wan model source code
- Created dataset handling utilities
<sub>Source: Git history, commits 56f5c98, 7e57a29, 380a3be, 48debe1 (March 1)</sub>

### Model Support (March 4, 2025)
- Cleaned up model code
- Added PEFT (Parameter-Efficient Fine-Tuning) support for LoRA training
- Updated import order and requirements
- Implemented 1.3B model training functionality
- Added ComfyUI conversion script
<sub>Source: Git history, commits b0f97bf, 8d704b5, 50cc6aa (March 4)</sub>

### Advanced Training Features (March 6, 2025)
- Added CFG distillation training capabilities
- Implemented gradient checkpointing for memory optimization
- Added negative embeddings for guided training
- Enhanced model performance
<sub>Source: Git history, commit 9d4c1fd (March 6)</sub>

### Final Refinements (March 10-13, 2025)
- Completed training implementation
- Added testing functionality
- Added control-based LoRA support
- Implemented video transformation capabilities
- Updated documentation
<sub>Source: Git history, commits 7b21707 (March 10), a64a600 (March 13)</sub>

---

*This readme provides a comprehensive guide for WanTraining, a repository focused on fine-tuning the Wan AI video generation models using LoRA technique.*