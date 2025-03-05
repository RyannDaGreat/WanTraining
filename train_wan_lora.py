import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import gc
import math
import random
import numpy as np
import argparse
import json
import datetime
from tqdm import tqdm
from contextlib import contextmanager
from time import perf_counter
from glob import glob

from safetensors.torch import load_file, save_file
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb
# from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from utils.temp_rng import temp_rng
from utils.dataset import CombinedDataset


def make_dir(base, folder):
    new_dir = os.path.join(base, folder)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


@contextmanager
def load_timer(target):
    print(f"loading {target}...")
    start_time = perf_counter()
    yield
    end_time = perf_counter()
    print(f"loaded {target} in {end_time - start_time:0.2f} seconds")


def download_model(args):
    from huggingface_hub import snapshot_download
    
    # get text encoder, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "google/*",
    )
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "models_t5_umt5-xxl-enc-bf16.pth",
        max_workers = 1,
    )
    
    # get vae, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/vae",
        allow_patterns = "Wan2.1_VAE.pth",
        max_workers = 1,
    )
    
    # get clip vision if it's i2v
    if "-I2V-" in args.pretrained_model_name_or_path:
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            max_workers = 1,
        )
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "xlm-roberta-large/*",
        )
    
    # get the correct diffusion model for HF identifier
    snapshot_download(
        repo_type = "model",
        repo_id = args.pretrained_model_name_or_path,
        local_dir = "./models/" + args.pretrained_model_name_or_path,
        allow_patterns = ["config.json", "diffusion_pytorch_model*"],
        max_workers = 1,
    )

@torch.inference_mode()
def cache_embeddings(args):
    if os.path.exists("./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"):
        with load_timer("UMT5 text encoder"):
            if os.path.exists("./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"):
                ckpt_dir = "./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"
                tk_dir = "./models/UMT5/google/umt5-xxl"
            else:
                ckpt_dir = args.pretrained_model_name_or_path
                tk_dir = args.pretrained_model_name_or_path + "/google/umt5-xxl"
            
            umt5_model = T5EncoderModel(
                text_len = 512,
                dtype = torch.bfloat16,
                device = device,
                checkpoint_path = ckpt_dir,
                tokenizer_path = tk_dir,
            )
    else:
        raise Exception("UMT5 model missing, download it first with --download_model")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # it would probably be better to dirwalk and batch the encoding
    # scanning all files at the start can be slow, and encoding one prompt at a time is inefficient
    caption_files = glob(os.path.join(args.dataset, "**", "*.txt" ), recursive=True)
    for file in tqdm(caption_files, desc="encoding captions"):
        embedding_path = os.path.splitext(file)[0] + "_wan.safetensors"
        
        if not os.path.exists(embedding_path):
            with open(file, "r") as f:
                caption = f.read()
            
            context = umt5_model([caption], umt5_model.device)[0]
            embedding_dict = {"context": context}
            save_file(embedding_dict, embedding_path)
    
    del umt5_model
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_output_dir = make_dir(args.output_dir, date_time)
    checkpoint_dir = make_dir(real_output_dir, "checkpoints")
    t_writer = SummaryWriter(log_dir=real_output_dir, flush_secs=60)
    with open(os.path.join(real_output_dir, "command_args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    train_dataset = os.path.join(args.dataset, "train")
    if not os.path.exists(train_dataset):
        train_dataset = args.dataset
        print(f"WARNING: train subfolder not found, using root folder {train_dataset} as train dataset")
    
    val_dataset = None
    for subfolder in ["val", "validation", "test"]:
        subfolder_path = os.path.join(args.dataset, subfolder)
        if os.path.exists(subfolder_path):
            val_dataset = subfolder_path
            break
    
    if val_dataset is None:
        val_dataset = args.dataset
        print(f"WARNING: val/validation/test subfolder not found, using root folder {val_dataset} for stable loss validation")
        print("\033[33mThis will make it impossible to judge overfitting by the validation loss. Using a val split held out from training is highly recommended\033[m")
    
    with load_timer("train dataset"):
        train_dataset = CombinedDataset(
            root_folder = train_dataset,
            token_limit = args.token_limit,
            max_frame_stride = args.max_frame_stride,
            bucket_resolution = args.base_res,
        )
    with load_timer("validation dataset"):
        val_dataset = CombinedDataset(
            root_folder = val_dataset,
            token_limit = args.token_limit,
            limit_samples = args.val_samples,
            max_frame_stride = args.max_frame_stride,
            bucket_resolution = args.base_res,
        )
    
    def collate_batch(batch):
        pixels = [sample["pixels"][0].movedim(0, 1) for sample in batch] # BFCHW -> FCHW -> CFHW
        context = [sample["embedding_dict"]["context"] for sample in batch]
        return pixels, context
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = False,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    
    with load_timer("VAE"):
        if os.path.exists("./models/vae/Wan2.1_VAE.pth"):
            ckpt_dir = "./models/vae/Wan2.1_VAE.pth"
        else:
            ckpt_dir = args.pretrained_model_name_or_path
        
        vae = WanVAE(vae_pth=ckpt_dir, dtype=torch.bfloat16, device=device)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    with load_timer("diffusion model"):
        if os.path.exists("./models/" + args.pretrained_model_name_or_path):
            ckpt_dir = "./models/" + args.pretrained_model_name_or_path
        else:
            ckpt_dir = args.pretrained_model_name_or_path
        
        diffusion_model = WanModel.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).to(device)
        diffusion_model.requires_grad_(False)
        
        if args.gradient_checkpointing:
            diffusion_model.enable_gradient_checkpointing()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    
    if args.fuse_lora is not None:
        loaded_lora_sd = load_file(args.fuse_lora)
        diffusion_model.load_lora_adapter(loaded_lora_sd, adapter_name="fuse_lora")
        diffusion_model.fuse_lora(adapter_names="fuse_lora", lora_scale=args.fuse_lora_weight, safe_fusing=True)
        diffusion_model.unload_lora_weights()
    
    if args.lora_target == "attn":
        lora_params = []
        for name, param in diffusion_model.named_parameters():
            if name.endswith(".weight"):
                if args.lora_target in name and ".norm_" not in name:
                    lora_params.append(name.replace(".weight", ""))
    
    elif args.lora_target == "all-linear":
        lora_params = args.lora_target
    
    else: raise NotImplementedError(f"{args.lora_target}")
    
    lora_config = LoraConfig(
        r = args.lora_rank,
        lora_alpha = args.lora_alpha or args.lora_rank,
        init_lora_weights = "gaussian",
        target_modules = lora_params,
    )
    diffusion_model.add_adapter(lora_config)
    
    if args.init_lora is not None:
        loaded_lora_sd = load_file(args.init_lora)
        outcome = set_peft_model_state_dict(diffusion_model, loaded_lora_sd)
        if len(outcome.unexpected_keys) > 0:
            for key in outcome.unexpected_keys:
                print(f"not loaded: {key}")
    
    
    train_parameters = []
    total_parameters = 0
    for param in diffusion_model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
            train_parameters.append(param)
            total_parameters += param.numel()
    print(f"total trainable parameters: {total_parameters:,}")
    
    # Instead of having just one optimizer, we will have a dict of optimizers
    # for every parameter so we could reference them in our hook.
    optimizer_dict = {p: bnb.optim.AdamW8bit([p], lr=args.learning_rate) for p in train_parameters}
    
    # Define our hook, which will call the optimizer step() and zero_grad()
    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    
    # Register the hook onto every trainable parameter
    for p in train_parameters:
        p.register_post_accumulate_grad_hook(optimizer_hook)
    
    
    def prepare_conditions(batch):
        pixels  = [p.to(dtype=torch.bfloat16, device=device) for p in batch[0]]
        context = [c.to(dtype=torch.bfloat16, device=device) for c in batch[1]]
        
        latents = vae.encode(pixels)
        noise = [torch.randn_like(l) for l in latents]
        
        sigmas = torch.rand(len(latents)).to(device)
        timesteps = torch.round(sigmas * 1000).long()
        
        target = []
        noisy_model_input = []
        for i in range(len(latents)):
            target.append(noise[i] - latents[i])
            noisy = noise[i] * sigmas[i] + latents[i] * (1 - sigmas[i])
            noisy_model_input.append(noisy.to(torch.bfloat16))
        
        return {
            "target": target,
            "context": context,
            "timesteps": timesteps,
            "noisy_model_input": noisy_model_input,
        }
    
    def predict_loss(conditions):
        target = conditions["target"]
        c, f, h, w = conditions["noisy_model_input"][0].shape
        seq_len = math.ceil((h / 2) * (w / 2) * f)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = diffusion_model(
                x = conditions["noisy_model_input"],
                t = conditions["timesteps"],
                context = conditions["context"],
                seq_len = seq_len,
            )
        
        # loss = torch.stack([F.mse_loss(p, t) for p, t in zip(pred, target)])
        return F.mse_loss(torch.stack(pred), torch.stack(target))
    
    gc.collect()
    torch.cuda.empty_cache()
    diffusion_model.train()
    
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps))
    while global_step < args.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            start_step = perf_counter()
            with torch.inference_mode():
                conditions = prepare_conditions(batch)
            
            # torch.cuda.empty_cache()
            loss = predict_loss(conditions)
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step)
            loss.backward()
            
            t_writer.add_scalar("debug/step_time", perf_counter() - start_step, global_step)
            # torch.cuda.empty_cache()
            progress_bar.update(1)
            global_step += 1
            
            if global_step == 1 or global_step % args.val_steps == 0:
                with torch.inference_mode(), temp_rng(args.seed):
                    val_loss = 0.0
                    for step, batch in enumerate(tqdm(val_dataloader, desc="validation", leave=False)):
                        conditions = prepare_conditions(batch)
                        # torch.cuda.empty_cache()
                        loss = predict_loss(conditions)
                        val_loss += loss.detach().item()
                        # torch.cuda.empty_cache()
                    t_writer.add_scalar("loss/validation", val_loss / len(val_dataloader), global_step)
                progress_bar.unpause()
            
            if global_step >= args.max_train_steps or global_step % args.checkpointing_steps == 0:
                save_file(
                    get_peft_model_state_dict(diffusion_model),
                    os.path.join(checkpoint_dir, f"wan-lora-{global_step:08}.safetensors"),
                )
            
            if global_step >= args.max_train_steps:
                break


def parse_args():
    parser = argparse.ArgumentParser(
        description = "HunyuanVideo training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--download_model",
        action = "store_true",
        help = "auto download all necessary models to ./models if missing",
    )
    parser.add_argument(
        "--cache_embeddings",
        action = "store_true",
        help = "preprocess dataset to encode captions",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="Path to pretrained model or model identifier from huggingface",
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs",
        help = "Output directory for training results"
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = None,
        help = "Path to dataset directory with train and val subdirectories",
    )
    parser.add_argument(
        "--val_samples",
        type = int,
        default = 4,
        help = "Maximum number of samples to use for validation loss"
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training"
    )
    parser.add_argument(
        "--fuse_lora",
        type = str,
        default = None,
        help = "LoRA checkpoint to fuse into model, before adding a new trainable LoRA",
    )
    parser.add_argument(
        "--fuse_lora_weight",
        type = float,
        default = 1.0,
        help = "strength to merge --fuse_lora into the base model",
    )
    parser.add_argument(
        "--init_lora",
        type = str,
        default = None,
        help = "LoRA checkpoint to load instead of random init, must be the same rank and target layers",
    )
    parser.add_argument(
        "--lora_target",
        type = str,
        default = "attn",
        choices=["attn", "all-linear"],
        help = "layers to target with LoRA, default is attention only"
    )
    parser.add_argument(
        "--lora_rank",
        type = int,
        default = 16,
        help = "The dimension of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type = int,
        default = None,
        help = "The alpha value for LoRA, defaults to alpha=rank. Note: changing alpha will affect the learning rate, and if alpha=rank then changing rank will also affect learning rate",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action = "store_true",
        help = "use gradient checkpointing to reduce memory usage at the cost of speed",
    )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 1e-5,
        help = "Base learning rate",
    )
    parser.add_argument(
        "--base_res",
        type = int,
        default = 624,
        choices=[624, 960],
        help = "Base resolution bucket, resized to equal area based on aspect ratio"
    )
    parser.add_argument(
        "--token_limit",
        type = int,
        default = 15_000,
        help = "Combined resolution/frame limit based on transformer patch sequence length: (width // 16) * (height // 16) * ((frames - 1) // 4 + 1)"
    )
    parser.add_argument(
        "--max_frame_stride",
        type = int,
        default = 2,
        help = "1: use native framerate only. Higher values allow randomly choosing lower framerates (skipping frames to speed up the video)"
    )
    parser.add_argument(
        "--val_steps",
        type = int,
        default = 100,
        help = "Validate after every n steps",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type = int,
        default = 100,
        help = "Save a checkpoint of the training state every X steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type = int,
        default = 10_000,
        help = "Total number of training steps",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.download_model:
        download_model(args)
        exit()
    
    if args.dataset is not None:
        if args.cache_embeddings:
            cache_embeddings(args)
            exit()
        
        main(args)
    else:
        raise Exception("--dataset is required but not provided")