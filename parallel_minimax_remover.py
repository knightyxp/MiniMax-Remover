#!/usr/bin/env python
import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
import imageio
import numpy as np
from PIL import Image
from decord import VideoReader
from diffusers.utils import export_to_video
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

def load_video(video_path, video_length=81):
    """Load video frames"""
    vr = VideoReader(video_path)
    images = vr.get_batch(list(range(video_length))).asnumpy()
    images = torch.from_numpy(images)/127.5 - 1.0
    return images

def load_mask(mask_path, video_length=81):
    """Load and process mask video"""
    vr = VideoReader(mask_path)
    masks = vr.get_batch(list(range(video_length))).asnumpy()
    masks = torch.from_numpy(masks)
    
    # 对于彩色mask，使用RGB通道的最大值
    masks_gray = torch.max(masks, dim=-1, keepdim=True)[0]
    
    # 二值化处理
    masks_binary = masks_gray.clone()
    masks_binary[masks_binary > 10] = 255  # 非黑色区域设为白色
    masks_binary[masks_binary <= 10] = 0   # 黑色区域保持黑色
    
    masks_binary = masks_binary / 255.0
    return masks_binary

def inference(pipe, pixel_values, masks, device, video_length=81, random_seed=42, iterations=6):
    """Run MiniMax Remover inference"""
    video = pipe(
        images=pixel_values,
        masks=masks,
        num_frames=video_length,
        height=480,
        width=832,
        num_inference_steps=12,
        generator=torch.Generator(device=device).manual_seed(random_seed),
        iterations=iterations
    ).frames[0]
    return video

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MiniMax Remover inference")
    parser.add_argument("--test_json", type=str, required=True,
                        help="Path to test JSON file")
    parser.add_argument("--base_path", type=str, default="/projects/D2DCRC/xiangpeng/Filter_Video_In_context_data",
                        help="Base path for video files")
    parser.add_argument("--model_path", type=str, default="./minimax-remover",
                        help="Path to MiniMax Remover model")
    parser.add_argument("--video_length", type=int, default=81,
                        help="Video length in frames")
    parser.add_argument("--iterations", type=int, default=6,
                        help="Iterations for mask dilation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # DDP initialization
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    print(f"[Rank {rank}] Initialized on device {device}")
    
    # Load test data from JSON
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    
    # Convert to list of items for distribution
    items = list(test_data.items())
    per_rank = len(items) // world_size
    start_idx = rank * per_rank
    end_idx = (rank + 1) * per_rank if rank != world_size - 1 else len(items)
    subset = items[start_idx:end_idx]
    
    print(f"[Rank {rank}] Processing {len(subset)} items ({start_idx} to {end_idx-1})")
    
    # Load MiniMax Remover models
    print(f"[Rank {rank}] Loading MiniMax Remover models...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_path, "vae"), 
        torch_dtype=torch.float16
    )
    transformer = Transformer3DModel.from_pretrained(
        os.path.join(args.model_path, "transformer"), 
        torch_dtype=torch.float16
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        os.path.join(args.model_path, "scheduler")
    )
    
    pipe = Minimax_Remover_Pipeline(
        transformer=transformer, 
        vae=vae, 
        scheduler=scheduler
    )
    pipe.to(device)
    print(f"[Rank {rank}] Models loaded successfully")
    
    # Process each item
    for key, item_data in subset:
        try:
            print(f"[Rank {rank}] Processing {key}...")
            
            # Construct file paths
            original_video_path = os.path.join(args.base_path, item_data["original_video"])
            mask_video_path = os.path.join(args.base_path, item_data["edited_video"])
            
            # Generate output path
            # Extract directory and filename from mask path
            mask_dir = os.path.dirname(item_data["edited_video"])
            mask_filename = os.path.basename(item_data["edited_video"])
            
            if "_mask_" in mask_filename:
                output_filename = mask_filename.replace("_mask_", "_rem_")
            else:
                # Fallback: add _rem before extension
                name, ext = os.path.splitext(mask_filename)
                output_filename = f"{name}_rem{ext}"
            
            output_video_path = os.path.join(args.base_path, mask_dir, output_filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            
            # Check if output already exists
            if os.path.exists(output_video_path):
                print(f"[Rank {rank}] Output already exists for {key}, skipping...")
                continue
            
            # Check if input files exist
            if not os.path.exists(original_video_path):
                print(f"[Rank {rank}] Original video not found: {original_video_path}")
                continue
            
            if not os.path.exists(mask_video_path):
                print(f"[Rank {rank}] Mask video not found: {mask_video_path}")
                continue
            
            # Load video and mask
            print(f"[Rank {rank}] Loading video: {original_video_path}")
            images = load_video(original_video_path, args.video_length)
            
            print(f"[Rank {rank}] Loading mask: {mask_video_path}")
            masks = load_mask(mask_video_path, args.video_length)
            
            # Run inference
            print(f"[Rank {rank}] Running inference...")
            video = inference(
                pipe, images, masks, device, 
                args.video_length, iterations=args.iterations
            )
            
            # Save result
            print(f"[Rank {rank}] Saving result to: {output_video_path}")
            export_to_video(video, output_video_path)
            
            # Save info file
            info_path = output_video_path.replace(".mp4", "_info.txt")
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"Key: {key}\n")
                f.write(f"Original video: {original_video_path}\n")
                f.write(f"Mask video: {mask_video_path}\n")
                f.write(f"Edit instruction: {item_data['edit_instruction']}\n")
                f.write(f"Video length: {args.video_length}\n")
                f.write(f"Iterations: {args.iterations}\n")
            
            print(f"[Rank {rank}] Successfully processed {key}")
            
        except Exception as e:
            print(f"[Rank {rank}] Error processing {key}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[Rank {rank}] Finished processing all items")

if __name__ == "__main__":
    main()
