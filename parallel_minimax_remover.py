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
    # 确保输入数据在正确的设备上
    pixel_values = pixel_values.to(device)
    masks = masks.to(device)
    
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

def compute_output_path(item_data, base_path):
    """统一的输出路径计算函数 - 保持原有逻辑"""
    mask_dir = os.path.dirname(item_data["edited_video"])
    mask_filename = os.path.basename(item_data["edited_video"])
    
    if "_mask_" in mask_filename:
        output_filename = mask_filename.replace("_mask_", "_rem_")
    else:
        # Fallback: add _rem before extension
        name, ext = os.path.splitext(mask_filename)
        output_filename = f"{name}_rem{ext}"
    
    return os.path.join(base_path, mask_dir, output_filename)

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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # DDP initialization
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    print(f"[Rank {rank}] Initialized on device {device}")
    
    # Load test data from JSON
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    
    # Convert to list of items
    items = list(test_data.items())

    # Pre-filter: skip items whose output has already been generated
    pending_items = []
    done_items = []
    for key, item_data in items:
        out_path = compute_output_path(item_data, args.base_path)
        if os.path.exists(out_path):
            done_items.append((key, item_data))
        else:
            # 检查输入文件是否存在
            original_video_path = os.path.join(args.base_path, item_data["original_video"])
            mask_video_path = os.path.join(args.base_path, item_data["edited_video"])
            
            if not os.path.exists(original_video_path):
                if rank == 0:
                    print(f"[Filter] Original video not found for {key}: {original_video_path}")
                continue
            
            if not os.path.exists(mask_video_path):
                if rank == 0:
                    print(f"[Filter] Mask video not found for {key}: {mask_video_path}")
                continue
                
            pending_items.append((key, item_data))

    if rank == 0:
        print(f"[Filter] Total: {len(items)}, Done: {len(done_items)}, Pending: {len(pending_items)}")

    # 如果没有待处理任务，退出
    if len(pending_items) == 0:
        print(f"[Rank {rank}] No pending items to process, exiting...")
        dist.destroy_process_group()
        return

    # Distribute only pending items across ranks
    items = pending_items
    per_rank = len(items) // world_size
    start_idx = rank * per_rank
    end_idx = (rank + 1) * per_rank if rank != world_size - 1 else len(items)
    subset = items[start_idx:end_idx]

    print(f"[Rank {rank}] Processing {len(subset)} pending items ({start_idx} to {max(end_idx-1, start_idx)})")
    
    # Load MiniMax Remover models on CPU first (following pattern from doc 4)
    print(f"[Rank {rank}] Loading MiniMax Remover models on CPU...")
    
    # Load models to CPU with explicit device mapping
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_path, "vae"), 
        torch_dtype=torch.float16,
        device_map={"": "cpu"}  # 明确指定加载到CPU
    )
    
    transformer = Transformer3DModel.from_pretrained(
        os.path.join(args.model_path, "transformer"), 
        torch_dtype=torch.float16,
        device_map={"": "cpu"}  # 明确指定加载到CPU
    )
    
    scheduler = UniPCMultistepScheduler.from_pretrained(
        os.path.join(args.model_path, "scheduler")
    )
    
    # Create pipeline on CPU
    pipe = Minimax_Remover_Pipeline(
        transformer=transformer, 
        vae=vae, 
        scheduler=scheduler
    )
    
    # Now move the entire pipeline to the target GPU
    print(f"[Rank {rank}] Moving models to {device}...")
    pipe = pipe.to(device)
    
    # Set evaluation mode
    pipe.vae.eval()
    pipe.transformer.eval()
    
    print(f"[Rank {rank}] Models loaded successfully on {device}")
    
    # Process each item
    successful = 0
    failed = 0
    
    for key, item_data in subset:
        try:
            # Construct file paths
            original_video_path = os.path.join(args.base_path, item_data["original_video"])
            mask_video_path = os.path.join(args.base_path, item_data["edited_video"])
            output_video_path = compute_output_path(item_data, args.base_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            
            # Double-check if output already exists
            if os.path.exists(output_video_path):
                print(f"[Rank {rank}] Output already exists for {key}, skipping...")
                continue
            
            print(f"[Rank {rank}] Processing {key}...")
            
            # Load video and mask - load to CPU first, then move to device
            print(f"[Rank {rank}] Loading video: {original_video_path}")
            images = load_video(original_video_path, args.video_length)
            images = images.to(device)
            
            print(f"[Rank {rank}] Loading mask: {mask_video_path}")
            masks = load_mask(mask_video_path, args.video_length)
            masks = masks.to(device)
            
            # Run inference
            print(f"[Rank {rank}] Running inference...")
            with torch.no_grad():
                video = inference(
                    pipe, images, masks, device, 
                    args.video_length, 
                    random_seed=args.seed,
                    iterations=args.iterations
                )
            
            # Save result with atomic write
            temp_output_path = output_video_path.replace(".mp4", "_temp.mp4")
            print(f"[Rank {rank}] Saving result to: {output_video_path}")
            export_to_video(video, temp_output_path)
            
            # Atomic rename
            os.rename(temp_output_path, output_video_path)
            
            # Save info file
            info_path = output_video_path.replace(".mp4", "_info.txt")
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"Key: {key}\n")
                f.write(f"Original video: {original_video_path}\n")
                f.write(f"Mask video: {mask_video_path}\n")
                f.write(f"Edit instruction: {item_data['edit_instruction']}\n")
                f.write(f"Video length: {args.video_length}\n")
                f.write(f"Iterations: {args.iterations}\n")
                f.write(f"Seed: {args.seed}\n")
                f.write(f"Processed by rank: {rank}\n")
            
            print(f"[Rank {rank}] Successfully processed {key}")
            successful += 1
            
            # 清理GPU缓存
            if successful % 5 == 0:  # 每5个视频清理一次
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Rank {rank}] Error processing {key}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            
            # 错误时也清理缓存
            torch.cuda.empty_cache()
            continue
    
    print(f"[Rank {rank}] Finished processing: {successful} successful, {failed} failed")
    
    # 同步所有rank
    dist.barrier()
    
    # 收集统计信息
    if rank == 0:
        all_successful = successful
        all_failed = failed
        for r in range(1, world_size):
            recv_success = torch.tensor([0], dtype=torch.int32, device=device)
            recv_fail = torch.tensor([0], dtype=torch.int32, device=device)
            dist.recv(recv_success, src=r)
            dist.recv(recv_fail, src=r)
            all_successful += recv_success.item()
            all_failed += recv_fail.item()
        
        print(f"\n[Summary] Total processed: {all_successful} successful, {all_failed} failed")
    else:
        dist.send(torch.tensor([successful], dtype=torch.int32, device=device), dst=0)
        dist.send(torch.tensor([failed], dtype=torch.int32, device=device), dst=0)
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()