#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parallel_minimax_remover.py
改进版：修复多卡并行时“Expected all tensors to be on the same device”的常见错误。
用法示例：
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 parallel_minimax_remover.py \
    --test_json grounding_multi_instance_gray.json \
    --base_path /scratch3/yan204/yxp/Senorita/ \
    --model_path ./minimax-remover \
    --video_length 81 \
    --iterations 6
"""
import os
import sys
import json
import argparse
import math
import traceback
import torch
import torch.distributed as dist
from decord import VideoReader, cpu
import numpy as np
from diffusers.utils import export_to_video
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler

# 你的自定义模块（和原脚本相同路径）
from transformer_minimax_remover import Transformer3DModel
from pipeline_minimax_remover import Minimax_Remover_Pipeline

# ---------------------- utils ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Parallel MiniMax Remover inference (fixed device handling)")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--base_path", type=str, default="/projects/D2DCRC/xiangpeng/Filter_Video_In_context_data",
                        help="Base path for video files")
    parser.add_argument("--model_path", type=str, default="./minimax-remover", help="Path to MiniMax Remover model")
    parser.add_argument("--video_length", type=int, default=81, help="Video length in frames")
    parser.add_argument("--iterations", type=int, default=6, help="Iterations for mask dilation")
    parser.add_argument("--num_inference_steps", type=int, default=12, help="Diffusion inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

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

def inference(pipe, pixel_values, masks, device, video_length=81, random_seed=42,
              num_inference_steps=12, iterations=6):
    """
    调用 pipeline。假设 pipe 在正确 device 上并且 pixel_values/masks 已经被移动到 device。
    """
    # 确保 generator 在正确 device
    gen = torch.Generator(device=device).manual_seed(random_seed)
    out = pipe(
        images=pixel_values,
        masks=masks,
        num_frames=video_length,
        height=pixel_values.shape[-2],
        width=pixel_values.shape[-1],
        num_inference_steps=num_inference_steps,
        generator=gen,
        iterations=iterations
    )
    # out.frames 很多实现会返回形状 (B, T, H, W, C) 或 list，使用你的原用法保留 .frames[0]
    return out.frames[0]

# ---------------------- main ----------------------
def main():
    args = parse_args()

    # init distributed
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # prefer LOCAL_RANK (torchrun 会设置)，否则 fallback 到 global_rank
    local_rank = int(os.environ.get("LOCAL_RANK", global_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[Rank {global_rank} Local {local_rank}] start. world_size={world_size}, device={device}")

    # load test json
    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    items = list(test_data.items())

    # 更稳健的分配：按 idx % world_size 分配
    subset = [items[i] for i in range(len(items)) if (i % world_size) == global_rank]
    print(f"[Rank {global_rank}] assigned {len(subset)} items")

    # load models -> 强制移动到当前 GPU
    print(f"[Rank {global_rank}] Loading models to {device} ...")
    try:
        vae = AutoencoderKLWan.from_pretrained(os.path.join(args.model_path, "vae"),
                                              torch_dtype=torch.float16)
    except Exception as e:
        print(f"[Rank {global_rank}] Failed to load vae: {e}")
        raise

    transformer = Transformer3DModel.from_pretrained(os.path.join(args.model_path, "transformer"),
                                                     torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained(os.path.join(args.model_path, "scheduler"))

    # move to device
    vae.to(device)
    transformer.to(device)
    # pipeline
    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(device)

    # diagnostic: print param device
    try:
        print(f"[diag Rank {global_rank}] vae param device: {next(vae.parameters()).device}")
        print(f"[diag Rank {global_rank}] transformer param device: {next(transformer.parameters()).device}")
    except StopIteration:
        pass

    # iterate items
    for idx, (key, item_data) in enumerate(subset):
        try:
            print(f"[Rank {global_rank}] [{idx+1}/{len(subset)}] Processing {key} ...")
            original_video_path = os.path.join(args.base_path, item_data["original_video"])
            mask_video_path = os.path.join(args.base_path, item_data["edited_video"])

            # build output filename / path (与你原逻辑一致)
            mask_dir = os.path.dirname(item_data["edited_video"])
            mask_filename = os.path.basename(item_data["edited_video"])
            if "_mask_" in mask_filename:
                output_filename = mask_filename.replace("_mask_", "_rem_")
            else:
                name, ext = os.path.splitext(mask_filename)
                output_filename = f"{name}_rem{ext}"
            output_video_path = os.path.join(args.base_path, mask_dir, output_filename)
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            if os.path.exists(output_video_path):
                print(f"[Rank {global_rank}] Output exists, skip: {output_video_path}")
                continue

            # check inputs
            if not os.path.exists(original_video_path):
                print(f"[Rank {global_rank}] Original not found: {original_video_path}")
                continue
            if not os.path.exists(mask_video_path):
                print(f"[Rank {global_rank}] Mask not found: {mask_video_path}")
                continue

            # load
            print(f"[Rank {global_rank}] Loading video: {original_video_path}")
            images = load_video(original_video_path, args.video_length)
            if images is None:
                print(f"[Rank {global_rank}] Skip {key}: video too short")
                continue

            print(f"[Rank {global_rank}] Loading mask: {mask_video_path}")
            masks = load_mask(mask_video_path, args.video_length)
            if masks is None:
                print(f"[Rank {global_rank}] Skip {key}: mask too short")
                continue

            # move inputs to device and ensure dtype
            images = images.to(device=device, dtype=torch.float32)  # shape (1,C,T,H,W)
            masks = masks.to(device=device, dtype=torch.float32)    # shape (1,1,T,H,W)

            # diag
            print(f"[diag {global_rank}] images device: {images.device}, masks device: {masks.device}")
            # run inference
            print(f"[Rank {global_rank}] Running inference ...")
            video_out = inference(pipe, images, masks, device,
                                  video_length=args.video_length,
                                  random_seed=args.seed,
                                  num_inference_steps=args.num_inference_steps,
                                  iterations=args.iterations)

            # save result - export_to_video 需要 numpy 或 list of frames
            print(f"[Rank {global_rank}] Saving result to: {output_video_path}")
            # 如果 video_out 是 torch.Tensor，转为 numpy；假设 shape (T, H, W, C) 或 (B?, T, H, W, C)
            if isinstance(video_out, torch.Tensor):
                v = video_out.detach().cpu().numpy()
            else:
                v = np.array(video_out)

            # 若 v 维度是 (B, T, H, W, C) 取第0个
            if v.ndim == 5 and v.shape[0] == 1:
                v = v[0]
            # export_to_video 希望 uint8 或 float in [-1,1]? diffusers.utils 会处理，但我们转换到 uint8
            # v assumed in [-1,1], convert back to [0,255]
            if v.dtype == np.float32 or v.dtype == np.float64:
                v = ((v + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            # ensure (T,H,W,C)
            export_to_video(v, output_video_path)

            # write info file
            info_path = os.path.splitext(output_video_path)[0] + "_info.txt"
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"Key: {key}\n")
                f.write(f"Original video: {original_video_path}\n")
                f.write(f"Mask video: {mask_video_path}\n")
                f.write(f"Edit instruction: {item_data.get('edit_instruction','')}\n")
                f.write(f"Video length: {args.video_length}\n")
                f.write(f"Iterations: {args.iterations}\n")
            print(f"[Rank {global_rank}] Successfully processed {key}")

        except Exception as e:
            print(f"[Rank {global_rank}] Error processing {key}: {e}")
            traceback.print_exc()
            # 继续下一个 item
            continue

    print(f"[Rank {global_rank}] All done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
