import torch
import imageio
import numpy as np
from diffusers.utils import export_to_video
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

random_seed = 42
video_length = 81
device = torch.device("cuda:0")

vae = AutoencoderKLWan.from_pretrained("./minimax-remover/vae", torch_dtype=torch.float16)
transformer = Transformer3DModel.from_pretrained("./minimax-remover/transformer", torch_dtype=torch.float16)
scheduler = UniPCMultistepScheduler.from_pretrained("./minimax-remover/scheduler")

pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
pipe.to(device)

# the iterations is the hyperparameter for mask dilation
def inference(pixel_values, masks, iterations=6):
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
    export_to_video(video, "./output.mp4")

def load_video(video_path):
    vr = VideoReader(video_path)
    images = vr.get_batch(list(range(video_length))).asnumpy()
    images = torch.from_numpy(images)/127.5 - 1.0
    return images

def load_mask(mask_path):
    vr = VideoReader(mask_path)
    masks = vr.get_batch(list(range(video_length))).asnumpy()
    
    # 保存原始彩色mask视频以便查看
    save_original_mask_video(masks, "./original_mask.mp4")
    
    masks = torch.from_numpy(masks)
    
    # 对于彩色mask，我们需要检测非黑色区域
    # 方法1: 使用RGB通道的最大值
    masks_gray = torch.max(masks, dim=-1, keepdim=True)[0]
    
    # 二值化处理
    masks_binary = masks_gray.clone()
    masks_binary[masks_binary > 10] = 255  # 非黑色区域设为白色
    masks_binary[masks_binary <= 10] = 0   # 黑色区域保持黑色
    
    
    masks_binary = masks_binary / 255.0
    return masks_binary

def save_original_mask_video(masks, output_path):
    """保存原始彩色mask视频"""
    masks_np = masks.astype(np.uint8)
    
    # 使用imageio保存视频
    with imageio.get_writer(output_path, fps=30) as writer:
        for frame in masks_np:
            writer.append_data(frame)
    
    print(f"原始彩色mask视频已保存到: {output_path}")
    print(f"原始视频形状: {masks_np.shape}")
    print(f"原始数据范围: {masks_np.min()} - {masks_np.max()}")

def save_processed_mask_video(masks, output_path):
    """保存处理后的mask视频"""
    # 将tensor转换为numpy数组
    if torch.is_tensor(masks):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks
    
    # 确保数据类型为uint8，范围在0-255
    masks_np = masks_np.astype(np.uint8)
    
    # 如果是单通道，扩展为3通道以便更好地可视化
    if masks_np.shape[-1] == 1:
        masks_np = np.repeat(masks_np, 3, axis=-1)
    
    # 使用imageio保存视频
    with imageio.get_writer(output_path, fps=30) as writer:
        for frame in masks_np:
            writer.append_data(frame)
    
    print(f"处理后的mask视频已保存到: {output_path}")
    print(f"视频形状: {masks_np.shape}")
    print(f"数据范围: {masks_np.min()} - {masks_np.max()}")

video_path = "/projects/D2DCRC/xiangpeng/Filter_Video_In_context_data/sample_videos/multi_instance_grounding_aligned/0011_src_e93f38782e58df3ef5adfae8ef6a8adb_org.mp4"
mask_path = "/projects/D2DCRC/xiangpeng/Filter_Video_In_context_data/sample_videos/multi_instance_grounding_aligned/0011_tgt_e93f38782e58df3ef5adfae8ef6a8adb.mp4"

images = load_video(video_path)
masks = load_mask(mask_path)

inference(images, masks)