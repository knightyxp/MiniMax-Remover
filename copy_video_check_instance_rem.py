#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层采样并提取视频，同时为每个样本生成 info.txt (用于检查qwen生成的instruction)
"""
import os
import json
import shutil
import random
from pathlib import Path

# -------- 配置区域 --------
BASE_DATASET_DIR = "/scratch3/yan204/yxp/Senorita"
JSON_PATH = "gen_instruction/grounding_multi_instance_rem.json"
OUTPUT_DIR = "sample_videos/grounding_multi_instance_rem_check"
TOP_K = 4000        # 取 VIE score top K
LAYER_SIZE = 1000      # 每层数据大小
SAMPLES_PER_LAYER = 10  # 每层采样数


# -------------------------

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 JSON
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    # 提取结果列表
    if isinstance(raw, list):
        results = raw
    elif isinstance(raw, dict) and 'results' in raw:
        results = raw['results']
    else:
        raise RuntimeError("JSON 文件格式不符合预期，请检查数据格式")
    
    # 筛选并排序（假设已经按dover_score排序）
    valid = [itm for itm in results if itm.get('vie_overall_score') is not None]
    top_items = valid[:TOP_K]

    # 分层采样
    sampled = []
    for i in range((len(top_items) + LAYER_SIZE - 1) // LAYER_SIZE):
        chunk = top_items[i*LAYER_SIZE : (i+1)*LAYER_SIZE]
        count = min(SAMPLES_PER_LAYER, len(chunk))
        sampled.extend(random.sample(chunk, count))

    print(f"总样本: {len(sampled)} 条 (分层采样自 Top {TOP_K})")

    # 遍历采样结果，拷贝视频并生成 info.txt
    for idx, item in enumerate(sampled):
        idx_str = f"{idx:03d}"
        # 复制 source 和 target 视频
        for role, tag in (('source_video_path', 'src'), ('target_video_path', 'tgt')):
            rel = item.get(role, '').lstrip('./')
            # rel = rel.replace('/scratch3/yan204/yxp/', '/projects/D2DCRC/xiangpeng/')
            if not rel:
                continue
            
            # 对于source视频路径，将_org_reshape.mp4替换为_org.mp4
            # if role == 'source_video_path' and rel.endswith('_org_reshape.mp4'):
            #     rel = rel.replace('_org_reshape.mp4', '_org.mp4')
                #print('rel', rel)
            
            # 直接使用原始路径，不进行替换
            src_path = Path(BASE_DATASET_DIR) / rel
            #print(src_path)
            if not src_path.is_file():
                print(f"[警告] 文件不存在: {src_path}")
                continue
            dest_name = f"{idx_str}_{tag}_{src_path.name}"
            dest_path = Path(OUTPUT_DIR) / dest_name
            shutil.copy2(src_path, dest_path)

        # 生成 info.txt
        info_path = Path(OUTPUT_DIR) / f"{idx_str}_info.txt"
        with open(info_path, 'w', encoding='utf-8') as info_f:
            info_f.write(f"sample_id: {idx}\n")
            info_f.write(f"original_vie_rank: {valid.index(item) + 1}\n")
            info_f.write(f"vie_overall_score: {item.get('vie_overall_score')}\n")
            info_f.write(f"instruction: {item.get('instruction', '')}\n")
            info_f.write(f"enhanced_instruction: {item.get('enhanced_instruction', '')}\n")
            info_f.write(f"qwen_vl_72b_refined_instruction: {item.get('qwen_vl_72b_refined_instruction', '')}\n")
            info_f.write(f"source_video_path: {item.get('source_video_path', '')}\n")
            info_f.write(f"target_video_path: {item.get('target_video_path', '')}\n")
            info_f.write(f"multi_instances: {item.get('multi_instances', '')}\n")

    # 生成简要采样汇总
    summary = {
        'total_original': len(results),
        'valid_with_score': len(valid),
        'top_k_used': len(top_items),
        'final_sampled': len(sampled)
    }
    with open(Path(OUTPUT_DIR) / 'sample_summary.json', 'w', encoding='utf-8') as sum_f:
        json.dump(summary, sum_f, ensure_ascii=False, indent=2)

    print("处理完成，视频已拷贝，info.txt 已生成，采样汇总保存在 sample_summary.json")

if __name__ == '__main__':
    main()