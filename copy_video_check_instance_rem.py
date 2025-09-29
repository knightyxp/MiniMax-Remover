#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层采样并提取视频，同时为每个样本生成 info.txt (用于检查qwen生成的instruction)
"""
import os
import json
import shutil
import random
import re
from pathlib import Path

# -------- 配置区域 --------
BASE_DATASET_DIR = "/scratch3/yan204/yxp/Senorita"
JSON_PATH = "grounding_multi_instance_rem.json"
OUTPUT_DIR = "sample_videos/grounding_multi_instance_rem_check"
TOP_K = 4000        # 取 VIE score top K
LAYER_SIZE = 1000      # 每层数据大小
SAMPLES_PER_LAYER = 10  # 每层采样数


# -------------------------

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 JSON（兼容字典结构：key -> {original_video, edited_video, edit_instruction}）
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    # 提取结果列表
    if isinstance(raw, list):
        results = raw
    elif isinstance(raw, dict):
        # 将字典展开为列表，保留原始键，便于后续 info 标注
        results = []
        for k, v in raw.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault('_id', k)
                results.append(item)
            else:
                results.append({'_id': k, 'value': v})
    else:
        raise RuntimeError("JSON 文件格式不符合预期，请检查数据格式")
    
    # 不再按 VIE 分数筛选；直接取前 TOP_K 条（如不足则全量）
    valid = results
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
        # 复制 original 和 edited 视频（mask 与 rem）
        for role, tag in (("original_video", 'ori'), ("edited_video", 'edit')):
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

        # 额外：推断并复制原始 org_reshape 视频
        org_rel = ''
        mask_rel = item.get('original_video', '').lstrip('./')
        if mask_rel:
            replaced = re.sub(r"_mask_\d+\.mp4$", "_org.mp4", mask_rel)
            if replaced != mask_rel:
                org_rel = replaced
        if not org_rel:
            rem_rel = item.get('edited_video', '').lstrip('./')
            if rem_rel:
                replaced = re.sub(r"_rem_\d+\.mp4$", "_org.mp4", rem_rel)
                if replaced != rem_rel:
                    org_rel = replaced

        if org_rel:
            org_path = Path(BASE_DATASET_DIR) / org_rel
            if org_path.is_file():
                dest_name = f"{idx_str}_org_{org_path.name}"
                dest_path = Path(OUTPUT_DIR) / dest_name
                shutil.copy2(org_path, dest_path)
            else:
                print(f"[警告] 原始视频不存在(推断): {org_path}")

        # 生成 info.txt
        info_path = Path(OUTPUT_DIR) / f"{idx_str}_info.txt"
        with open(info_path, 'w', encoding='utf-8') as info_f:
            info_f.write(f"sample_id: {idx}\n")
            info_f.write(f"json_key: {item.get('_id', '')}\n")
            info_f.write(f"edit_instruction: {item.get('edit_instruction', '')}\n")
            info_f.write(f"original_video: {item.get('original_video', '')}\n")
            info_f.write(f"edited_video: {item.get('edited_video', '')}\n")
            # 记录推断的原始 org_reshape 路径（若存在推断）
            mask_rel = item.get('original_video', '').lstrip('./')
            rem_rel = item.get('edited_video', '').lstrip('./')
            inferred_org = ''
            if mask_rel:
                tmp = re.sub(r"_mask_\d+\.mp4$", "_org_reshape.mp4", mask_rel)
                if tmp != mask_rel:
                    inferred_org = tmp
            if not inferred_org and rem_rel:
                tmp = re.sub(r"_rem_\d+\.mp4$", "_org_reshape.mp4", rem_rel)
                if tmp != rem_rel:
                    inferred_org = tmp
            info_f.write(f"original_org_reshape_inferred: {inferred_org}\n")

    # 生成简要采样汇总
    summary = {
        'total_original': len(results),
        'top_k_used': len(top_items),
        'final_sampled': len(sampled)
    }
    with open(Path(OUTPUT_DIR) / 'sample_summary.json', 'w', encoding='utf-8') as sum_f:
        json.dump(summary, sum_f, ensure_ascii=False, indent=2)

    print("处理完成，视频已拷贝，info.txt 已生成，采样汇总保存在 sample_summary.json")

if __name__ == '__main__':
    main()