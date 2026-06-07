#!/usr/bin/env python3
"""
分析 combined_distime_balanced.jsonl:
1. 各数据集来源的样本数和比例
2. 各任务类型 (QA/MR/Multi) 的数量和比例
3. 视频文件实际存在率（按数据集分组）
"""

import json
import os
import sys
from collections import defaultdict

DATA_PATH = "/work/hdd/bffz/yzou1/data/combined_distime_balanced.jsonl"
VIDEO_FOLDER = "/work/hdd/bffz/yzou1/data/"

# ============================================================
# 路径解析 (和 dataset.py 的 _resolve_video_path 一致)
# ============================================================
def resolve_video_path(video_path):
    full = os.path.join(VIDEO_FOLDER, video_path)
    if os.path.exists(full):
        return full

    for part in video_path.split('/'):
        if part.startswith('split_video_'):
            alt = os.path.join('/work/hdd/bffz/yzou1/data/videos_new2', part, os.path.basename(video_path))
            return alt

    if 'coin/videos' in video_path:
        parts = full.split('/')
        filename = parts[-1]
        if filename.startswith('video-'):
            filename = filename[6:]
        return '/'.join(parts[:-2]) + '/' + filename + '.mp4'

    elif 'queryd/QuerYD_downloader' in video_path:
        parts = full.split('/')
        filename = parts[-1]
        if filename.startswith('video-'):
            filename = filename[6:]
        return '/'.join(parts[:-2]) + '/' + filename + '.mp4'

    elif 'yttemporal/videos' in video_path:
        parts = full.split('/')
        filename = parts[-1]
        if filename.startswith('video-'):
            filename = filename[6:]
        return '/'.join(parts[:-1]) + '/' + filename + '.mp4'

    elif 'didemo/videos' in video_path:
        parts = full.split('/')
        filename = parts[-1]
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        return '/'.join(parts[:-1]) + '/train/' + filename + '.mp4'

    # 尝试加扩展名
    for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
        if os.path.exists(full + ext):
            return full + ext

    return full


# ============================================================
# 判断数据集来源
# ============================================================
def get_dataset_source(video_path):
    if not video_path:
        return "unknown"
    v = video_path.lower()
    if 'split_video_' in v:
        return "internvid"
    elif 'valley' in v:
        return "valley"
    elif 'yttemporal' in v:
        return "yttemporal"
    elif 'sharegpt4video' in v:
        return "sharegpt4video"
    elif 'activitynet' in v or 'anet' in v:
        return "activitynet"
    elif 'didemo' in v:
        return "didemo"
    elif 'coin' in v:
        return "coin"
    elif 'vitt' in v:
        return "vitt"
    elif 'queryd' in v:
        return "queryd"
    elif 'textvr' in v:
        return "textvr"
    elif 'videochat' in v:
        return "videochat"
    else:
        return "other"


# ============================================================
# 判断任务类型
# ============================================================
def get_task_type(item):
    times = item.get('times', [])
    n = len(times)
    if n == 0:
        return "QA"
    elif n == 1:
        return "MR"
    else:
        return f"Multi({n})"


def get_task_category(item):
    """粗分类: QA / MR / Multi"""
    times = item.get('times', [])
    n = len(times)
    if n == 0:
        return "QA"
    elif n == 1:
        return "MR"
    else:
        return "Multi"


# ============================================================
# 主流程
# ============================================================
def main():
    print(f"Loading {DATA_PATH} ...")
    
    # 统计变量
    total = 0
    source_counts = defaultdict(int)
    task_counts = defaultdict(int)
    task_category_counts = defaultdict(int)
    
    # 按数据集统计视频存在性 (去重, 每个唯一视频只检查一次)
    source_videos = defaultdict(set)       # source -> set of video paths
    source_video_exists = defaultdict(int)  # source -> count of existing videos
    checked_videos = {}                     # video_path -> exists (bool)

    with open(DATA_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1

            video = item.get('video', '')
            source = get_dataset_source(video)
            source_counts[source] += 1

            task = get_task_type(item)
            task_counts[task] += 1
            task_category_counts[get_task_category(item)] += 1

            # 记录唯一视频
            source_videos[source].add(video)

            if total % 200000 == 0:
                print(f"  processed {total:,} ...")

    print(f"\nTotal samples: {total:,}\n")

    # ============================================================
    # 1. 数据集来源分布
    # ============================================================
    print("=" * 70)
    print("1. 数据集来源分布")
    print("=" * 70)
    print(f"{'Source':<20} {'Samples':>10} {'Pct':>8} {'Unique Videos':>15}")
    print("-" * 70)
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total
        n_videos = len(source_videos[source])
        print(f"{source:<20} {count:>10,} {pct:>7.1f}% {n_videos:>15,}")

    # ============================================================
    # 2. 任务类型分布
    # ============================================================
    print(f"\n{'=' * 70}")
    print("2. 任务类型分布 (粗分类)")
    print("=" * 70)
    print(f"{'Type':<10} {'Count':>10} {'Pct':>8}")
    print("-" * 30)
    for cat in ["QA", "MR", "Multi"]:
        c = task_category_counts.get(cat, 0)
        print(f"{cat:<10} {c:>10,} {100.0*c/total:>7.1f}%")

    print(f"\n--- 细分 (Multi 事件数) ---")
    print(f"{'Type':<15} {'Count':>10} {'Pct':>8}")
    print("-" * 35)
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"{task:<15} {count:>10,} {100.0*count/total:>7.1f}%")

    # ============================================================
    # 3. 视频存在性检查 (按数据集抽样)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("3. 视频存在性检查")
    print("=" * 70)

    total_videos = 0
    total_exists = 0

    print(f"{'Source':<20} {'Total Videos':>14} {'Exists':>10} {'Missing':>10} {'Exist%':>8}")
    print("-" * 70)

    for source in sorted(source_videos.keys(), key=lambda s: -len(source_videos[s])):
        videos = source_videos[source]
        n_total = len(videos)
        n_exists = 0

        for v in videos:
            if v in checked_videos:
                if checked_videos[v]:
                    n_exists += 1
            else:
                resolved = resolve_video_path(v)
                exists = os.path.exists(resolved)
                checked_videos[v] = exists
                if exists:
                    n_exists += 1

        n_missing = n_total - n_exists
        pct = 100.0 * n_exists / n_total if n_total > 0 else 0
        total_videos += n_total
        total_exists += n_exists

        print(f"{source:<20} {n_total:>14,} {n_exists:>10,} {n_missing:>10,} {pct:>7.1f}%")

    print("-" * 70)
    total_missing = total_videos - total_exists
    total_pct = 100.0 * total_exists / total_videos if total_videos > 0 else 0
    print(f"{'TOTAL':<20} {total_videos:>14,} {total_exists:>10,} {total_missing:>10,} {total_pct:>7.1f}%")

    # 缺失样本数估算
    missing_samples = 0
    for source, videos in source_videos.items():
        for v in videos:
            if not checked_videos.get(v, False):
                # 这个视频缺失，统计它影响多少样本 (近似: 用 samples/videos 比例)
                pass
    
    # 精确统计缺失样本数
    print(f"\n--- 受影响样本数 ---")
    missing_video_set = {v for v, exists in checked_videos.items() if not exists}
    affected_samples = 0
    with open(DATA_PATH, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('video', '') in missing_video_set:
                affected_samples += 1
    print(f"缺失视频影响的样本数: {affected_samples:,} / {total:,} ({100.0*affected_samples/total:.1f}%)")
    print(f"可用样本数: {total - affected_samples:,} ({100.0*(total-affected_samples)/total:.1f}%)")


if __name__ == "__main__":
    main()
