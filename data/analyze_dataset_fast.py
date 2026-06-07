#!/usr/bin/env python3
"""
快速版: 复用 dataset.py 的 _resolve_video_path 逻辑,
按目录聚合做 listdir 而不是逐个 stat.
"""

import json
import os
from collections import defaultdict

DATA_PATH = "/work/hdd/bffz/yzou1/data/combined_distime_balanced.jsonl"
VIDEO_FOLDER = "/work/hdd/bffz/yzou1/data/"

# ============================================================
# 和 dataset.py 完全一致的路径解析
# ============================================================
def resolve_video_path(video_path):
    full = os.path.join(VIDEO_FOLDER, video_path)

    for part in video_path.split('/'):
        if part.startswith('split_video_'):
            basename = os.path.basename(video_path)
            full = os.path.join('/work/hdd/bffz/yzou1/data/videos_new2', part, basename)
            break

    if 'coin/videos' in full:
        parts = full.split('/')
        filename = parts[-1]
        if filename.startswith('video-'):
            filename = filename[6:]
        full = '/'.join(parts[:-2]) + '/' + filename + '.mp4'

    elif 'queryd/QuerYD_downloader' in full:
        parts = full.split('/')
        filename = parts[-1]
        if filename.startswith('video-'):
            filename = filename[6:]
        full = '/'.join(parts[:-2]) + '/' + filename + '.mp4'

    elif 'yttemporal/videos' in full:
        parts = full.split('/')
        filename = parts[-1]
        if filename.startswith('video-'):
            filename = filename[6:]
        full = '/'.join(parts[:-1]) + '/' + filename + '.mp4'

    elif 'didemo/videos' in full:
        parts = full.split('/')
        filename = parts[-1]
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        full = '/'.join(parts[:-1]) + '/train/' + filename + '.mp4'

    elif 'vitt/videos' in full:
        parts = full.split('/')
        filename = parts[-1]
        if filename.startswith('video-'):
            filename = filename[6:]
        full = '/'.join(parts[:-1]) + '/' + filename + '.mp4'

    return full


def get_source(video):
    if not video:
        return "unknown"
    v = video.lower()
    if 'split_video_' in v: return "internvid"
    elif 'valley' in v: return "valley"
    elif 'yttemporal' in v: return "yttemporal"
    elif 'sharegpt4video' in v: return "sharegpt4video"
    elif 'activitynet' in v or 'anet' in v: return "activitynet"
    elif 'didemo' in v: return "didemo"
    elif 'coin' in v: return "coin"
    elif 'vitt' in v: return "vitt"
    elif 'queryd' in v: return "queryd"
    elif 'textvr' in v: return "textvr"
    elif 'videochat' in v: return "videochat"
    else: return "other"


# ============================================================
# Pass 1: 扫描 jsonl, 收集统计 + 唯一视频的 resolved 路径
# ============================================================
print(f"Pass 1: Scanning {DATA_PATH} ...")

total = 0
source_counts = defaultdict(int)
task_category_counts = defaultdict(int)
task_detail_counts = defaultdict(int)
source_videos = defaultdict(set)

# video_path -> resolved_path
resolved_cache = {}

with open(DATA_PATH, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        total += 1

        video = item.get('video', '')
        source = get_source(video)
        source_counts[source] += 1
        source_videos[source].add(video)

        if video not in resolved_cache:
            resolved_cache[video] = resolve_video_path(video)

        times = item.get('times', [])
        n = len(times)
        if n == 0:
            task_category_counts["QA"] += 1
            task_detail_counts["QA"] += 1
        elif n == 1:
            task_category_counts["MR"] += 1
            task_detail_counts["MR"] += 1
        else:
            task_category_counts["Multi"] += 1
            task_detail_counts[f"Multi({n})"] += 1

        if total % 200000 == 0:
            print(f"  processed {total:,} ...")

print(f"  Total: {total:,}, Unique videos: {len(resolved_cache):,}")

# ============================================================
# Pass 2: 按目录聚合, 每个目录只做一次 listdir
# ============================================================
print(f"\nPass 2: Checking video existence by directory...")

dir_to_files = defaultdict(set)
video_to_dir = {}

for video, resolved in resolved_cache.items():
    d = os.path.dirname(resolved)
    b = os.path.basename(resolved)
    dir_to_files[d].add(b)
    video_to_dir[video] = (d, b)

print(f"  {len(dir_to_files)} unique directories to check")

dir_contents = {}
for i, d in enumerate(dir_to_files):
    try:
        dir_contents[d] = set(os.listdir(d))
    except (FileNotFoundError, PermissionError):
        dir_contents[d] = set()
    if (i + 1) % 20 == 0:
        print(f"  listed {i+1}/{len(dir_to_files)} directories...")

print(f"  Done listing all directories")

video_exists = {}
for video, (d, b) in video_to_dir.items():
    video_exists[video] = b in dir_contents.get(d, set())

# ============================================================
# 输出
# ============================================================
print(f"\n{'=' * 75}")
print(f"Total samples: {total:,}")
print(f"{'=' * 75}")

print(f"\n1. 数据集来源分布")
print(f"{'=' * 75}")
print(f"{'Source':<20} {'Samples':>10} {'Pct':>8} {'Unique Videos':>15}")
print("-" * 75)
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    pct = 100.0 * count / total
    n_videos = len(source_videos[source])
    print(f"{source:<20} {count:>10,} {pct:>7.1f}% {n_videos:>15,}")

print(f"\n2. 任务类型分布")
print(f"{'=' * 75}")
print(f"{'Type':<10} {'Count':>10} {'Pct':>8}")
print("-" * 30)
for cat in ["QA", "MR", "Multi"]:
    c = task_category_counts.get(cat, 0)
    print(f"{cat:<10} {c:>10,} {100.0*c/total:>7.1f}%")

print(f"\n--- Multi 细分 ---")
print(f"{'Type':<15} {'Count':>10} {'Pct':>8}")
print("-" * 35)
for task, count in sorted(task_detail_counts.items(), key=lambda x: -x[1]):
    if task.startswith("Multi"):
        print(f"{task:<15} {count:>10,} {100.0*count/total:>7.1f}%")

print(f"\n3. 视频存在性")
print(f"{'=' * 75}")
print(f"{'Source':<20} {'Videos':>10} {'Exists':>10} {'Missing':>10} {'Exist%':>8}")
print("-" * 75)

total_v = 0
total_e = 0
for source in sorted(source_videos.keys(), key=lambda s: -len(source_videos[s])):
    videos = source_videos[source]
    n = len(videos)
    e = sum(1 for v in videos if video_exists.get(v, False))
    m = n - e
    pct = 100.0 * e / n if n > 0 else 0
    total_v += n
    total_e += e
    print(f"{source:<20} {n:>10,} {e:>10,} {m:>10,} {pct:>7.1f}%")

print("-" * 75)
print(f"{'TOTAL':<20} {total_v:>10,} {total_e:>10,} {total_v-total_e:>10,} {100.0*total_e/total_v:>7.1f}%")

# 受影响样本数
affected = 0
with open(DATA_PATH, 'r') as f:
    for line in f:
        item = json.loads(line.strip())
        if not video_exists.get(item.get('video', ''), False):
            affected += 1

print(f"\n--- 受影响样本 ---")
print(f"缺失视频影响样本: {affected:,} / {total:,} ({100.0*affected/total:.1f}%)")
print(f"可用样本: {total - affected:,} ({100.0*(total-affected)/total:.1f}%)")

# 缺失示例
print(f"\n--- 缺失示例 (每数据集最多3个) ---")
for source in sorted(source_videos.keys()):
    missing = [v for v in source_videos[source] if not video_exists.get(v, False)]
    if missing:
        print(f"\n{source} (缺失 {len(missing)}):")
        for v in missing[:3]:
            print(f"  原始: {v}")
            print(f"  解析: {resolved_cache[v]}")