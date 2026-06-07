"""
TRACE 数据筛选脚本（保留 TRACE 格式）

从 stage-1-v2-shorten.json 筛选数据：
  1. 过滤 image 数据（有 "image" 字段且无 "video" 字段）
  2. 过滤无 video / 无对话的无效样本
  3. 保留原始 TRACE 格式不做任何截断

输入: stage-1-v2-shorten.json (TRACE 原始格式)
输出: stage-1-v2-shorten_trace.jsonl (保留 TRACE 格式)
"""

import json
import os
import sys
from collections import Counter


def clean_times(times_raw):
    """清洗 times: [[]] → [], [[30.0], [31.0]] → [[30.0], [31.0]]"""
    if not times_raw:
        return []
    return [t for t in times_raw if isinstance(t, list) and len(t) > 0]


def convert_sample(item):
    """处理单个样本，返回 (result, status)，result=None 表示跳过"""
    # 过滤 image
    if 'image' in item and 'video' not in item:
        return None, 'filtered_image'
    if 'video' not in item:
        return None, 'filtered_no_video'

    conversations = item.get('conversations', [])
    if len(conversations) < 2:
        return None, 'filtered_invalid'

    human_msg = conversations[0].get('value', '')
    gpt_msg = conversations[1].get('value', '')

    if not gpt_msg.strip():
        return None, 'filtered_empty'

    times_raw = item.get('times', [])
    scores_raw = item.get('scores', [])
    valid_times = clean_times(times_raw)

    # 构建输出（保持 TRACE 原始格式，不截断）
    result = {
        'video': item['video'],
        'conversations': [
            {'from': 'human', 'value': human_msg},
            {'from': 'gpt', 'value': gpt_msg},
        ],
        'times': valid_times if valid_times else [[]],
        'scores': scores_raw if scores_raw else [[]],
    }

    if 'id' in item:
        result['id'] = item['id']
    if 'task' in item:
        result['task'] = item['task']

    return result, 'converted'


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/work/hdd/bffz/yzou1/data/TRACE/trace/stage-1-v2-shorten.json"

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_trace.jsonl"

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")

    print("读取数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始样本数: {len(data)}")

    stats = Counter()
    converted = []

    for item in data:
        result, status = convert_sample(item)
        stats[status] += 1
        if result is not None:
            converted.append(result)

    # 保存
    print(f"\n写入 {len(converted)} 样本...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 统计
    print(f"\n{'='*50}")
    print(f"筛选统计:")
    print(f"  原始总数:       {len(data)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # 任务类型统计
    task_stats = Counter()
    for s in converted:
        n = len(clean_times(s.get('times', [])))
        if n == 0:
            task_stats['QA (0事件)'] += 1
        elif n == 1:
            task_stats['单事件 (1事件)'] += 1
        else:
            task_stats[f'多事件 ({n}事件)'] += 1
    print(f"\n任务分布:")
    for k, v in sorted(task_stats.items()):
        print(f"  {k}: {v}")
    print(f"{'='*50}")
    print(f"输出: {output_path}")


if __name__ == '__main__':
    main()
