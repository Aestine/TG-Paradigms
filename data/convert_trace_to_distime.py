"""
TRACE → SmolVLM-DisTime 格式转换器

TRACE 特殊 token:
  <sync>  = 时间段锚点 → 对应 DisTime 的 <TIME_STAMP>
  <time>  = 时间分布编码 token (多个) → 移除 (DisTime 用 TimeEncoder/Decoder)
  <score> = 显著性评分 token (多个) → 移除 (DisTime 不支持 saliency)

转换规则:
  1. 过滤掉 image 数据 (有 "image" 字段且无 "video" 字段)
  2. <sync> → <TIME_STAMP> (仅当对应 times 条目非空时保留)
  3. <time>, <score> → 移除
  4. times: [[]] → [] (无事件, General QA)
  5. times: [[t]] → [[t]] (时间点)
  6. times: [[s, e]] → [[s, e]] (时间区间)
  7. 对齐: <TIME_STAMP> 数量 == 有效 times 数量
  8. 保留 <video> 占位符
  9. Highlight Detection: 有 scores 且事件数 > MAX_EVENTS 时,
     保留 top-K 高分时间点, 按时间顺序排列
"""

import json
import re
import os
import sys
from collections import Counter


# ============================================================
# 配置
# ============================================================
MAX_EVENTS = 8       # highlight detection 最多保留的事件数
MIN_SCORE = 3.0      # 低于此分数的时间点直接丢弃


def clean_times(times_raw):
    """
    清洗 times 字段:
    - [[]] → []  (空内层列表 = 无事件)
    - [[30.0], [31.0]] → [[30.0], [31.0]]  (时间点)
    - [[10.0, 15.0]] → [[10.0, 15.0]]  (区间)
    """
    if not times_raw:
        return []

    valid = []
    for t in times_raw:
        if isinstance(t, list) and len(t) > 0:
            valid.append(t)
    return valid


def clean_scores(scores_raw):
    """
    清洗 scores 字段:
    - [[4.4], [3.4]] → [4.4, 3.4]
    - [[]] → []
    """
    if not scores_raw:
        return []

    flat = []
    for s in scores_raw:
        if isinstance(s, list) and len(s) > 0:
            flat.append(float(s[0]))
        else:
            flat.append(0.0)  # 无效 score 当 0
    return flat


def filter_by_score_and_topk(valid_times, scores, min_score=MIN_SCORE, k=MAX_EVENTS):
    """
    1. 过滤 score < min_score 的时间点
    2. 剩余的取 top-K (按 score 降序)
    3. 按时间顺序排列

    Returns:
        selected_times: 筛选后的 times, 按时间顺序
        selected_indices: 原始 indices (时间顺序), 用于对齐 gpt 文本段
    """
    if len(scores) != len(valid_times):
        # score 和 times 对不上, 退回均匀采样
        step = max(1, len(valid_times) // k)
        indices = sorted(list(range(0, len(valid_times), step))[:k])
        return [valid_times[i] for i in indices], indices

    # Step 1: 过滤低分
    candidates = [(i, scores[i]) for i in range(len(scores)) if scores[i] >= min_score]

    if len(candidates) == 0:
        # 全部低于阈值 → 保底取 top-3, 仍保留多事件特征
        MIN_KEEP = 3
        all_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(all_sorted[:min(MIN_KEEP, len(scores))])
        return [valid_times[i] for i in top_indices], top_indices

    # Step 2: 按 score 降序取 top-K
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in candidates[:k]]

    # Step 3: 按时间顺序排
    top_indices.sort()

    return [valid_times[i] for i in top_indices], top_indices


def convert_gpt_text(text, valid_times):
    """
    转换 gpt response 中的 TRACE token → DisTime token

    策略:
    1. 移除所有 <time> 和 <score>
    2. <sync> → <TIME_STAMP>
    3. 对齐 TIME_STAMP 数量和 valid_times 数量
    """
    # Step 1: 移除 <time> 和 <score>
    text = text.replace('<time>', '')
    text = text.replace('<score>', '')

    # Step 2: <sync> → <TIME_STAMP>
    text = text.replace('<sync>', '<TIME_STAMP>')

    # Step 3: 清理多余空白
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    # Step 4: 对齐
    n_stamps = text.count('<TIME_STAMP>')
    n_valid = len(valid_times)

    if n_valid == 0:
        # 无事件 → 移除所有 TIME_STAMP (纯 QA)
        text = text.replace('<TIME_STAMP>', '')
        text = re.sub(r' +', ' ', text).strip()
    elif n_stamps > n_valid:
        # TIME_STAMP 多于 times (TRACE 常见: 开头有 double <sync>)
        # 从头部移除多余的
        excess = n_stamps - n_valid
        for _ in range(excess):
            text = text.replace('<TIME_STAMP>', '', 1)
        text = re.sub(r' +', ' ', text).strip()
    elif n_stamps < n_valid:
        # TIME_STAMP 少于 times → 截断 times
        valid_times = valid_times[:n_stamps]

    return text, valid_times


def select_segments_from_text(converted_gpt, selected_indices):
    """
    从已转换的 gpt text 中, 只保留 selected_indices 对应的 TIME_STAMP 段.

    例:
        text = "<TIME_STAMP>A<TIME_STAMP>B<TIME_STAMP>C"
        selected_indices = [0, 2]
        → "<TIME_STAMP>A<TIME_STAMP>C"
    """
    # 按 <TIME_STAMP> 切分
    parts = converted_gpt.split('<TIME_STAMP>')
    # parts[0] = 前缀 (通常为空), parts[1:] = 各段文本

    segments = parts[1:]  # 每段对应一个事件
    prefix = parts[0].strip()

    selected_segments = [segments[i] for i in selected_indices if i < len(segments)]

    result = '<TIME_STAMP>'.join([''] + selected_segments)
    if prefix:
        result = prefix + result

    # 清理
    result = re.sub(r' +', ' ', result).strip()
    return result


def convert_sample(item):
    """转换单个样本, 返回 None 表示跳过"""

    # 过滤 image 数据
    if 'image' in item and 'video' not in item:
        return None

    # 过滤无 video 字段的数据
    if 'video' not in item:
        return None

    # 检查 conversations 格式
    conversations = item.get('conversations', [])
    if len(conversations) < 2:
        return None

    human_msg = conversations[0].get('value', '')
    gpt_msg = conversations[1].get('value', '')

    # 清洗 times 和 scores
    times_raw = item.get('times', [])
    scores_raw = item.get('scores', [])
    valid_times = clean_times(times_raw)
    valid_scores = clean_scores(scores_raw)

    # 转换 gpt text (sync→TIME_STAMP, 移除 time/score, 对齐)
    converted_gpt, aligned_times = convert_gpt_text(gpt_msg, valid_times)

    # ========================================================
    # Highlight Detection 过滤:
    # 有有效 scores 时, 过滤低分 + top-K 截断
    # ========================================================
    n_events = len(aligned_times)
    if n_events > 1 and len(valid_scores) >= n_events:
        aligned_scores = valid_scores[:n_events]

        # 检查是否需要过滤 (有低分 或 超过 MAX_EVENTS)
        has_low_scores = any(s < MIN_SCORE for s in aligned_scores)
        needs_topk = n_events > MAX_EVENTS

        if has_low_scores or needs_topk:
            selected_times, selected_indices = filter_by_score_and_topk(
                aligned_times, aligned_scores
            )

            # 从 gpt text 中提取对应段落
            converted_gpt = select_segments_from_text(converted_gpt, selected_indices)
            aligned_times = selected_times

    # 跳过空 gpt 回答
    if not converted_gpt.strip():
        return None

    # 构建输出
    result = {
        'video': item['video'],
        'conversations': [
            {'from': 'human', 'value': human_msg},
            {'from': 'gpt', 'value': converted_gpt},
        ],
        'times': aligned_times,
    }

    # 保留 id
    if 'id' in item:
        result['id'] = item['id']

    # 保留 task 字段
    if 'task' in item:
        result['task'] = item['task']

    return result


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/work/hdd/bffz/yzou1/data/TRACE/trace/stage-1-v2-shorten.json"

    # 输出: 同目录, 加 _distime 后缀, JSONL 格式
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_distime.jsonl"

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"MAX_EVENTS: {MAX_EVENTS}, MIN_SCORE: {MIN_SCORE}")

    # 读取
    print("读取数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始样本数: {len(data)}")

    # 统计
    stats = Counter()
    highlight_before_after = []  # 记录 highlight 截断前后的事件数
    converted = []

    for item in data:
        # 分类统计
        if 'image' in item and 'video' not in item:
            stats['filtered_image'] += 1
            continue

        # 记录截断前事件数 (用于统计)
        orig_n = len(clean_times(item.get('times', [])))

        result = convert_sample(item)
        if result is None:
            stats['filtered_invalid'] += 1
            continue

        n_events = len(result['times'])

        # 统计 highlight 过滤
        final_n = len(result['times'])
        if orig_n > 1 and final_n < orig_n:
            stats['highlight_filtered'] += 1
            highlight_before_after.append((orig_n, final_n))

        if n_events == 0:
            stats['qa_0_events'] += 1
        elif n_events == 1:
            stats['mr_1_event'] += 1
        else:
            stats['multi_n_events'] += 1

        # 验证对齐
        n_stamps = result['conversations'][1]['value'].count('<TIME_STAMP>')
        if n_stamps != n_events:
            stats['alignment_error'] += 1
            print(f"  ⚠ 对齐错误 id={result.get('id','?')}: "
                  f"{n_stamps} stamps vs {n_events} times")
            continue

        converted.append(result)
        stats['converted'] += 1

    # 保存 (JSONL 格式，方便合并)
    print(f"\n写入 {len(converted)} 样本...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 打印统计
    print(f"\n{'='*50}")
    print(f"转换统计:")
    print(f"  原始总数:           {len(data)}")
    print(f"  过滤 (image):       {stats['filtered_image']}")
    print(f"  过滤 (invalid):     {stats['filtered_invalid']}")
    print(f"  对齐错误:           {stats['alignment_error']}")
    print(f"  成功转换:           {stats['converted']}")
    print(f"    - QA (0事件):       {stats['qa_0_events']}")
    print(f"    - MR (1事件):       {stats['mr_1_event']}")
    print(f"    - Multi (N事件):    {stats['multi_n_events']}")
    print(f"    - Highlight 过滤:   {stats['highlight_filtered']}")
    if highlight_before_after:
        avg_before = sum(b for b, _ in highlight_before_after) / len(highlight_before_after)
        avg_after = sum(a for _, a in highlight_before_after) / len(highlight_before_after)
        print(f"      avg 截断: {avg_before:.1f} → {avg_after:.1f}")
    print(f"{'='*50}")
    print(f"输出: {output_path}")

    # 打印样例
    print(f"\n--- 样例 ---")
    examples = {'QA': None, 'MR': None, 'Multi': None}
    for s in converted:
        n = len(s['times'])
        if n == 0 and examples['QA'] is None:
            examples['QA'] = s
        elif n == 1 and examples['MR'] is None:
            examples['MR'] = s
        elif n > 1 and examples['Multi'] is None:
            examples['Multi'] = s
        if all(v is not None for v in examples.values()):
            break

    for label, s in examples.items():
        if s:
            n_ev = len(s['times'])
            gpt_short = s['conversations'][1]['value'][:150]
            print(f"\n[{label}] video={s['video']}")
            print(f"    events={n_ev}, times={s['times'][:4]}{'...' if n_ev > 4 else ''}")
            print(f"    gpt: {gpt_short}{'...' if len(gpt_short) >= 150 else ''}")


if __name__ == '__main__':
    main()