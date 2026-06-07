"""
InternVid DisTime → TRACE 格式转换器

DisTime InternVid 格式:
  {"video": "...", "tgt": [s, e], "conversations": [
    {"from": "human", "value": "<video>\n...caption..."},
    {"from": "gpt", "value": "The event is featured at <TIME_STAMP>."}
  ]}

TRACE 格式:
  {"video": "...", "conversations": [
    {"from": "human", "value": "<video>\n...query..."},
    {"from": "gpt", "value": "<sync><time>×14<score>A large group of tourists is gathered..."}
  ], "times": [[s, e]], "scores": [[]]}

转换规则:
  1. tgt: [s, e] → times: [[s, e]]
  2. scores: [[]] (InternVid 无 saliency 评分)
  3. gpt 回答 = TRACE 前缀 + human 引号内的 query 描述
     （原 gpt 模板 "The event is featured at <TIME_STAMP>." 整个丢弃）
  4. TRACE 前缀: <sync><time>×N<score>×M
     - 区间 [s, e]: N=14, M=1
"""

import json
import os
import re
import sys
from collections import Counter


def compute_time_token_count(time_pair):
    """
    计算时间编码的 token 数量

    TimeTower.encode 格式:
      单值 [t]: format(t, '0>6.1f') + '<sync>' = 6+1 = 7 tokens
      区间 [s, e]: format(s, '0>6.1f') + '<sep>' + format(e, '0>6.1f') + '<sync>' = 6+1+6+1 = 14 tokens
    """
    if len(time_pair) == 1:
        return 7  # 6 chars + 1 sync
    elif len(time_pair) == 2:
        return 14  # 6 + 1(sep) + 6 + 1(sync)
    else:
        # N values: N*6 + (N-1) seps + 1 sync
        n = len(time_pair)
        return n * 6 + (n - 1) + 1


def compute_score_token_count(score_list):
    """
    计算 score 编码的 token 数量

    ScoreTower.encode 格式:
      空 []: '<sync>' = 1 token
      单值 [v]: format(v, '0>3.1f') + '<sync>' = 3+1 = 4 tokens
      多值 [v1, v2]: format(v1, '0>3.1f') + '<sep>' + format(v2, '0>3.1f') + '<sync>'
    """
    if not score_list:
        return 1  # just <sync>
    n = len(score_list)
    return n * 3 + (n - 1) + 1  # N*3 chars + (N-1) seps + 1 sync


def convert_sample(item):
    """转换单个 InternVid DisTime 样本到 TRACE 格式"""
    conversations = item.get('conversations', [])
    if len(conversations) < 2:
        return None

    human_msg = conversations[0].get('value', '')
    gpt_msg = conversations[1].get('value', '')

    # 获取时间区间
    tgt = item.get('tgt', None)
    if tgt is None:
        # 尝试从 times 字段获取
        times = item.get('times', [])
        if times and len(times) > 0 and len(times[0]) > 0:
            tgt = times[0]
        else:
            return None

    # 确保 tgt 是 [start, end] 格式
    if isinstance(tgt, (int, float)):
        tgt = [float(tgt), float(tgt)]
    elif len(tgt) == 1:
        tgt = [float(tgt[0]), float(tgt[0])]
    else:
        tgt = [float(tgt[0]), float(tgt[1])]

    # 计算 TRACE token 数量
    time_token_count = compute_time_token_count(tgt)
    score_token_count = compute_score_token_count([])  # InternVid 无 score

    # 构建 TRACE 事件前缀
    trace_prefix = "<sync>" + "<time>" * time_token_count + "<score>" * score_token_count

    # 从 human 引号中提取 query 描述作为 caption
    # 匹配 '...' 或 "..." 中的内容
    quote_match = re.search(r"['\u2018\u2019]([^']+)['\u2018\u2019]|\"([^\"]+)\"", human_msg)
    if quote_match:
        caption = quote_match.group(1) or quote_match.group(2)
        caption = caption.strip()
    else:
        # 没找到引号，fallback: 移除 <video>\n 和问句部分，取中间描述
        caption = human_msg.replace('<video>', '').strip()
        # 去掉换行后的第一行作为 caption
        lines = caption.split('\n')
        caption = lines[0].strip() if lines else gpt_msg.replace('<TIME_STAMP>', '').strip()

    gpt_trace = trace_prefix + caption

    result = {
        'video': item['video'],
        'conversations': [
            {'from': 'human', 'value': human_msg},
            {'from': 'gpt', 'value': gpt_trace},
        ],
        'times': [tgt],
        'scores': [[]],
    }

    if 'id' in item:
        result['id'] = item['id']

    return result


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/work/hdd/bffz/yzou1/data/internvid_100k_v1_overlap_608k_v1_distime.jsonl"

    # 输出: 替换 _distime 为 _trace
    if '_distime' in input_path:
        output_path = input_path.replace('_distime', '_trace')
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_trace{ext}"

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")

    stats = Counter()
    converted = []

    # 读取 JSONL
    print("读取数据...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats['parse_error'] += 1
                continue

            result = convert_sample(item)
            if result is None:
                stats['filtered'] += 1
            else:
                converted.append(result)
                stats['converted'] += 1

            if (line_num + 1) % 100000 == 0:
                print(f"  processed {line_num + 1} lines...")

    # 保存
    print(f"\n写入 {len(converted)} 样本...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 统计
    print(f"\n{'='*50}")
    print(f"转换统计:")
    total = sum(stats.values())
    print(f"  总读取:     {total}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"{'='*50}")
    print(f"输出: {output_path}")

    # 样例
    if converted:
        print(f"\n--- 样例 ---")
        s = converted[0]
        print(f"video: {s['video']}")
        print(f"times: {s['times']}")
        print(f"scores: {s['scores']}")
        gpt = s['conversations'][1]['value']
        print(f"gpt: {gpt[:200]}{'...' if len(gpt) > 200 else ''}")


if __name__ == '__main__':
    main()
