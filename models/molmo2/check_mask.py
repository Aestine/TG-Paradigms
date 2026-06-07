# 验证 Molmo2 DisTime label masking 是否正确 (video格式)
# 在服务器上运行: python check_mask.py /path/to/Molmo2-4B

from transformers import AutoTokenizer
import torch
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/projects/bffz/yzou1/models/Molmo2-4B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 模拟 _add_special_tokens: 添加 DisTime 特殊 token
TIME_STAMP_TOKEN = "<TIME_STAMP>"
FRAME_TIME_TOKEN = "<FRAME_TIME>"
num_added = tokenizer.add_special_tokens({
    'additional_special_tokens': [TIME_STAMP_TOKEN, FRAME_TIME_TOKEN]
})
print(f"Added {num_added} special tokens to tokenizer")
print(f"  {TIME_STAMP_TOKEN} -> {tokenizer.convert_tokens_to_ids(TIME_STAMP_TOKEN)}")
print(f"  {FRAME_TIME_TOKEN} -> {tokenizer.convert_tokens_to_ids(FRAME_TIME_TOKEN)}")

# 模拟真实 video 格式的 frame prompt (3帧, 每帧81个<im_patch>)
num_frames = 3
patch_tokens = "<im_patch>" * 81  # 9x9 grid
frame_strings = []
for i in range(num_frames):
    frame_strings.append(f"{FRAME_TIME_TOKEN}: <frame_start>{patch_tokens}<frame_end>")
frame_prompt = "\n".join(frame_strings)

# 模拟真实的 DisTime 数据
user_text = f"{frame_prompt}\nWhen does the person start walking?"
assistant_text = f"{TIME_STAMP_TOKEN} The person starts walking at the beginning of the video."

conversation = (
    f"<|im_start|>user\n{user_text}<|im_end|>\n"
    f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
)

# tokenize
encoded = tokenizer(conversation, return_tensors="pt", add_special_tokens=True)
input_ids = encoded.input_ids.squeeze(0)

# 模拟 _create_labels (molmo2 branch)
IGNORE_INDEX = -100
labels = input_ids.clone()
assistant_marker = "<|im_start|>assistant\n"
assistant_marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)

input_ids_list = input_ids.tolist()
assistant_start_idx = None
for i in range(len(input_ids_list) - len(assistant_marker_ids) + 1):
    if input_ids_list[i:i + len(assistant_marker_ids)] == assistant_marker_ids:
        assistant_start_idx = i + len(assistant_marker_ids)
        break

if assistant_start_idx is not None:
    labels[:assistant_start_idx] = IGNORE_INDEX
else:
    print("WARNING: assistant marker NOT found!")
    labels[:len(labels) // 2] = IGNORE_INDEX

# 打印结果 (只打印关键位置，省略重复的 <im_patch>)
print("\n" + "="*80)
print(f"assistant_marker_ids: {assistant_marker_ids}")
print(f"assistant_start_idx: {assistant_start_idx}")
print(f"total tokens: {len(input_ids)}")
print()

# 统计各类 token
im_patch_id = tokenizer.convert_tokens_to_ids("<im_patch>")
frame_start_id = tokenizer.convert_tokens_to_ids("<frame_start>")
frame_end_id = tokenizer.convert_tokens_to_ids("<frame_end>")
frame_time_id = tokenizer.convert_tokens_to_ids(FRAME_TIME_TOKEN)
time_stamp_id = tokenizer.convert_tokens_to_ids(TIME_STAMP_TOKEN)

print(f"Token IDs:")
print(f"  <im_patch>     = {im_patch_id}")
print(f"  <frame_start>  = {frame_start_id}")
print(f"  <frame_end>    = {frame_end_id}")
print(f"  <FRAME_TIME>   = {frame_time_id}")
print(f"  <TIME_STAMP>   = {time_stamp_id}")
print()

# 打印非 <im_patch> tokens (省略大量重复)
print(f"{'idx':>4} {'token_id':>8} {'label':>8} {'masked':>7}  token_text")
print("-"*70)
patch_count = 0
for i, (tid, lid) in enumerate(zip(input_ids.tolist(), labels.tolist())):
    if tid == im_patch_id:
        patch_count += 1
        continue
    else:
        if patch_count > 0:
            masked_str = "MASKED" if labels[i-1].item() == IGNORE_INDEX else "LEARN"
            print(f"     ... {patch_count}x <im_patch> (all {masked_str}) ...")
            patch_count = 0

    token_text = tokenizer.decode([tid])
    masked = "MASKED" if lid == IGNORE_INDEX else "LEARN"
    marker = ""
    if i == assistant_start_idx:
        marker = " <-- LEARN starts here"
    print(f"{i:4d} {tid:8d} {lid:8d} {masked:>7}  {repr(token_text)}{marker}")

if patch_count > 0:
    masked_str = "MASKED" if labels[len(input_ids)-1].item() == IGNORE_INDEX else "LEARN"
    print(f"     ... {patch_count}x <im_patch> (all {masked_str}) ...")

# 统计
num_masked = (labels == IGNORE_INDEX).sum().item()
num_learn = (labels != IGNORE_INDEX).sum().item()
print(f"\nMasked (user/system): {num_masked} tokens")
print(f"Learning (assistant): {num_learn} tokens")
print(f"Mask ratio: {num_masked}/{len(labels)} = {num_masked/len(labels):.1%}")

# 关键检查
print("\n" + "="*80)
print("KEY CHECKS:")

# 1. assistant marker 是否找到
if assistant_start_idx is None:
    print("[FAIL] Could not find assistant marker!")
else:
    print(f"[OK] Assistant marker found at index {assistant_start_idx}")

    # 2. user prompt 完全 masked
    user_labels = labels[:assistant_start_idx]
    if (user_labels == IGNORE_INDEX).all():
        print("[OK] User/system prompt is fully masked")
    else:
        num_leaked = (user_labels != IGNORE_INDEX).sum().item()
        print(f"[FAIL] User/system prompt has {num_leaked} UNMASKED tokens!")

    # 3. assistant response 完全 learn
    assistant_labels = labels[assistant_start_idx:]
    if (assistant_labels == IGNORE_INDEX).all():
        print("[FAIL] Assistant response is ALL masked - model learns NOTHING!")
    elif (assistant_labels != IGNORE_INDEX).all():
        print("[OK] Assistant response is fully visible for learning")
    else:
        print("[WARN] Assistant response is partially masked")

# 4. 检查 <TIME_STAMP> 是否为单个 token
if time_stamp_id == tokenizer.unk_token_id or time_stamp_id is None:
    print(f"[FAIL] <TIME_STAMP> not in tokenizer vocab (got id={time_stamp_id})")
else:
    print(f"[OK] <TIME_STAMP> is a single token (id={time_stamp_id})")

# 5. 检查 <FRAME_TIME> 是否为单个 token
if frame_time_id == tokenizer.unk_token_id or frame_time_id is None:
    print(f"[FAIL] <FRAME_TIME> not in tokenizer vocab (got id={frame_time_id})")
else:
    print(f"[OK] <FRAME_TIME> is a single token (id={frame_time_id})")

# 6. 检查 <frame_start>/<frame_end> 是否为单个 token
if frame_start_id == tokenizer.unk_token_id:
    print(f"[FAIL] <frame_start> not in tokenizer!")
else:
    print(f"[OK] <frame_start> is a single token (id={frame_start_id})")

if frame_end_id == tokenizer.unk_token_id:
    print(f"[FAIL] <frame_end> not in tokenizer!")
else:
    print(f"[OK] <frame_end> is a single token (id={frame_end_id})")

# 7. 检查所有视频帧 tokens 都在 masked 区域
frame_token_ids = {im_patch_id, frame_start_id, frame_end_id, frame_time_id}
frame_tokens_in_learn = 0
for i in range(assistant_start_idx if assistant_start_idx else 0, len(input_ids)):
    if input_ids[i].item() in frame_token_ids and labels[i].item() != IGNORE_INDEX:
        frame_tokens_in_learn += 1
if frame_tokens_in_learn == 0:
    print("[OK] No video frame tokens leaked into learning region")
else:
    print(f"[WARN] {frame_tokens_in_learn} video frame tokens in learning region")
