"""
Training script for DisTime / TRACE with distributed training support.
Supports multiple VLM backends: model_type="smolvlm" | "fastvlm"
Supports multiple paradigms: paradigm="distime" | "trace"
Supports DeepSpeed, wandb logging, and SLURM.
"""

import os
import sys
import logging
import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Import local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import ModelArguments, DataArguments, TrainingArguments
from utils.dist_utils import init_dist, get_rank, get_world_size, is_main_process, print_rank0
from utils.losses import DisTimeLoss
from utils.metrics import compute_metrics
from data.dataset import LazySupervisedDataset, collate_fn, build_datasets

# =============================================================================
# 按 model_type 选择默认参数
# =============================================================================

# 每种模型的默认 image_size
MODEL_DEFAULT_IMAGE_SIZE = {
    "smolvlm": 384,    # SmolVLM (SigLIP vision encoder)
    "fastvlm": 1024,   # FastVLM (fastvit_mci3, crop_size=1024)
    "molmo2": 378,     # Molmo2 (SigLIP2 So400m/14, 378×378)
}

# 每种模型的默认 image_seq_len (用于 pre-check)
MODEL_DEFAULT_IMAGE_SEQ_LEN = {
    "smolvlm": 81,     # SmolVLM: confirmed from processor
    "fastvlm": 256,    # FastVLM: (1024 // 64)^2 = 256
    "molmo2": 81,      # Molmo2: (378/14/3)^2 = 9^2 = 81 (video pooling_size=[3,3])
}


def load_model_by_type(model_type, paradigm, model_args, distime_config, torch_dtype, device_map, vision_pool_stride=1):
    """按 model_type + paradigm 动态加载对应的模型类。"""
    if paradigm == "distime":
        if model_type == "smolvlm":
            from models.smolvlm_distime import SmolVLMDisTime, DisTimeConfig
            model = SmolVLMDisTime(
                model_name_or_path=model_args.model_name_or_path,
                distime_config=distime_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        elif model_type == "fastvlm":
            from models.fastvlm_distime import FastVLMDisTime, DisTimeConfig
            model = FastVLMDisTime(
                model_name_or_path=model_args.model_name_or_path,
                distime_config=distime_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        elif model_type == "molmo2":
            from models.molmo2_distime import Molmo2DisTime, DisTimeConfig
            model = Molmo2DisTime(
                model_name_or_path=model_args.model_name_or_path,
                distime_config=distime_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    elif paradigm == "trace":
        if model_type == "smolvlm":
            from models.smolvlm_trace import SmolVLMTrace, TraceConfig
            trace_config = TraceConfig()
            model = SmolVLMTrace(
                model_name_or_path=model_args.model_name_or_path,
                trace_config=trace_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        elif model_type == "fastvlm":
            from models.fastvlm_trace import FastVLMTrace, TraceConfig
            trace_config = TraceConfig(vision_pool_stride=vision_pool_stride)
            model = FastVLMTrace(
                model_name_or_path=model_args.model_name_or_path,
                trace_config=trace_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        elif model_type == "molmo2":
            from models.molmo2_trace import Molmo2Trace, TraceConfig
            trace_config = TraceConfig()
            model = Molmo2Trace(
                model_name_or_path=model_args.model_name_or_path,
                trace_config=trace_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            raise ValueError(f"Unknown model_type for paradigm='trace': {model_type}")
    elif paradigm == "text":
        # Text Numeral paradigm: base VLM + LoRA, plain-text timestamps, LM loss only.
        if model_type == "smolvlm":
            from models.smolvlm_text import SmolVLMText
            model = SmolVLMText(
                model_name_or_path=model_args.model_name_or_path,
                distime_config=distime_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        elif model_type == "fastvlm":
            from models.fastvlm_text import FastVLMText
            model = FastVLMText(
                model_name_or_path=model_args.model_name_or_path,
                distime_config=distime_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        elif model_type == "molmo2":
            from models.molmo2_text import Molmo2Text
            model = Molmo2Text(
                model_name_or_path=model_args.model_name_or_path,
                distime_config=distime_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            raise ValueError(f"Unknown model_type for paradigm='text': {model_type}")
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}. Supported: distime, trace, text")

    return model


class DisTimeTrainer(Trainer):
    """
    Custom Trainer for DisTime / TRACE with proper loss handling and logging.
    Supports both SmolVLM and FastVLM backends, both distime and trace paradigms.
    """

    def __init__(self, distime_config, model_type: str = "smolvlm",
                 paradigm: str = "distime", **kwargs):
        super().__init__(**kwargs)
        self.distime_config = distime_config
        self.model_type = model_type
        self.paradigm = paradigm

        # Initialize loss function (DisTime only; TRACE computes loss inside model.forward)
        if paradigm == "distime":
            self.distime_loss = DisTimeLoss(
                reg_max=distime_config.reg_max,
                dfl_weight=distime_config.time_loss_weight,
                iou_weight=distime_config.iou_loss_weight,
            )

    @staticmethod
    def _allrank_dummy_loss(model):
        """
        ZeRO-3 safe dummy loss: touches ALL trainable parameters so that
        backward triggers all-gather on every partition, keeping ranks in sync.
        """
        return sum(p.sum() * 0.0 for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _allrank_skip_flag(local_bad: bool, device):
        """
        All-reduce a skip flag across ranks. If ANY rank is bad, ALL ranks skip.
        """
        flag = torch.tensor([1 if local_bad else 0], device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with pre-flight safety checks.
        按 model_type 适配不同的检查逻辑。
        """
        input_ids = inputs['input_ids']
        labels = inputs.get('labels', None)
        device = input_ids.device

        local_bad = False
        bad_reason = []

        # ------------------------------------------------------------------
        # CHECK 1: vocab bounds
        # ------------------------------------------------------------------
        raw_model = model.module if hasattr(model, 'module') else model
        base = getattr(raw_model, 'base_model', raw_model)
        cfg = getattr(base, 'config', None)
        vocab_size = getattr(cfg, 'vocab_size', None)
        if vocab_size is None and hasattr(cfg, 'text_config'):
            vocab_size = getattr(cfg.text_config, 'vocab_size', None)

        if vocab_size is not None:
            mx = int(input_ids.max().item())
            mn = int(input_ids.min().item())

            # TRACE extended IDs 超过 vocab_size 是合法的 (sync/time/score tokens)
            if self.paradigm == "trace":
                # TRACE: extended range = vocab_size + 1(sync) + 13(time) + 13(score) = vocab_size + 27
                max_valid_id = vocab_size + 27
            else:
                max_valid_id = vocab_size

            if mn < 0 or mx >= max_valid_id:
                local_bad = True
                bad_reason.append(f"input_ids OOB: min={mn} max={mx} max_valid={max_valid_id}")

            if labels is not None:
                lmn = int(labels.min().item())
                lmx = int(labels.max().item())
                # TRACE: labels 中 sync 位置值为 vocab_size (text+sync head 的最后一个 class)
                max_valid_label = vocab_size + 1 if self.paradigm == "trace" else vocab_size
                if (lmn < -100) or (lmx >= max_valid_label) or (lmn < 0 and lmn != -100):
                    local_bad = True
                    bad_reason.append(f"labels OOB: min={lmn} max={lmx} max_valid={max_valid_label}")

        # ------------------------------------------------------------------
        # CHECK 2: image token alignment
        # 按 model_type 使用对应的 IMAGE_SEQ_LEN
        # ------------------------------------------------------------------
        IMAGE_SEQ_LEN = MODEL_DEFAULT_IMAGE_SEQ_LEN.get(self.model_type, 81)
        img_token_id = getattr(cfg, 'image_token_id', None)
        # Molmo2 uses 'image_patch_id' instead of 'image_token_id'
        if img_token_id is None:
            img_token_id = getattr(cfg, 'image_patch_id', None)

        # Molmo2 uses pixel_values_videos (not pixel_values), skip this validation
        has_valid_pixel_values = (
            'pixel_values' in inputs
            and inputs['pixel_values'] is not None
            and self.model_type != "molmo2"
        )
        if img_token_id is not None and has_valid_pixel_values:
            per_img = (input_ids == img_token_id).sum(dim=1)  # [B]

            if 'image_flags' in inputs and inputs['image_flags'] is not None:
                nf = inputs['image_flags'].sum(dim=1)  # [B] real frame count
            else:
                nf = torch.full_like(per_img, fill_value=inputs['pixel_values'].shape[1])

            expected = nf * IMAGE_SEQ_LEN

            if not torch.equal(per_img, expected):
                local_bad = True
                bad_reason.append(
                    f"img_token mismatch: per_sample={per_img.tolist()} "
                    f"expected={expected.tolist()} nf={nf.tolist()}"
                )

            mod = per_img % IMAGE_SEQ_LEN
            if torch.any(mod != 0):
                local_bad = True
                bad_reason.append(f"img_tokens not divisible by {IMAGE_SEQ_LEN}: mod={mod.tolist()}")

        # ------------------------------------------------------------------
        # ALL-RANK SYNC SKIP
        # ------------------------------------------------------------------
        skip = self._allrank_skip_flag(local_bad, device)
        if skip:
            if local_bad:
                logger.warning(
                    f"[PRECHECK SKIP] step={self.state.global_step} "
                    f"reasons={bad_reason}"
                )
            dummy_loss = self._allrank_dummy_loss(model)
            return (dummy_loss, None) if return_outputs else dummy_loss

        # ------------------------------------------------------------------
        # SAFE: forward pass
        # 两个模型/范式的 forward 签名不同:
        #   DisTime: 需要 time_gt, num_events, duration, frame_times
        #   TRACE:   需要 time_labels, score_labels
        # 统一传所有字段, 让各模型自己处理
        # ------------------------------------------------------------------
        forward_kwargs = dict(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs.get('pixel_values'),
            image_flags=inputs.get('image_flags'),
            labels=inputs.get('labels'),
            return_dict=True,
        )

        # Molmo2: 使用 video path (pixel_values_videos) 而非 image path
        if self.model_type == "molmo2":
            # 移除 pixel_values (Molmo2 用 pixel_values_videos 代替)
            forward_kwargs.pop('pixel_values', None)
            for k in ('pixel_values_videos', 'video_token_pooling', 'video_grids'):
                if k in inputs:
                    forward_kwargs[k] = inputs[k]

        if self.paradigm == "trace":
            forward_kwargs['time_labels'] = inputs.get('time_labels')
            forward_kwargs['score_labels'] = inputs.get('score_labels')
        else:
            forward_kwargs['time_gt'] = inputs.get('time_gt')
            forward_kwargs['num_events'] = inputs.get('num_events')
            forward_kwargs['duration'] = inputs.get('duration')
            forward_kwargs['frame_times'] = inputs.get('frame_times')

        outputs = model(**forward_kwargs)

        loss = outputs['loss']

        # ------------------------------------------------------------------
        # [FIX] POST-FORWARD LOSS SANITY CHECK
        # 检测 loss 爆炸 (NaN / Inf / 异常大), 降级为 dummy loss
        # 原因: time_decoder 初期输出不稳定 → DIoU loss 爆炸 → 梯度爆炸
        #       → 干扰 lm_loss 的正常优化, 导致训练发散
        # SmolVLM 只在前 ~13 步爆炸然后稳定, FastVLM 持续爆炸无法收敛
        # ------------------------------------------------------------------
        MAX_SAFE_LOSS = 100.0  # 正常 total_loss ≈ 5-10, 超过 100 必然异常
        loss_val = loss.item()
        loss_bad = (
            torch.isnan(loss) or torch.isinf(loss) or loss_val > MAX_SAFE_LOSS
        )
        loss_skip = self._allrank_skip_flag(loss_bad, device)

        if loss_skip:
            if loss_bad and self.args.local_rank in [-1, 0]:
                logger.warning(
                    f"[LOSS SKIP] step={self.state.global_step} "
                    f"loss={loss_val:.4g} "
                    f"lm={outputs.get('lm_loss', '?')}, "
                    f"dfl={outputs.get('time_loss', '?')}, "
                    f"diou={outputs.get('iou_loss', '?')}"
                )
            dummy_loss = self._allrank_dummy_loss(model)
            return (dummy_loss, None) if return_outputs else dummy_loss

        # Log individual losses (only on main process)
        if (self.state.global_step % self.args.logging_steps == 0 and
                self.args.local_rank in [-1, 0]):
            self._log_losses(outputs)

        return (loss, outputs) if return_outputs else loss

    def _log_losses(self, outputs: Dict[str, Any]):
        """Log individual loss components."""
        logs = {}

        if self.paradigm == "trace":
            # TRACE: text_loss, time_loss, score_loss
            for key, log_name in [('text_loss', 'train/text_loss'),
                                   ('time_loss', 'train/time_loss'),
                                   ('score_loss', 'train/score_loss')]:
                if key in outputs and outputs[key] is not None:
                    val = outputs[key]
                    if isinstance(val, torch.Tensor):
                        logs[log_name] = val.item()
        else:
            # DisTime: lm_loss, time_loss (dfl), iou_loss
            if 'lm_loss' in outputs and outputs['lm_loss'] is not None:
                lm_loss = outputs['lm_loss']
                if isinstance(lm_loss, torch.Tensor):
                    logs['train/lm_loss'] = lm_loss.item()

            if 'time_loss' in outputs and outputs['time_loss'] is not None:
                time_loss = outputs['time_loss']
                if isinstance(time_loss, torch.Tensor):
                    logs['train/dfl_loss'] = time_loss.item()

            if 'iou_loss' in outputs and outputs['iou_loss'] is not None:
                iou_loss = outputs['iou_loss']
                if isinstance(iou_loss, torch.Tensor):
                    logs['train/diou_loss'] = iou_loss.item()

        if logs:
            self.log(logs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model.

        DisTime/TRACE 各自的 save_pretrained() 已经负责保存各自的专有模块。
        base_model 权重由 model.save_pretrained() 或 DeepSpeed 自行处理。
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            model = self.model
            if hasattr(model, 'module'):
                model = model.module
            model.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        logger.info(f"Model saved to {output_dir}")


def setup_wandb(training_args: TrainingArguments):
    """Setup wandb logging."""
    if "wandb" in training_args.report_to and is_main_process():
        try:
            import wandb
            os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "distime")
            wandb.init(
                name=training_args.run_name,
                config=training_args.to_dict() if hasattr(training_args, 'to_dict') else vars(training_args),
                resume="allow",
            )
            logger.info(f"Wandb initialized with run name: {training_args.run_name}")
        except ImportError:
            logger.warning("wandb not installed, disabling wandb logging")
            training_args.report_to = [r for r in training_args.report_to if r != "wandb"]


def get_parameter_groups(model, training_args):
    """Create parameter groups with different learning rates."""
    time_module_params = []
    other_params = []

    # DisTime: time_encoder, time_decoder
    # TRACE: time_tower, score_tower, sync_tower, time_head, score_head, sync_head
    time_module_names = ['time_encoder', 'time_decoder',
                         'time_tower', 'score_tower', 'sync_tower',
                         'time_head', 'score_head', 'sync_head']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(m in name for m in time_module_names):
            time_module_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': training_args.learning_rate,
        })
    if time_module_params:
        time_lr = training_args.time_module_lr or training_args.learning_rate
        param_groups.append({
            'params': time_module_params,
            'lr': time_lr,
            'name': 'time_modules',
        })

    return param_groups


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        import yaml
        with open(sys.argv[1], 'r') as f:
            config = yaml.safe_load(f)
        model_args, data_args, training_args = parser.parse_dict(config)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # <<<< 新增: 获取 model_type 和 paradigm
    model_type = model_args.model_type
    paradigm = model_args.paradigm
    assert model_type in ("smolvlm", "fastvlm", "molmo2"), f"Unknown model_type: {model_type}"
    assert paradigm in ("distime", "trace"), f"Unknown paradigm: {paradigm}"
    logger.info(f"Using model_type: {model_type}, paradigm: {paradigm}")

    # Initialize distributed training
    launcher = os.environ.get('LAUNCHER', 'pytorch')
    if training_args.local_rank != -1 or os.environ.get('WORLD_SIZE', '1') != '1':
        init_dist(launcher=launcher, backend='nccl')

    # Setup logging level
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Process rank: {get_rank()}, world size: {get_world_size()}")
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")

    set_seed(training_args.seed)
    setup_wandb(training_args)

    # Check for checkpoint
    checkpoint = None
    if "checkpoint" in model_args.model_name_or_path:
        checkpoint = model_args.model_name_or_path
        logger.info(f"Resuming from checkpoint: {checkpoint}")

    # Create paradigm-specific config
    distime_config = None
    # vision_pool_stride 仅 FastVLM 使用
    vision_pool_stride = model_args.vision_pool_stride if model_type == "fastvlm" else 1

    if paradigm in ("distime", "text"):
        # 'text' reuses DisTimeConfig only to carry hidden_size / vision_pool_stride;
        # its time-specific fields (reg_max etc.) are ignored by the Text models.
        if model_type == "smolvlm":
            from models.smolvlm_distime import DisTimeConfig
        elif model_type == "molmo2":
            from models.molmo2_distime import DisTimeConfig
        else:
            from models.fastvlm_distime import DisTimeConfig

        distime_kwargs = dict(
            reg_max=model_args.reg_max,
            num_time_layers=model_args.num_time_layers,
            time_loss_weight=model_args.time_loss_weight,
            iou_loss_weight=model_args.iou_loss_weight,
        )
        # vision_pool_stride 仅 FastVLM DisTimeConfig 支持
        if model_type == "fastvlm":
            distime_kwargs["vision_pool_stride"] = vision_pool_stride
        distime_config = DisTimeConfig(**distime_kwargs)

    # <<<< 按 model_type + paradigm 加载模型
    torch_dtype = getattr(torch, model_args.torch_dtype)

    is_distributed = training_args.local_rank != -1 or os.environ.get('WORLD_SIZE', '1') != '1'
    device_map = None if (training_args.deepspeed or is_distributed) else "auto"

    # ================================================================
    # 本地缓存: 把模型从共享文件系统拷贝到节点本地 /tmp,
    # 避免多 rank 同时从共享文件系统读取导致 I/O 过载.
    # 每个节点只由 local_rank==0 执行拷贝, 其他 rank 等待.
    # ================================================================
    import shutil
    original_model_path = model_args.model_name_or_path
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    model_basename = os.path.basename(original_model_path.rstrip('/'))
    local_model_path = f"/tmp/model_cache_{slurm_job_id}/{model_basename}"

    logger.info(f"[Rank {global_rank}, local_rank {local_rank}] "
                f"Model path: {original_model_path}, is_dir: {os.path.isdir(original_model_path)}")

    if is_distributed and os.path.isdir(original_model_path):
        if local_rank == 0:
            if not os.path.exists(local_model_path):
                logger.info(f"[Rank {global_rank}, local_rank 0] "
                            f"Copying model to local cache: {local_model_path}")
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                def _ignore_non_model(directory, files):
                    """只拷贝模型必要文件, 跳过 wandb/logs/checkpoint 等垃圾"""
                    ignore = set()
                    for f in files:
                        if f in ('wandb', 'runs', 'logs', '__pycache__', '.git'):
                            ignore.add(f)
                    return ignore
                shutil.copytree(original_model_path, local_model_path,
                                ignore=_ignore_non_model,
                                ignore_dangling_symlinks=True)
                logger.info(f"[Rank {global_rank}, local_rank 0] Model copy complete")
            else:
                logger.info(f"[Rank {global_rank}, local_rank 0] "
                            f"Local cache already exists: {local_model_path}")

        # 全局同步: 等所有节点的 local_rank 0 拷贝完成
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if os.path.exists(local_model_path):
            logger.info(f"[Rank {global_rank}] Loading model from local cache: {local_model_path}")
            model_args.model_name_or_path = local_model_path
        else:
            logger.warning(f"[Rank {global_rank}] Local cache not found, "
                           f"falling back to: {original_model_path}")
    else:
        logger.info(f"Not distributed or model path is not a directory, loading directly.")

    logger.info(f"Loading model from {model_args.model_name_or_path} "
                f"(model_type={model_type}, paradigm={paradigm})")

    print(f"[DEBUG Rank {global_rank} local_rank {local_rank}] 模型加载开始", flush=True)
    from transformers.integrations import is_deepspeed_zero3_enabled
    print(f"[DEBUG] is_deepspeed_zero3_enabled = {is_deepspeed_zero3_enabled()}", flush=True)
    model = load_model_by_type(
        model_type=model_type,
        paradigm=paradigm,
        model_args=model_args,
        distime_config=distime_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        vision_pool_stride=vision_pool_stride,
    )
    print(f"[DEBUG Rank {global_rank} local_rank {local_rank}] 模型加载完成", flush=True)

    # Setup training (freeze, LoRA, etc.)
    model.setup_training(
        use_lora=model_args.use_lora,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_target_modules=model_args.lora_target_modules,
        freeze_vision=model_args.freeze_vision_tower,
    )
    model.print_training_setup()

    trainable, total = model.get_trainable_parameters()
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if training_args.gradient_checkpointing:
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

    # <<<< 按 model_type 选择 image_size (优先从 model config 自动检测)
    image_size = MODEL_DEFAULT_IMAGE_SIZE[model_type]  # fallback default
    # 优先从 model.config.vision_config.image_size 读取 (最可靠)
    config = getattr(model, 'base_model', model)
    config = getattr(config, 'config', None)
    if config is not None:
        vc = getattr(config, 'vision_config', None)
        if vc is not None and hasattr(vc, 'image_size'):
            detected_size = vc.image_size
            if isinstance(detected_size, int) and detected_size != image_size:
                logger.info(f"Auto-detected image_size={detected_size} from vision_config "
                           f"(default was {image_size})")
                image_size = detected_size
    logger.info(f"Using image_size={image_size} for model_type={model_type}")

    # <<<< 自动检测 image_seq_len (优先从 processor 读取)
    base_image_seq_len = MODEL_DEFAULT_IMAGE_SEQ_LEN[model_type]  # fallback default
    if hasattr(model, 'processor') and hasattr(model.processor, 'image_seq_len'):
        detected_seq_len = model.processor.image_seq_len
        if detected_seq_len != base_image_seq_len:
            logger.info(f"Auto-detected image_seq_len={detected_seq_len} from processor "
                       f"(default was {base_image_seq_len})")
            base_image_seq_len = detected_seq_len
            MODEL_DEFAULT_IMAGE_SEQ_LEN[model_type] = base_image_seq_len
    if vision_pool_stride > 1:
        image_seq_len_override = base_image_seq_len // (vision_pool_stride * vision_pool_stride)
        logger.info(f"Vision pooling: stride={vision_pool_stride}, "
                   f"image_seq_len {base_image_seq_len} → {image_seq_len_override}")
    else:
        image_seq_len_override = None

    # 更新 MODEL_DEFAULT_IMAGE_SEQ_LEN 用于 trainer 的 pre-check
    if image_seq_len_override is not None:
        MODEL_DEFAULT_IMAGE_SEQ_LEN[model_type] = image_seq_len_override

    # <<<< TRACE: 获取模型的 vocab_size, 传给 dataset 以对齐 extended token ID
    # Qwen2 (FastVLM): len(tokenizer)=151647 ≠ config.vocab_size=151936
    # 如果不传, dataset 会用 len(tokenizer), 导致 TRACE token ID 与模型不匹配
    trace_vocab_size = getattr(model, 'vocab_size', None) if paradigm == "trace" else None

    # <<<< Load dataset, 传入 model_type
    logger.info(f"Loading dataset from {data_args.data_path}")

    if not data_args.multi_dataset:
        train_dataset = LazySupervisedDataset(
            data_path=data_args.data_path,
            processor=model.processor,
            model_type=model_type,
            paradigm=paradigm,               # <<<< 新增
            video_folder=data_args.video_folder,
            max_frames=data_args.max_frames,
            fps=data_args.fps,
            max_length=data_args.model_max_length,
            image_size=image_size,
            is_train=True,
            image_seq_len_override=image_seq_len_override,
            vocab_size_override=trace_vocab_size,
        )
    else:
        import json
        with open(data_args.data_path, 'r') as f:
            data_configs = json.load(f)

        train_dataset = build_datasets(
            data_configs=data_configs,
            processor=model.processor,
            model_type=model_type,
            paradigm=paradigm,               # <<<< 新增
            video_folder=data_args.video_folder,
            max_frames=data_args.max_frames,
            image_size=image_size,
            use_data_resampling=data_args.use_data_resampling,
            use_packed_ds=data_args.use_packed_ds,
            max_packed_tokens=data_args.max_packed_tokens,
            image_seq_len_override=image_seq_len_override,
            vocab_size_override=trace_vocab_size,
        )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            processor=model.processor,
            model_type=model_type,
            paradigm=paradigm,               # <<<< 新增
            video_folder=data_args.video_folder,
            max_frames=data_args.max_frames,
            fps=data_args.fps,
            max_length=data_args.model_max_length,
            image_size=image_size,
            is_train=False,
            image_seq_len_override=image_seq_len_override,
            vocab_size_override=trace_vocab_size,
        )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # ================================================================
    # 同步屏障: 确保所有 rank 完成模型加载 + 数据集初始化
    # 不同 rank 从共享文件系统加载模型的速度差异很大 (32 it/s vs 289 it/s),
    # 如果快的 rank 先进入 DeepSpeed.initialize 的 NCCL collective,
    # 慢的 rank 还没到, NCCL 连接会超时 → 训练卡住.
    # 这个 barrier 确保所有 rank 都准备好了再一起进入 DeepSpeed.
    # ================================================================
    if dist.is_available() and dist.is_initialized():
        logger.info(f"[Rank {get_rank()}] Waiting for all ranks to finish model/data loading...")
        dist.barrier()
        logger.info(f"[Rank {get_rank()}] All ranks synchronized, proceeding to trainer")

    # Custom optimizer for different LR
    optimizer = None
    if training_args.time_module_lr is not None and training_args.time_module_lr != training_args.learning_rate:
        param_groups = get_parameter_groups(model, training_args)
        optimizer = torch.optim.AdamW(param_groups, lr=training_args.learning_rate)
        logger.info(f"Using custom optimizer with time_module_lr={training_args.time_module_lr}")

    # <<<< Create trainer, 传入 model_type + paradigm
    trainer = DisTimeTrainer(
        distime_config=distime_config,
        model_type=model_type,
        paradigm=paradigm,                   # <<<< 新增
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics if eval_dataset else None,
        optimizers=(optimizer, None) if optimizer else (None, None),
    )

    # Training
    if training_args.do_train:
        logger.info("*** Starting training ***")

        # ================================================================
        # FIX: TRACE 多节点 DeepSpeed 初始化卡住修复
        # 问题: DeepSpeed broadcast 999 个参数后, ZeRO-2 optimizer 初始化
        #       调用 param.data.cpu() 触发 cuda.synchronize(), 如果 NCCL
        #       broadcast kernel 还在 GPU stream 上排队未完成, .cpu() 会死等.
        # 修复: 在 ZeRO optimizer init 之前强制 cuda.synchronize() + barrier,
        #       确保所有 rank 的 GPU 都空闲后再开始参数拷贝.
        # ================================================================
        import deepspeed
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as _ZeroOpt
        _orig_zero_init = _ZeroOpt.__init__

        def _patched_zero_init(self_zero, *za, **zkw):
            rank = get_rank()
            logger.info(f"[Rank {rank}] cuda.synchronize() before ZeRO param copy")
            torch.cuda.synchronize()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            logger.info(f"[Rank {rank}] All ranks synced, starting ZeRO optimizer init")
            return _orig_zero_init(self_zero, *za, **zkw)

        _ZeroOpt.__init__ = _patched_zero_init
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        finally:
            _ZeroOpt.__init__ = _orig_zero_init

        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("*** Training finished ***")

    # Evaluation
    if training_args.do_eval and eval_dataset:
        logger.info("*** Running evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Cleanup
    if "wandb" in training_args.report_to and is_main_process():
        try:
            import wandb
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()
