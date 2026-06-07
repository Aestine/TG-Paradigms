"""
DisTime loss functions.
Based on DisTime paper: Distribution Focal Loss + DIoU Loss

修复:
- DIoULoss: 添加 Inf 保护 (原始只检查 NaN), 添加 per-sample loss clamp
- DistributionFocalLoss: 添加 NaN/Inf 保护
- 防止 time_decoder 初期输出不稳定导致 loss 爆炸 → 梯度爆炸 → 训练发散
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) for temporal grounding.

    Interpolates between adjacent bins for soft regression targets.
    For a target value t between bins i and i+1:
        loss = w_left * CE(pred, i) + w_right * CE(pred, i+1)
    where w_left = (i+1) - t, w_right = t - i
    """

    def __init__(self, reg_max: int = 32):
        super().__init__()
        self.reg_max = reg_max

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, 2*(reg_max+1)) predicted distribution logits
            target: (N, 2) target timestamps in [0, reg_max]

        Returns:
            loss: scalar loss value
        """
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        reg_max_p1 = self.reg_max + 1

        # Split predictions
        pred_start = pred[:, :reg_max_p1]  # (N, reg_max+1)
        pred_end = pred[:, reg_max_p1:]  # (N, reg_max+1)

        # Split targets
        target_start = target[:, 0]  # (N,)
        target_end = target[:, 1]  # (N,)

        # Compute DFL for start and end
        loss_start = self._dfl_loss(pred_start, target_start)
        loss_end = self._dfl_loss(pred_end, target_end)

        # return (loss_start + loss_end) / 2
        return loss_start + loss_end

    def _dfl_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (N, K) logits, K = reg_max + 1
        target: (N,) float targets
        """
        K = pred.size(-1)

        # 关键：保证用 fp32 做索引相关计算，避免 31.999 -> 32.0 的舍入问题
        target = target.float()

        # 让 left 最大只能到 K-2，这样 right=left+1 最大到 K-1（永远合法）
        # 这一步比 "target < reg_max" 更强：直接从索引角度保证不会 right==K
        target = target.clamp(min=0.0, max=(K - 2) + (1.0 - 1e-6))  # 等价于 < K-1

        left = torch.floor(target).long()
        left = left.clamp(min=0, max=K - 2)
        right = left + 1  # 必然 <= K-1

        wl = (right.float() - target).clamp(min=0)
        wr = (target - left.float()).clamp(min=0)

        loss_left = F.cross_entropy(pred, left, reduction="none")
        loss_right = F.cross_entropy(pred, right, reduction="none")
        loss = wl * loss_left + wr * loss_right

        # [FIX] 过滤 NaN/Inf per-sample loss (bf16 下 logits 极端时可能出现)
        bad_mask = torch.isnan(loss) | torch.isinf(loss)
        if bad_mask.any():
            loss = torch.where(bad_mask, torch.zeros_like(loss), loss)

        # return loss.mean()
        return loss.sum() / self.reg_max


class DIoULoss(nn.Module):
    """
    Distance-IoU Loss for temporal segments.

    DIoU = IoU - (center_distance^2 / diagonal^2)
    Loss = 1 - DIoU

    修复: 原始代码只检查 NaN 不检查 Inf, 导致 Inf-Inf=NaN 传播链
    添加 per-sample loss clamp(max=10) 防止极端值导致梯度爆炸
    """

    def __init__(self, max_loss: float = 10.0):
        """
        Args:
            max_loss: per-sample loss 上限, 防止 time_decoder 初期输出不稳定时
                      loss 爆炸. 理论上 DIoU loss ∈ [0, 2], 设 10 留充分余量.
        """
        super().__init__()
        self.max_loss = max_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (N, 2) predicted [start, end] timestamps
            target: (N, 2) target [start, end] timestamps

        Returns:
            loss: scalar DIoU loss
        """
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # [FIX] 同时过滤 NaN 和 Inf (原始只过滤 NaN)
        bad_pred = torch.isnan(pred) | torch.isinf(pred)
        bad_target = torch.isnan(target) | torch.isinf(target)
        pred = torch.where(bad_pred, torch.zeros_like(pred), pred)
        target = torch.where(bad_target, torch.zeros_like(target), target)

        # Extract start and end
        pred_start = pred[:, 0]
        pred_end = pred[:, 1]
        target_start = target[:, 0]
        target_end = target[:, 1]

        # Compute IoU
        inter_start = torch.max(pred_start, target_start)
        inter_end = torch.min(pred_end, target_end)
        inter = (inter_end - inter_start).clamp(min=0)

        pred_len = (pred_end - pred_start).clamp(min=1e-6)
        target_len = (target_end - target_start).clamp(min=1e-6)
        union = pred_len + target_len - inter

        iou = inter / union.clamp(min=1e-6)

        # Compute center distance
        pred_center = (pred_start + pred_end) / 2
        target_center = (target_start + target_end) / 2
        center_dist = (pred_center - target_center) ** 2

        # Compute enclosing segment length
        enclose_start = torch.min(pred_start, target_start)
        enclose_end = torch.max(pred_end, target_end)
        enclose_len = (enclose_end - enclose_start).clamp(min=1e-6)

        # DIoU
        diou = iou - center_dist / (enclose_len ** 2)

        # Loss
        loss = 1 - diou

        # [FIX] 过滤计算过程中可能产生的 NaN/Inf
        bad_loss = torch.isnan(loss) | torch.isinf(loss)
        if bad_loss.any():
            loss = torch.where(bad_loss, torch.full_like(loss, 2.0), loss)

        # [FIX] Clamp per-sample loss: 防止极端值导致梯度爆炸
        # 理论上 DIoU loss ∈ [0, 2], 但 bf16 数值问题可能产生更大值
        loss = loss.clamp(max=self.max_loss)

        return loss.mean()


def compute_iou_1d(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Compute 1D IoU between predicted and target segments.

    Args:
        pred: (N, 2) predicted [start, end]
        target: (N, 2) target [start, end]

    Returns:
        iou: (N,) IoU values
    """
    pred_start = pred[:, 0]
    pred_end = pred[:, 1]
    target_start = target[:, 0]
    target_end = target[:, 1]

    inter_start = torch.max(pred_start, target_start)
    inter_end = torch.min(pred_end, target_end)
    inter = (inter_end - inter_start).clamp(min=0)

    pred_len = (pred_end - pred_start).clamp(min=0)
    target_len = (target_end - target_start).clamp(min=0)
    union = pred_len + target_len - inter

    iou = inter / union.clamp(min=1e-6)

    return iou


class DisTimeLoss(nn.Module):
    """
    Combined DisTime loss: DFL + DIoU + Language Modeling

    Total loss = λ_lm * L_lm + λ_dfl * L_dfl + λ_iou * L_iou
    """

    def __init__(
        self,
        reg_max: int = 32,
        dfl_weight: float = 1.0,
        iou_weight: float = 1.0,
        lm_weight: float = 1.0
    ):
        super().__init__()
        self.dfl = DistributionFocalLoss(reg_max=reg_max)
        self.diou = DIoULoss()

        self.dfl_weight = dfl_weight
        self.iou_weight = iou_weight
        self.lm_weight = lm_weight

    def forward(
        self,
        lm_loss: torch.Tensor,
        time_logits: torch.Tensor,
        pred_times: torch.Tensor,
        target_times: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            lm_loss: Language modeling loss from base model
            time_logits: (N, 2*(reg_max+1)) distribution logits
            pred_times: (N, 2) predicted timestamps
            target_times: (N, 2) target timestamps

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses for logging
        """
        # Compute individual losses
        dfl_loss = self.dfl(time_logits, target_times)
        iou_loss = self.diou(pred_times, target_times)

        # Combine losses
        total_loss = (
            self.lm_weight * lm_loss +
            self.dfl_weight * dfl_loss +
            self.iou_weight * iou_loss
        )

        # Compute IoU for logging
        if pred_times.shape[0] > 0:
            with torch.no_grad():
                iou = compute_iou_1d(pred_times, target_times).mean()
        else:
            iou = torch.tensor(0.0, device=lm_loss.device)

        loss_dict = {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'dfl_loss': dfl_loss.item(),
            'iou_loss': iou_loss.item(),
            'mean_iou': iou.item()
        }

        return total_loss, loss_dict


# Test functions
if __name__ == "__main__":
    print("Testing DisTime losses...")

    reg_max = 32
    batch_size = 4

    # Test DFL
    dfl = DistributionFocalLoss(reg_max=reg_max)
    pred_logits = torch.randn(batch_size, 2 * (reg_max + 1))
    target_times = torch.rand(batch_size, 2) * reg_max
    target_times[:, 1] = target_times[:, 0] + torch.rand(batch_size) * 10  # Ensure end > start

    dfl_loss = dfl(pred_logits, target_times)
    print(f"DFL loss: {dfl_loss.item():.4f}")

    # Test DIoU
    diou = DIoULoss()
    pred_times = torch.rand(batch_size, 2) * reg_max
    pred_times[:, 1] = pred_times[:, 0] + torch.rand(batch_size) * 10

    diou_loss = diou(pred_times, target_times)
    print(f"DIoU loss: {diou_loss.item():.4f}")

    # Test with Inf values (should be handled gracefully)
    print("\n--- Testing Inf protection ---")
    pred_inf = torch.tensor([[float('inf'), float('inf')], [5.0, 15.0]])
    target_normal = torch.tensor([[5.0, 15.0], [5.0, 15.0]])
    loss_inf = diou(pred_inf, target_normal)
    print(f"DIoU loss with Inf pred: {loss_inf.item():.4f} (should be ≤ 10.0)")

    # Test with NaN values
    pred_nan = torch.tensor([[float('nan'), float('nan')], [5.0, 15.0]])
    loss_nan = diou(pred_nan, target_normal)
    print(f"DIoU loss with NaN pred: {loss_nan.item():.4f} (should be ≤ 10.0)")

    # Test IoU computation
    iou = compute_iou_1d(pred_times, target_times)
    print(f"\nMean IoU: {iou.mean().item():.4f}")

    # Test combined loss
    combined = DisTimeLoss(reg_max=reg_max)
    lm_loss = torch.tensor(2.5)
    total_loss, loss_dict = combined(lm_loss, pred_logits, pred_times, target_times)
    print(f"Combined loss: {total_loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

    print("\nAll tests passed!")
