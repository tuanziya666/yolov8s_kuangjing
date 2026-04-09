# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_ciou_components, bbox_iou, probiou
from .tal import bbox2dist


def _env_str(name: str, default: str) -> str:
    """Read a lowercase string hyperparameter from the environment."""
    value = os.getenv(name, default)
    return default if value is None else str(value).strip().lower()


def _env_float(name: str, default: float) -> float:
    """Read a float hyperparameter from the environment."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean hyperparameter from the environment."""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int_list(name: str, default: list[int]) -> list[int]:
    """Read a comma-separated integer list from the environment."""
    value = os.getenv(name)
    if value is None:
        return list(default)
    out = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out or list(default)


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss for classification targets carrying localization quality."""

    def __init__(self, beta: float = 2.0):
        """Initialize QFL with the standard modulation exponent."""
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Apply BCE with a quality-aware modulation factor."""
        with autocast(enabled=False):
            pred = pred.float()
            label = label.float()
            pred_prob = pred.sigmoid()
            scale_factor = (pred_prob - label).abs().pow(self.beta)
            loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none") * scale_factor
        return loss.mean(1).sum()


class AdaptiveThresholdFocalLoss(nn.Module):
    """Adaptive Threshold Focal Loss (ATFL) from EFLNet."""

    def __init__(self, loss_fcn: nn.Module, eps: float = 1e-7):
        """Wrap a BCE-style loss and apply adaptive threshold modulation elementwise."""
        super().__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"
        self.eps = eps

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute ATFL with different modulation on the two sides of the 0.5 threshold."""
        with autocast(enabled=False):
            pred = pred.float()
            true = true.float()
            loss = self.loss_fcn(pred, true)
            pred_prob = pred.sigmoid()
            p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
            mean_pt = p_t.mean().clamp_(self.eps, 1.0)
            gamma_high = -mean_pt.log()
            gamma_low = -p_t.clamp_min(self.eps).log()

            modulating_factor = torch.zeros_like(p_t)
            high_mask = p_t > 0.5
            low_mask = ~high_mask
            modulating_factor[high_mask] = (1.000001 - p_t[high_mask]).pow(gamma_high)
            modulating_factor[low_mask] = (1.5 - p_t[low_mask]).pow(gamma_low[low_mask])
            loss = loss * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16, hyp: Any | None = None):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_type = _env_str("ULTRALYTICS_IOU_LOSS", "ciou")
        self.inner_iou_ratio = _env_float("ULTRALYTICS_INNER_IOU_RATIO", 0.8)
        self.wciou_ac_lambda = _env_float("ULTRALYTICS_WCIOU_AC_LAMBDA", 0.7)
        self.wciou_ac_gamma = _env_float("ULTRALYTICS_WCIOU_AC_GAMMA", 2.0)
        self.conf_eps = 1e-6
        self.sa_box_enable = bool(getattr(hyp, "sa_box_enable", _env_bool("ULTRALYTICS_SA_BOX_ENABLE", False)))
        self.sa_small_alpha = float(getattr(hyp, "sa_small_alpha", _env_float("ULTRALYTICS_SA_SMALL_ALPHA", 0.5)))
        self.sa_elong_beta = float(getattr(hyp, "sa_elong_beta", _env_float("ULTRALYTICS_SA_ELONG_BETA", 0.7)))
        self.sa_class_gamma = float(getattr(hyp, "sa_class_gamma", _env_float("ULTRALYTICS_SA_CLASS_GAMMA", 0.5)))
        self.sa_small_area_thr = float(
            getattr(hyp, "sa_small_area_thr", _env_float("ULTRALYTICS_SA_SMALL_AREA_THR", 0.012521))
        )
        self.sa_elong_ratio_thr = float(
            getattr(hyp, "sa_elong_ratio_thr", _env_float("ULTRALYTICS_SA_ELONG_RATIO_THR", 3.0))
        )
        self.sa_target_class_ids = self._normalize_target_class_ids(
            getattr(hyp, "sa_target_class_ids", _env_int_list("ULTRALYTICS_SA_TARGET_CLASS_IDS", [2]))
        )
        self.tal_reg_enable = bool(getattr(hyp, "tal_reg_enable", _env_bool("ULTRALYTICS_TAL_REG_ENABLE", False)))
        self.tal_reg_start_epoch = int(getattr(hyp, "tal_reg_start_epoch", _env_float("ULTRALYTICS_TAL_REG_START_EPOCH", 150)))
        self.tal_reg_ema_decay = float(getattr(hyp, "tal_reg_ema_decay", _env_float("ULTRALYTICS_TAL_REG_EMA_DECAY", 0.95)))
        self.tal_reg_gain = float(getattr(hyp, "tal_reg_gain", _env_float("ULTRALYTICS_TAL_REG_GAIN", 1.0)))
        self.tal_reg_threshold = float(
            getattr(hyp, "tal_reg_threshold", _env_float("ULTRALYTICS_TAL_REG_THRESHOLD", 0.25))
        )
        self.current_epoch = 0
        self.tal_conf_ema = None
        self._reset_sample_aware_stats()

    @staticmethod
    def _normalize_target_class_ids(value: Any) -> list[int]:
        """Normalize target class IDs to a list of integers."""
        if value is None:
            return [2]
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            out = []
            for token in value.replace(";", ",").strip("[]() ").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    out.append(int(token))
                except ValueError:
                    continue
            return out or [2]
        if isinstance(value, (list, tuple, set)):
            out = []
            for item in value:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
            return out or [2]
        return [2]

    def _reset_sample_aware_stats(self) -> None:
        """Reset running sample-aware weighting statistics."""
        self.sa_num_positive_total = 0
        self.sa_num_small_positive = 0
        self.sa_num_elong_positive = 0
        self.sa_num_target_cls_positive = 0
        self.sa_weight_sum = 0.0
        self.sa_weight_max = 0.0
        self.tal_num_weighted_positive = 0
        self.tal_weight_sum = 0.0
        self.tal_weight_max = 0.0

    def _accumulate_sample_aware_stats(
        self, sample_weight: torch.Tensor, is_small: torch.Tensor, is_elong: torch.Tensor, is_target_cls: torch.Tensor
    ) -> None:
        """Accumulate sample-aware weighting statistics for the current epoch."""
        self.sa_num_positive_total += int(sample_weight.numel())
        self.sa_num_small_positive += int(is_small.sum().item())
        self.sa_num_elong_positive += int(is_elong.sum().item())
        self.sa_num_target_cls_positive += int(is_target_cls.sum().item())
        self.sa_weight_sum += float(sample_weight.sum().item())
        self.sa_weight_max = max(self.sa_weight_max, float(sample_weight.max().item()))

    def _accumulate_tal_stats(self, tal_weight: torch.Tensor) -> None:
        """Accumulate late-stage TAL reweighting statistics."""
        self.tal_num_weighted_positive += int(tal_weight.numel())
        self.tal_weight_sum += float(tal_weight.sum().item())
        self.tal_weight_max = max(self.tal_weight_max, float(tal_weight.max().item()))

    def get_sample_aware_metrics(self, reset: bool = True) -> dict[str, float]:
        """Return epoch-level sample-aware weighting metrics."""
        if not self.sa_box_enable and not self.tal_reg_enable:
            return {}

        metrics = {}
        if self.sa_box_enable:
            total = max(self.sa_num_positive_total, 1)
            metrics.update(
                {
                    "sa/num_positive_total": float(self.sa_num_positive_total),
                    "sa/num_small_positive": float(self.sa_num_small_positive),
                    "sa/num_elong_positive": float(self.sa_num_elong_positive),
                    "sa/num_target_cls_positive": float(self.sa_num_target_cls_positive),
                    "sa/num_drill_pipe_positive": float(self.sa_num_target_cls_positive),
                    "sa/mean_sample_weight": float(self.sa_weight_sum / total),
                    "sa/max_sample_weight": float(self.sa_weight_max),
                }
            )
        if self.tal_reg_enable:
            tal_total = max(self.tal_num_weighted_positive, 1)
            metrics.update(
                {
                    "tal/ema_conf": float(self.tal_conf_ema or 0.0),
                    "tal/num_weighted_positive": float(self.tal_num_weighted_positive),
                    "tal/mean_reg_weight": float(self.tal_weight_sum / tal_total),
                    "tal/max_reg_weight": float(self.tal_weight_max),
                }
            )
        if reset:
            self._reset_sample_aware_stats()
        return metrics

    def _build_sample_aware_weight(
        self, target_bboxes_normalized: torch.Tensor, target_scores: torch.Tensor, fg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Build per-positive sample weights for small, elongated, and target-class boxes."""
        if not self.sa_box_enable or target_bboxes_normalized is None:
            return torch.ones((int(fg_mask.sum().item()), 1), device=target_scores.device, dtype=target_scores.dtype)

        target_boxes_fg = target_bboxes_normalized[fg_mask]
        target_scores_fg = target_scores[fg_mask]
        widths = (target_boxes_fg[:, 2] - target_boxes_fg[:, 0]).clamp_min(self.conf_eps)
        heights = (target_boxes_fg[:, 3] - target_boxes_fg[:, 1]).clamp_min(self.conf_eps)
        areas = widths * heights
        ratios = torch.maximum(widths / heights, heights / widths)
        class_ids = target_scores_fg.argmax(-1)

        is_small = areas < self.sa_small_area_thr
        is_elong = ratios > self.sa_elong_ratio_thr
        is_target_cls = torch.zeros_like(class_ids, dtype=torch.bool)
        for class_id in self.sa_target_class_ids:
            is_target_cls |= class_ids == class_id

        sample_weight = (
            1.0
            + self.sa_small_alpha * is_small.float()
            + self.sa_elong_beta * is_elong.float()
            + self.sa_class_gamma * is_target_cls.float()
        )
        self._accumulate_sample_aware_stats(sample_weight, is_small, is_elong, is_target_cls)
        return sample_weight.unsqueeze(-1).to(dtype=target_scores.dtype)

    def _build_tal_reg_weight(self, pred_scores: torch.Tensor | None, target_scores: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        """Late-stage task-adaptive regression weighting based on EMA-smoothed classification confidence."""
        if (
            not self.tal_reg_enable
            or pred_scores is None
            or self.current_epoch < self.tal_reg_start_epoch
            or int(fg_mask.sum().item()) == 0
        ):
            return torch.ones((int(fg_mask.sum().item()), 1), device=target_scores.device, dtype=target_scores.dtype)

        matched_target_scores = target_scores[fg_mask]
        matched_pred_scores = pred_scores[fg_mask].sigmoid()
        matched_conf = (
            (matched_pred_scores * matched_target_scores).sum(-1, keepdim=True)
            / matched_target_scores.sum(-1, keepdim=True).clamp_min(self.conf_eps)
        ).detach()

        batch_mean_conf = float(matched_conf.mean().item())
        if self.tal_conf_ema is None:
            self.tal_conf_ema = batch_mean_conf
        else:
            self.tal_conf_ema = self.tal_reg_ema_decay * self.tal_conf_ema + (1.0 - self.tal_reg_ema_decay) * batch_mean_conf

        tal_weight = torch.ones_like(matched_conf)
        valid_mask = matched_conf >= self.tal_reg_threshold
        tal_weight[valid_mask] = 1.0 + self.tal_reg_gain * (matched_conf[valid_mask] - float(self.tal_conf_ema)).clamp_min(0.0)
        self._accumulate_tal_stats(tal_weight)
        return tal_weight.to(dtype=target_scores.dtype)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        pred_scores: torch.Tensor | None = None,
        target_bboxes_normalized: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        sample_weight = self._build_sample_aware_weight(target_bboxes_normalized, target_scores, fg_mask)
        tal_reg_weight = self._build_tal_reg_weight(pred_scores, target_scores, fg_mask)
        reg_weight = sample_weight * tal_reg_weight
        if self.iou_type == "inner_iou":
            iou = bbox_iou(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                xywh=False,
                InnerIoU=True,
                inner_ratio=self.inner_iou_ratio,
            )
            loss_iou = ((1.0 - iou) * weight * reg_weight).sum() / target_scores_sum
        elif self.iou_type == "wciou_acloss":
            if pred_scores is None:
                raise ValueError("pred_scores must be provided when ULTRALYTICS_IOU_LOSS='wciou_acloss'.")
            ciou_parts = bbox_ciou_components(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
            matched_target_scores = target_scores[fg_mask]
            matched_pred_scores = pred_scores[fg_mask].sigmoid()
            matched_confidence = (
                (matched_pred_scores * matched_target_scores).sum(-1, keepdim=True)
                / matched_target_scores.sum(-1, keepdim=True).clamp_min(self.conf_eps)
            ).clamp_(self.conf_eps, 1.0)
            occlusion_weight = torch.exp(-matched_confidence)
            loss_wciou = (1.0 - ciou_parts["ciou"]) * occlusion_weight
            confidence_weight = (1.0 - matched_confidence).pow(self.wciou_ac_gamma)
            loss_ac = -torch.log(matched_confidence) * confidence_weight
            combined_loss = self.wciou_ac_lambda * loss_wciou + (1.0 - self.wciou_ac_lambda) * loss_ac
            loss_iou = (combined_loss * weight * reg_weight).sum() / target_scores_sum
        else:
            if self.iou_type != "ciou":
                LOGGER.warning(
                    f"Unknown ULTRALYTICS_IOU_LOSS='{self.iou_type}', falling back to CIoU regression loss."
                )
                self.iou_type = "ciou"
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight * reg_weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
                * weight
                * reg_weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max, h).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.current_epoch = 0
        if self.bbox_loss.iou_type == "wciou_acloss":
            LOGGER.info(
                "Using WCIoU-ACLoss for box regression "
                f"(lambda={self.bbox_loss.wciou_ac_lambda:.3f}, gamma={self.bbox_loss.wciou_ac_gamma:.3f})."
            )
        if self.bbox_loss.tal_reg_enable:
            LOGGER.info(
                "Using late-stage TAL regression reweighting "
                f"(start_epoch={self.bbox_loss.tal_reg_start_epoch}, ema_decay={self.bbox_loss.tal_reg_ema_decay:.3f}, "
                f"gain={self.bbox_loss.tal_reg_gain:.3f}, threshold={self.bbox_loss.tal_reg_threshold:.3f})."
            )
        if self.bbox_loss.sa_box_enable:
            LOGGER.info(
                "Using sample-aware box/DFL weighting "
                f"(alpha={self.bbox_loss.sa_small_alpha:.3f}, beta={self.bbox_loss.sa_elong_beta:.3f}, "
                f"gamma={self.bbox_loss.sa_class_gamma:.3f}, small_thr={self.bbox_loss.sa_small_area_thr:.6f}, "
                f"elong_thr={self.bbox_loss.sa_elong_ratio_thr:.3f}, target_cls={self.bbox_loss.sa_target_class_ids})."
            )
        self.aux_head_enable = _env_bool("ULTRALYTICS_AUX_HEAD_ENABLE", False)
        self.aux_head_gain = _env_float("ULTRALYTICS_AUX_HEAD_GAIN", 0.25)
        self.aux_small_area_thr = _env_float("ULTRALYTICS_AUX_SMALL_AREA_THR", 0.012521)
        self.aux_target_class_ids = _env_int_list("ULTRALYTICS_AUX_TARGET_CLASS_IDS", [2])
        self.aux_last_positive = 0
        self.cls_loss_type = _env_str("ULTRALYTICS_CLS_LOSS", "bce")
        if self.cls_loss_type == "atfl":
            self.cls_loss_fcn = AdaptiveThresholdFocalLoss(self.bce)
            LOGGER.info("Using Adaptive Threshold Focal Loss (ATFL) for detection classification loss.")
        elif self.cls_loss_type == "qfl":
            self.qfl_beta = _env_float("ULTRALYTICS_QFL_BETA", 2.0)
            self.cls_loss_fcn = QualityFocalLoss(beta=self.qfl_beta)
            LOGGER.info(f"Using Quality Focal Loss (QFL) for detection classification loss (beta={self.qfl_beta:.3f}).")
        else:
            if self.cls_loss_type != "bce":
                LOGGER.warning(
                    f"Unknown ULTRALYTICS_CLS_LOSS='{self.cls_loss_type}', falling back to BCE classification loss."
                )
            self.cls_loss_fcn = self.bce
        if self.aux_head_enable:
            LOGGER.info(
                "Using drill_pipe small-target auxiliary head "
                f"(gain={self.aux_head_gain:.3f}, small_thr={self.aux_small_area_thr:.6f}, "
                f"target_cls={self.aux_target_class_ids})."
            )

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _hyp_value(self, name: str, default: float) -> float:
        """Read a hyperparameter from either a namespace-like object or a dict."""
        if isinstance(self.hyp, dict):
            return float(self.hyp.get(name, default))
        return float(getattr(self.hyp, name, default))

    def _filter_aux_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Keep only small drill_pipe-style targets for the auxiliary head."""
        cls = batch["cls"].view(-1)
        bboxes = batch["bboxes"]
        area = bboxes[:, 2] * bboxes[:, 3]

        target_mask = torch.zeros_like(cls, dtype=torch.bool)
        for class_id in self.aux_target_class_ids:
            target_mask |= cls == class_id
        target_mask &= area < self.aux_small_area_thr

        self.aux_last_positive = int(target_mask.sum().item())
        if self.aux_last_positive == 0:
            return None

        aux_batch = dict(batch)
        aux_batch["batch_idx"] = batch["batch_idx"][target_mask]
        aux_batch["cls"] = batch["cls"][target_mask]
        aux_batch["bboxes"] = batch["bboxes"][target_mask]
        return aux_batch

    def _compute_path_loss(self, feats: list[torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute standard YOLO detection loss for one prediction path."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride[: len(feats)], 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.cls_loss_fcn(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            self.bbox_loss.current_epoch = self.current_epoch
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                pred_scores,
                target_bboxes_normalized=target_bboxes / imgsz[[1, 0, 1, 0]],
            )

        loss[0] *= self._hyp_value("box", 7.5)
        loss[1] *= self._hyp_value("cls", 0.5)
        loss[2] *= self._hyp_value("dfl", 1.5)
        return loss

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        self.bbox_loss.current_epoch = self.current_epoch
        batch_size = batch["img"].shape[0]

        if isinstance(preds, dict):
            main_feats = preds["main"]
            loss = self._compute_path_loss(main_feats, batch)
            aux_feats = preds.get("aux")
            if self.aux_head_enable and aux_feats is not None:
                aux_batch = self._filter_aux_batch(batch)
                if aux_batch is not None:
                    loss = loss + self.aux_head_gain * self._compute_path_loss(aux_feats, aux_batch)
            return loss * batch_size, loss.detach()

        feats = preds[1] if isinstance(preds, tuple) else preds
        loss = self._compute_path_loss(feats, batch)
        return loss * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.cls_loss_fcn(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                pred_scores,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, pose, kobj, cls, dfl
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.cls_loss_fcn(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                pred_scores,
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.cls_loss_fcn(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                pred_scores,
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        cls_loss = vp_loss[0][1]
        return cls_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
