import torch
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class PartialLabelDCAndCE_loss(nn.Module):
    """
    DC + CE for datasets with different annotation protocols.

    ``domain_class_masks[d, c]`` tells the loss whether class ``c`` is
    annotated by domain ``d``. Unannotated classes are excluded from the
    per-sample softmax and Dice calculation instead of being treated as
    background.
    """

    def __init__(
        self,
        batch_dice,
        ddp,
        domain_class_masks,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        smooth=1e-5,
        do_bg=False,
    ):
        super().__init__()
        domain_class_masks = torch.as_tensor(domain_class_masks, dtype=torch.bool)
        if domain_class_masks.ndim != 2:
            raise ValueError(
                "domain_class_masks must have shape (num_domains, num_classes), "
                f"got {tuple(domain_class_masks.shape)}"
            )
        self.register_buffer("domain_class_masks", domain_class_masks, persistent=False)
        self.batch_dice = batch_dice
        self.ddp = ddp
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ignore_label = ignore_label
        self.smooth = smooth
        self.do_bg = do_bg

    def _get_domain_class_masks(self, domain_ids, num_classes):
        domain_ids = domain_ids.long().reshape(-1)
        if domain_ids.numel() == 0:
            raise ValueError("domain_ids must contain at least one sample")
        if domain_ids.min() < 0 or domain_ids.max() >= self.domain_class_masks.shape[0]:
            raise ValueError(
                f"domain_ids out of range: {domain_ids.detach().cpu().tolist()}"
            )
        if self.domain_class_masks.shape[1] != num_classes:
            raise ValueError(
                "Partial-label mask class count does not match network output: "
                f"{self.domain_class_masks.shape[1]} vs {num_classes}"
            )
        return self.domain_class_masks.to(device=domain_ids.device)[domain_ids]

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, domain_ids: torch.Tensor):
        if target.ndim != net_output.ndim or target.shape[1] != 1:
            raise ValueError(
                "PartialLabelDCAndCE_loss expects target shape (B, 1, ...), "
                f"got target={tuple(target.shape)}, output={tuple(net_output.shape)}"
            )

        batch_size, num_classes = net_output.shape[:2]
        domain_ids = domain_ids.reshape(-1)
        if domain_ids.shape[0] != batch_size:
            raise ValueError(
                f"domain_ids batch mismatch: {domain_ids.shape[0]} vs {batch_size}"
            )

        target_idx = target[:, 0].long()
        domain_class_mask = self._get_domain_class_masks(domain_ids, num_classes)
        class_mask = domain_class_mask.view(
            batch_size, num_classes, *([1] * (net_output.ndim - 2))
        )

        target_in_range = (target_idx >= 0) & (target_idx < num_classes)
        safe_target = target_idx.clamp(0, num_classes - 1)
        target_class_valid = domain_class_mask.gather(
            1, safe_target.reshape(batch_size, -1)
        ).reshape_as(target_idx)
        valid_voxels = target_in_range & target_class_valid
        if self.ignore_label is not None:
            valid_voxels &= target_idx != self.ignore_label

        # The softmax denominator contains only classes annotated by this
        # sample's domain. Invalid class logits therefore receive no gradient.
        masked_logits = net_output.masked_fill(~class_mask, -1e4)
        log_prob = F.log_softmax(masked_logits, dim=1)

        ce_per_voxel = -log_prob.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        if self.weight_ce != 0 and torch.any(valid_voxels):
            ce_loss = ce_per_voxel[valid_voxels].mean()
        else:
            ce_loss = net_output.sum() * 0.0

        if self.weight_dice != 0:
            probabilities = log_prob.exp()
            target_onehot = F.one_hot(
                safe_target, num_classes=num_classes
            ).movedim(-1, 1).to(dtype=probabilities.dtype)
            valid_class_voxels = class_mask & valid_voxels.unsqueeze(1)
            probabilities = probabilities * valid_class_voxels
            target_onehot = target_onehot * valid_class_voxels

            axes = tuple(range(2, net_output.ndim))
            true_positive = (probabilities * target_onehot).sum(axes)
            sum_prediction = probabilities.sum(axes)
            sum_target = target_onehot.sum(axes)

            if self.batch_dice:
                true_positive = true_positive.sum(0)
                sum_prediction = sum_prediction.sum(0)
                sum_target = sum_target.sum(0)
                dice = (
                    2 * true_positive + self.smooth
                ) / (sum_prediction + sum_target + self.smooth).clamp_min(1e-8)
                domain_has_annotation = domain_class_mask.any(dim=0)
                if not self.do_bg:
                    domain_has_annotation = domain_has_annotation.clone()
                    domain_has_annotation[0] = False
                dice_loss = (
                    -dice[domain_has_annotation].mean()
                    if torch.any(domain_has_annotation)
                    else net_output.sum() * 0.0
                )
            else:
                dice = (
                    2 * true_positive + self.smooth
                ) / (sum_prediction + sum_target + self.smooth).clamp_min(1e-8)
                class_has_annotation = domain_class_mask
                if not self.do_bg:
                    class_has_annotation = class_has_annotation.clone()
                    class_has_annotation[:, 0] = False
                dice_loss = (
                    -dice[class_has_annotation].mean()
                    if torch.any(class_has_annotation)
                    else net_output.sum() * 0.0
                )
        else:
            dice_loss = net_output.sum() * 0.0

        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
