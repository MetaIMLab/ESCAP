"""Final source-conditioned ESCAP trainer for three cardiac CT protocols."""

from __future__ import annotations

from contextlib import nullcontext
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.loss.compound_losses import PartialLabelDCAndCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler

from src.model.MAPS.MAPS import MAPS


SOURCE_PREFIXES = (("CHD68_", 0), ("ImageALCAPA_", 1),
                   ("MM_WHS_", 2), ("MM-WHS_", 2))
SOURCE_CLASS_MASKS = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
    dtype=torch.bool,
)


def source_id(case_identifier: str) -> int:
    for prefix, index in SOURCE_PREFIXES:
        if case_identifier.startswith(prefix):
            return index
    raise ValueError(f"Unknown source for case {case_identifier!r}")


def source_ids(keys, device):
    if keys is None:
        raise ValueError("ESCAP requires source-prefixed case identifiers")
    return torch.tensor([source_id(key) for key in keys], dtype=torch.long, device=device)


def balanced_evidential_loss(alpha, target, domain_ids):
    target_idx = target.squeeze(1).long()
    num_classes = alpha.shape[1]
    class_masks = SOURCE_CLASS_MASKS.to(alpha.device)[domain_ids]
    valid_range = (target_idx >= 0) & (target_idx < num_classes)
    safe_target = target_idx.clamp(0, num_classes - 1)
    valid_class = class_masks.gather(1, safe_target.reshape(alpha.shape[0], -1)).reshape_as(target_idx)
    valid = valid_range & valid_class
    if not torch.any(valid):
        return alpha.sum() * 0.0

    alpha_y = alpha.gather(1, safe_target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
    mask = class_masks.view(alpha.shape[0], num_classes, *([1] * (alpha.ndim - 2)))
    alpha_sum = (alpha * mask).sum(dim=1).clamp_min(1e-6)
    per_voxel = (torch.digamma(alpha_sum) - torch.digamma(alpha_y))[valid]

    domains = domain_ids.reshape((-1,) + (1,) * (target_idx.ndim - 1))
    group_ids = (domains * num_classes + safe_target)[valid]
    num_domains = int(domain_ids.max().item()) + 1
    bins = num_domains * num_classes
    counts = torch.bincount(group_ids, minlength=bins).reshape(num_domains, num_classes)
    sums = per_voxel.new_zeros(bins, dtype=torch.float64).scatter_add(
        0, group_ids, per_voxel.double()
    ).reshape(num_domains, num_classes)
    present = counts > 0
    class_means = sums / counts.clamp_min(1)
    source_means = class_means.sum(1) / present.sum(1).clamp_min(1)
    return source_means[present.any(1)].mean().to(per_voxel.dtype)


def evidence_weighted_source_alignment(alpha, selected_queries, domain_ids, temperature=0.25):
    eps = 1e-6
    pooled = alpha.detach().mean(dim=tuple(range(2, alpha.ndim)))
    alpha_sum = pooled.sum(dim=1, keepdim=True).clamp_min(eps)
    normalized = pooled / alpha_sum
    concentration = pooled.max(dim=1).values / alpha_sum.squeeze(1)
    queries = F.normalize(selected_queries.float(), dim=1)
    total_loss = queries.sum() * 0.0
    total_weight = queries.new_tensor(0.0)

    for domain_id in torch.unique(domain_ids):
        indices = torch.nonzero(domain_ids == domain_id, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        discrepancy = torch.cdist(normalized[indices], normalized[indices], p=1)
        pair_uncertainty = 1.0 - 0.5 * (
            concentration[indices, None] + concentration[None, indices]
        )
        weights = torch.exp(-discrepancy * pair_uncertainty.clamp_min(0.0) / temperature).detach()
        off_diagonal = ~torch.eye(indices.numel(), dtype=torch.bool, device=indices.device)
        distances = 1.0 - queries[indices] @ queries[indices].T
        total_loss = total_loss + (weights[off_diagonal] * distances[off_diagonal]).sum()
        total_weight = total_weight + weights[off_diagonal].sum()
    if total_weight.item() <= 0:
        return total_loss
    return total_loss / total_weight.clamp_min(eps)


class SourceBatchDataLoader3D(nnUNetDataLoader3D):
    """Sample one known annotation protocol per training batch."""

    def get_indices(self):
        if not hasattr(self, "_source_groups"):
            groups = {index: [] for index in range(3)}
            for key in self.indices:
                groups[source_id(key)].append(key)
            self._source_groups = {index: keys for index, keys in groups.items() if keys}
        indices = np.asarray(sorted(self._source_groups), dtype=np.int64)
        weights = np.asarray([2.0 if index in (0, 1) else 1.0 for index in indices])
        chosen = int(np.random.choice(indices, p=weights / weights.sum()))
        keys = self._source_groups[chosen]
        return np.random.choice(np.asarray(keys, dtype=object), self.batch_size,
                                replace=len(keys) < self.batch_size).tolist()


class MAPSBalancedThreeEvidenceHeadsTrainer(nnUNetTrainer):
    """Final three-source, three-evidence-head ESCAP configuration."""

    def __init__(self, loss="DC_and_CE_loss", optimizer="AdamW", lr=1e-4,
                 num_epochs=100, batch_size=8, plans=None, configuration="3d_fullres",
                 fold=0, dataset_json=None, unpack_dataset=True, device=torch.device("cuda")):
        if plans is None or plans.get("dataset_name") != "Dataset130_MCWHS":
            raise ValueError("ESCAP requires Dataset130_MCWHS plans")
        if dataset_json is None or len(dataset_json["labels"]) != 10 or len(dataset_json["channel_names"]) != 1:
            raise ValueError("ESCAP requires one CT channel and ten output classes")
        if configuration != "3d_fullres" or loss != "DC_and_CE_loss" or optimizer != "AdamW":
            raise ValueError("Unsupported ESCAP training configuration")
        self.augmentation_seed = 42
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        self.loss_name = loss
        self.optimizer_name = optimizer
        self.lr = lr
        self.batch_size_per_gpu = batch_size
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        plans["configurations"][configuration]["batch_size"] = world_size * batch_size
        super().__init__(loss, optimizer, lr, num_epochs, batch_size, plans, configuration,
                         fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.save_every = 10

    @staticmethod
    def build_network_architecture(architecture_class_name=None, arch_init_kwargs=None,
                                   arch_init_kwargs_req_import=None, num_input_channels=1,
                                   num_output_channels=10, enable_deep_supervision=False):
        if num_input_channels != 1 or num_output_channels != 10:
            raise ValueError("ESCAP requires one CT channel and ten output classes")
        model = MAPS(in_channels=1, num_classes=10, num_domains=3, weight_std=True,
                     hidden_dim=256, num_queries=3, dyn_head_dep_wid=(3, 8),
                     domain_specific_evidence=True)
        model.maps_partial_labels = True
        model.maps_evidential_weight = 0.1
        model.maps_consistency_weight = 0.0
        model.maps_eau_weight = 0.1
        model.maps_evidential_class_balanced = True
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _build_loss(self):
        return PartialLabelDCAndCE_loss(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
            domain_class_masks=SOURCE_CLASS_MASKS,
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            smooth=1e-5,
            do_bg=False,
        )

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError("ESCAP trainer is already initialized")
        self.num_input_channels = 1
        self.network = self.build_network_architecture().to(self.device)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)
        self.loss = self._build_loss()
        self.was_initialized = True

    def configure_optimizers(self):
        shared, evidence = [], []
        for name, parameter in self.network.named_parameters():
            (evidence if name.startswith("evidence_heads.") else shared).append(parameter)
        optimizer = torch.optim.AdamW(
            [{"params": shared, "lr": self.lr, "name": "shared_and_dynamic"},
             {"params": evidence, "lr": self.lr * 10, "name": "evidence_heads"}],
            betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
        )
        scheduler = get_scheduler("cosine_with_restarts", optimizer, num_warmup_steps=50,
                                  num_training_steps=self.num_epochs * self.num_iterations_per_epoch)
        return optimizer, scheduler

    def on_train_epoch_start(self):
        self.network.train()
        completed_steps = self.current_epoch * self.num_iterations_per_epoch
        if self.lr_scheduler.last_epoch < completed_steps:
            self.lr_scheduler.step(completed_steps)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.8f}")
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)
        self.print_to_log_file("Learning rates by parameter group:", {
            group["name"]: group["lr"] for group in self.optimizer.param_groups
        })

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        (rotation, dummy_2d, initial_patch_size, mirror_axes) = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        training_transforms = self.get_training_transforms(
            patch_size, rotation, self._get_deep_supervision_scales(), mirror_axes, dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        validation_transforms = self.get_validation_transforms(
            self._get_deep_supervision_scales(), is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        train_dataset, val_dataset = self.get_tr_and_val_datasets()
        train_loader = SourceBatchDataLoader3D(
            train_dataset, self.batch_size, initial_patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=training_transforms,
        )
        val_loader = nnUNetDataLoader3D(
            val_dataset, self.batch_size, patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=validation_transforms,
        )
        workers = get_allowed_n_proc_DA()
        if workers == 0:
            train_aug = SingleThreadedAugmenter(train_loader, None)
            val_aug = SingleThreadedAugmenter(val_loader, None)
        else:
            val_workers = max(1, workers // 2)
            train_aug = NonDetMultiThreadedAugmenter(
                data_loader=train_loader, transform=None, num_processes=workers,
                num_cached=max(6, workers // 2),
                seeds=[self.augmentation_seed + i for i in range(workers)],
                pin_memory=self.device.type == "cuda", wait_time=0.002,
            )
            val_aug = NonDetMultiThreadedAugmenter(
                data_loader=val_loader, transform=None, num_processes=val_workers,
                num_cached=max(3, workers // 4),
                seeds=[self.augmentation_seed + 1000 + i for i in range(val_workers)],
                pin_memory=self.device.type == "cuda", wait_time=0.002,
            )
        _ = next(train_aug)
        _ = next(val_aug)
        return train_aug, val_aug

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)
        domains = source_ids(batch.get("keys"), self.device)
        self.optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(self.device.type) if self.device.type == "cuda" else nullcontext()
        with context:
            output, aux = self.network(data, domain_ids=domains, return_aux=True)
            loss = self.loss(output, target, domains)
            loss = loss + 0.1 * balanced_evidential_loss(aux["alpha"], target, domains)
            loss = loss + 0.1 * evidence_weighted_source_alignment(
                aux["alpha"], aux["selected_queries"], domains
            )
        invalid = (~torch.isfinite(loss)).float()
        if self.is_ddp:
            dist.all_reduce(invalid, op=dist.ReduceOp.SUM)
        if invalid.item() > 0:
            raise FloatingPointError("Non-finite ESCAP training loss")
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
        self.lr_scheduler.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)
        domains = source_ids(batch.get("keys"), self.device)
        context = torch.autocast(self.device.type) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data, domain_ids=domains)
            loss = self.loss(output, target, domains)
        prediction = torch.zeros_like(output).scatter_(1, output.argmax(1, keepdim=True), 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            prediction, target, axes=list(range(2, output.ndim)), mask=None
        )
        class_mask = SOURCE_CLASS_MASKS.to(output.device)[domains].to(tp.dtype)
        tp = (tp * class_mask).sum(0)
        fp = (fp * class_mask).sum(0)
        fn = (fn * class_mask).sum(0)
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp[1:].detach().cpu().numpy(),
            "fp_hard": fp[1:].detach().cpu().numpy(),
            "fn_hard": fn[1:].detach().cpu().numpy(),
        }

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, _ = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        self.inference_allowed_mirroring_axes = None
        return rotation, dummy_2d, initial_patch_size, None

    def configure_inference_for_case(self, predictor, case_identifier: str, properties=None):
        network = predictor.network.module if isinstance(predictor.network, DDP) else predictor.network
        index = source_id(case_identifier)
        self.print_to_log_file(f"{case_identifier}, source_id {index}")
        network.set_inference_domain_ids(index)

    def clear_inference_for_case(self, predictor):
        network = predictor.network.module if isinstance(predictor.network, DDP) else predictor.network
        network.clear_inference_domain_ids()
