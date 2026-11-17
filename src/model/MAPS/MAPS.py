from contextlib import nullcontext
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.MAPS.blocks import StdConv3d, NoBottleneck, conv3x3x3, IN_PLACE


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm="IN", activation="relu", weight_std=True, kernel_size=1):
        super().__init__()
        conv_cls = StdConv3d if weight_std else nn.Conv3d
        self.conv = conv_cls(in_channels, out_channels, kernel_size=kernel_size, bias=True)
        if norm == "IN":
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "GN":
            self.norm = nn.GroupNorm(16, out_channels)
        else:
            self.norm = nn.BatchNorm3d(out_channels)
        if activation == "relu":
            self.act = nn.ReLU(inplace=IN_PLACE)
        else:
            self.act = nn.LeakyReLU(1e-2, inplace=IN_PLACE)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MAPSBackbone(nn.Module):
    """
    HeartSeg backbone without structure-aware attention blocks.
    Keeps the shared CNN encoder-decoder and exposes the same two features:
    decoder features for the dynamic head and bottleneck features for routing.
    """

    def __init__(self, in_channels=1, weight_std=True):
        super().__init__()
        self.weight_std = weight_std
        self.conv1 = conv3x3x3(in_channels, 32, stride=(1, 1, 1), weight_std=weight_std)
        self.layer0 = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, 2, stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, 2, stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, 2, stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 320, 2, stride=(2, 2, 2))

        self.bottle = ConvBlock3d(320, 256, norm="IN", activation="relu", weight_std=weight_std, kernel_size=1)

        self.shortcut3 = ConvBlock3d(256, 256, norm="IN", activation="relu", weight_std=weight_std, kernel_size=1)
        self.shortcut2 = ConvBlock3d(128, 128, norm="IN", activation="relu", weight_std=weight_std, kernel_size=1)
        self.shortcut1 = ConvBlock3d(64, 64, norm="IN", activation="relu", weight_std=weight_std, kernel_size=1)
        self.shortcut0 = ConvBlock3d(32, 32, norm="IN", activation="relu", weight_std=weight_std, kernel_size=1)

        self.transposeconv_stage3 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, bias=False)

        self.stage3_de = self._make_layer(NoBottleneck, 256, 256, 1, stride=(1, 1, 1))
        self.stage2_de = self._make_layer(NoBottleneck, 128, 128, 1, stride=(1, 1, 1))
        self.stage1_de = self._make_layer(NoBottleneck, 64, 64, 1, stride=(1, 1, 1))
        self.stage0_de = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            ConvBlock3d(32, 8, norm="IN", activation="relu", weight_std=weight_std, kernel_size=1)
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=IN_PLACE),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0, weight_std=self.weight_std),
            )

        layers = [block(inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=multi_grid,
                        weight_std=self.weight_std)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes, dilation=dilation, multi_grid=multi_grid, weight_std=self.weight_std))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        skip0 = x

        x = self.layer1(x)
        skip1 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        skip3 = x

        x = self.layer4(x)
        bottleneck = self.bottle(x)

        x = self.transposeconv_stage3(bottleneck)
        x = x + self.shortcut3(skip3)
        x = self.stage3_de(x)

        x = self.transposeconv_stage2(x)
        x = x + self.shortcut2(skip2)
        x = self.stage2_de(x)

        x = self.transposeconv_stage1(x)
        x = x + self.shortcut1(skip1)
        x = self.stage1_de(x)

        x = self.transposeconv_stage0(x)
        x = x + self.shortcut0(skip0)
        x = self.stage0_de(x)

        head_inputs = self.precls_conv(x)
        return head_inputs, bottleneck


class DomainEmbeddingTransformer(nn.Module):
    """
    Transformer-based dynamic-head conditioner for MAPS.
    The domain token selects source-specific behavior while all domains share
    the same multi-class label space.
    """

    def __init__(self, num_domains=3, hidden_dim=256, feature_dim=256, num_queries=3, nhead=8, num_layers=2):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.domain_proj = nn.Linear(num_domains, hidden_dim)
        self.memory_proj = nn.Linear(feature_dim, hidden_dim)

    def forward(self, pooled_feature, domain_one_hot):
        b, _, _, _, _ = pooled_feature.shape
        memory = pooled_feature.flatten(2).transpose(1, 2)
        memory = self.memory_proj(memory)

        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        domain_token = self.domain_proj(domain_one_hot).unsqueeze(1)
        transformer_input = torch.cat([domain_token, queries, memory], dim=1)
        encoded = self.encoder(transformer_input)
        return encoded[:, 1:1 + self.num_queries]


class MAPS(nn.Module):
    """
    Source-conditioned dynamic segmentation with optional evidential outputs.

    The final trainer uses source-specific evidence and selected queries for
    evidence-weighted source alignment.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_domains: int = 3,
        weight_std: bool = True,
        hidden_dim: int = 256,
        num_queries: int = 3,
        dyn_head_dep_wid=(3, 8),
        domain_conditioning: bool = True,
        domain_specific_evidence: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_domains = num_domains
        self.num_queries = num_queries
        self.head_channels = dyn_head_dep_wid[1]
        self.num_head_layers = dyn_head_dep_wid[0]
        self.dyn_head_dep_wid = dyn_head_dep_wid
        self.domain_conditioning = domain_conditioning
        self.domain_specific_evidence = domain_specific_evidence
        if self.num_queries < self.num_domains:
            raise ValueError("num_queries must be >= num_domains for domain-conditioned routing")
        self._inference_domain_ids = None
        self._inference_task_ids = None

        self.backbone = MAPSBackbone(in_channels=in_channels, weight_std=weight_std)
        self.transformer_head = DomainEmbeddingTransformer(
            num_domains=num_domains,
            hidden_dim=hidden_dim,
            feature_dim=256,
            num_queries=num_queries,
            nhead=8,
            num_layers=2,
        )

        num_dyn_params = (self.head_channels * self.head_channels + self.head_channels) * (self.num_head_layers - 1) + (
            self.head_channels * num_classes + num_classes
        )
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=IN_PLACE),
            nn.Linear(hidden_dim, num_dyn_params),
        )
        def make_evidence_head():
            return nn.Sequential(
                nn.Conv3d(self.head_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Softplus(),
            )

        if domain_specific_evidence:
            self.evidence_heads = nn.ModuleList(make_evidence_head() for _ in range(num_domains))
        else:
            # Preserve the original architecture and checkpoint names by default.
            self.evidence_head = make_evidence_head()

    def _forward_evidence(self, features, domain_ids):
        if not self.domain_specific_evidence:
            return self.evidence_head(features)

        sample_indices, evidence = [], []
        for domain_id, head in enumerate(self.evidence_heads):
            indices = torch.nonzero(domain_ids == domain_id, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            sample_indices.append(indices)
            evidence.append(head(features.index_select(0, indices)))
        # Concatenation and reordering keep the gradient path through only the
        # selected heads. Inactive heads have grad=None, including under AdamW.
        order = torch.cat(sample_indices).argsort()
        return torch.cat(evidence, dim=0).index_select(0, order)

    def initialize_from_shared_evidence(self, state_dict, *, copy_evidence=False):
        """Load shared-head model weights as a new experiment's initialization.

        By default keep freshly initialized independent evidence heads; copying
        the old evidence head into every domain is an explicit opt-in. This is
        a weights-only transfer, not an optimizer/checkpoint resume operation.
        """
        if not self.domain_specific_evidence:
            raise ValueError("Shared-head transfer requires domain_specific_evidence=True")
        converted = OrderedDict(state_dict)
        shared = {}
        for name, expected in self.evidence_heads[0].state_dict().items():
            key = "evidence_head." + name
            if key not in converted:
                raise ValueError(f"Expected a shared-evidence checkpoint with key {key}")
            value = converted.pop(key)
            if value.shape != expected.shape:
                raise ValueError(f"Incompatible shared evidence parameter {key}: {value.shape} vs {expected.shape}")
            shared[name] = value
        for domain_id, head in enumerate(self.evidence_heads):
            values = shared if copy_evidence else head.state_dict()
            for name, value in values.items():
                converted[f"evidence_heads.{domain_id}.{name}"] = value.clone()
        return self.load_state_dict(converted, strict=True)

    def set_inference_domain_ids(self, domain_ids=None):
        self._inference_domain_ids = domain_ids
        self._inference_task_ids = domain_ids

    def clear_inference_domain_ids(self):
        self._inference_domain_ids = None
        self._inference_task_ids = None

    def set_inference_task_ids(self, task_ids=None):
        self.set_inference_domain_ids(task_ids)

    def clear_inference_task_ids(self):
        self.clear_inference_domain_ids()

    @staticmethod
    def _unwrap_network(network):
        return network.module if isinstance(network, DDP) else network

    def _format_domain_ids(self, x, domain_ids=None):
        batch_size = x.shape[0]
        device = x.device
        if domain_ids is None:
            domain_ids = self._inference_domain_ids
        if domain_ids is None:
            domain_ids = self._inference_task_ids
        if domain_ids is None:
            raise ValueError("MAPS requires domain_ids during training/inference")
        domain_ids = torch.as_tensor(domain_ids, dtype=torch.long, device=device)
        if domain_ids.ndim == 0:
            domain_ids = domain_ids.view(1)
        if domain_ids.ndim == 2 and domain_ids.shape[1] == 1:
            domain_ids = domain_ids.squeeze(1)
        if domain_ids.ndim != 1:
            raise ValueError(f"domain_ids must be 0D/1D/(B,1), got shape {tuple(domain_ids.shape)}")
        if domain_ids.shape[0] == 1 and batch_size > 1:
            domain_ids = domain_ids.expand(batch_size)
        if domain_ids.shape[0] != batch_size:
            raise ValueError(f"domain_ids batch mismatch: expected {batch_size}, got {domain_ids.shape[0]}")
        if torch.any(domain_ids < 0) or torch.any(domain_ids >= self.num_domains):
            raise ValueError(f"domain_ids must be in [0, {self.num_domains - 1}]")
        return domain_ids

    def _format_task_ids(self, x, task_ids=None):
        return self._format_domain_ids(x, task_ids)

    def encode_domain(self, domain_ids):
        return F.one_hot(domain_ids, num_classes=self.num_domains).float()

    @staticmethod
    def parse_dynamic_params(params, channels, weight_nums, bias_nums, final_out_channels):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)
        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * final_out_channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * final_out_channels)
        return weight_splits, bias_splits

    @staticmethod
    def heads_forward(features, weights, biases, num_insts):
        x = features
        n_layers = len(weights)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    @staticmethod
    def _remap_multiclass_logits(multiclass_logits):
        return multiclass_logits

    def _forward_impl(self, x, domain_ids, return_aux=False):
        routing_ids = domain_ids if self.domain_conditioning else torch.zeros_like(domain_ids)
        domain_one_hot = self.encode_domain(routing_ids).to(dtype=x.dtype, device=x.device)
        if not self.domain_conditioning:
            domain_one_hot = torch.zeros_like(domain_one_hot)

        head_inputs, bottleneck = self.backbone(x)
        domain_embeddings = self.transformer_head(bottleneck, domain_one_hot)

        selected_queries = domain_embeddings[torch.arange(x.shape[0], device=x.device), routing_ids]
        params = self.controller(selected_queries)
        # Evidence uses the actual source even if query routing is disabled by
        # a separate ablation; the original shared-head mode ignores these IDs.
        evidence = self._forward_evidence(head_inputs, domain_ids)
        alpha = evidence + 1.0

        n, _, d, h, w = head_inputs.shape
        head_inputs = head_inputs.reshape(1, -1, d, h, w)

        weight_nums = [self.head_channels * self.head_channels] * (self.num_head_layers - 1)
        bias_nums = [self.head_channels] * (self.num_head_layers - 1)
        weight_nums.append(self.head_channels * self.num_classes)
        bias_nums.append(self.num_classes)

        weights, biases = self.parse_dynamic_params(
            params,
            self.head_channels,
            weight_nums,
            bias_nums,
            self.num_classes,
        )
        multiclass_logits = self.heads_forward(head_inputs, weights, biases, n).reshape(-1, self.num_classes, d, h, w)

        output = self._remap_multiclass_logits(multiclass_logits)
        if not return_aux:
            return output

        aux = {
            "evidence": evidence,
            "alpha": alpha,
            "selected_queries": selected_queries,
            "domain_embeddings": domain_embeddings,
            "dynamic_params": params,
            "bottleneck": bottleneck,
        }
        return output, aux

    def forward(self, x, task_ids=None, domain_ids=None, return_aux=False):
        if domain_ids is None:
            domain_ids = task_ids
        domain_ids = self._format_domain_ids(x, domain_ids)

        autocast_ctx = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if x.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            return self._forward_impl(x.float(), domain_ids, return_aux=return_aux)
