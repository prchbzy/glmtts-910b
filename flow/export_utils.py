from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn


DEFAULT_FLOW_BUCKETS = (256, 512, 768, 1024)
DEFAULT_FLOW_BLOCK_PATTERN = (25, 50, 200)


def parse_buckets(buckets: Sequence[int] | str | None) -> list[int]:
    if buckets is None:
        values = list(DEFAULT_FLOW_BUCKETS)
    elif isinstance(buckets, str):
        values = [int(item.strip()) for item in buckets.split(",") if item.strip()]
    else:
        values = [int(item) for item in buckets]
    values = sorted(set(v for v in values if v > 0))
    if not values:
        raise ValueError("at least one positive bucket is required")
    return values


def select_flow_bucket(target_mel_len: int, buckets: Sequence[int] | str | None = None) -> int:
    target_mel_len = int(target_mel_len)
    if target_mel_len <= 0:
        raise ValueError(f"target_mel_len must be positive, got {target_mel_len}")
    for bucket in parse_buckets(buckets):
        if target_mel_len <= bucket:
            return bucket
    raise ValueError(
        f"target_mel_len={target_mel_len} exceeds configured buckets={parse_buckets(buckets)}"
    )


def estimate_token_len(flow, mel_len: int) -> int:
    return max(1, math.ceil(float(mel_len) * flow.input_frame_rate / flow.mel_framerate))


class FlowEstimatorExportWrapper(nn.Module):
    """V1 wrapper: keeps text_emb_layer inside exported graph."""

    def __init__(self, flow, is_causal: bool = False, block_pattern: Iterable[int] | None = None):
        super().__init__()
        self.estimator = flow.estimator
        self.is_causal = bool(is_causal)
        self.block_pattern = list(block_pattern) if block_pattern is not None else None

    def forward(
        self,
        middle_point_btd: torch.Tensor,
        condition_btd: torch.Tensor,
        text: torch.Tensor,
        time_step_1d: torch.Tensor,
        padding_mask_bt: torch.Tensor,
        spkr_emb_bd: torch.Tensor,
    ) -> torch.Tensor:
        return self.estimator(
            middle_point_btd=middle_point_btd,
            condition_btd=condition_btd,
            text=text,
            time_step_1d=time_step_1d,
            padding_mask_bt=padding_mask_bt,
            spkr_emb_bd=spkr_emb_bd,
            is_causal=self.is_causal,
            block_pattern=self.block_pattern,
        )


class FlowEstimatorExportWrapperV2(nn.Module):
    """V2 wrapper: text embedding is precomputed outside exported graph."""

    def __init__(self, flow, is_causal: bool = False, block_pattern: Iterable[int] | None = None):
        super().__init__()
        self.estimator = flow.estimator
        self.is_causal = bool(is_causal)
        self.block_pattern = list(block_pattern) if block_pattern is not None else None

    def forward(
        self,
        middle_point_btd: torch.Tensor,
        condition_btd: torch.Tensor,
        precomputed_text_embed: torch.Tensor,
        time_step_1d: torch.Tensor,
        padding_mask_bt: torch.Tensor,
        spkr_emb_bd: torch.Tensor,
    ) -> torch.Tensor:
        return self.estimator(
            middle_point_btd=middle_point_btd,
            condition_btd=condition_btd,
            text=torch.zeros(1, 1, device=middle_point_btd.device, dtype=torch.long),
            time_step_1d=time_step_1d,
            padding_mask_bt=padding_mask_bt,
            spkr_emb_bd=spkr_emb_bd,
            is_causal=self.is_causal,
            block_pattern=self.block_pattern,
            precomputed_text_embed=precomputed_text_embed,
        )


def _build_common_inputs(flow, bucket: int, batch_size: int, dtype, device):
    middle_point_btd = torch.randn(batch_size, bucket, flow.mel_dim, device=device, dtype=dtype)
    mel_cond_btd = torch.zeros(batch_size, bucket, flow.mel_dim, device=device, dtype=dtype)

    if not flow.remove_spkr_concat_condition:
        spkr_in = 192
        spkr_emb_bd = torch.randn(batch_size, spkr_in, device=device, dtype=dtype)
        spkr_embedding = flow.spk_embed_affine_layer(spkr_emb_bd)
        spkr_embedding_expanded = spkr_embedding.unsqueeze(1).expand(-1, bucket, -1)
        condition_btd = torch.cat([mel_cond_btd, spkr_embedding_expanded], dim=-1)
    else:
        spkr_in = 192 if not flow.use_wavlm_emb else 192 + 256
        spkr_emb_bd = torch.randn(batch_size, spkr_in, device=device, dtype=dtype)
        condition_btd = mel_cond_btd

    time_step_1d = torch.zeros(batch_size, device=device, dtype=dtype)
    padding_mask_bt = torch.ones(batch_size, bucket, device=device, dtype=torch.bool)
    return middle_point_btd, condition_btd, time_step_1d, padding_mask_bt, spkr_emb_bd


def build_flow_export_inputs(
    flow,
    bucket: int,
    *,
    batch_size: int = 1,
    token_len: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    bucket = int(bucket)
    if bucket <= 0:
        raise ValueError(f"bucket must be positive, got {bucket}")

    device = device or next(flow.parameters()).device
    dtype = dtype or next(flow.parameters()).dtype
    token_len = int(token_len or estimate_token_len(flow, bucket))
    middle_point_btd, condition_btd, time_step_1d, padding_mask_bt, spkr_emb_bd = _build_common_inputs(
        flow, bucket, batch_size, dtype, device
    )
    text = torch.zeros(batch_size, token_len, device=device, dtype=torch.long)
    return {
        'middle_point_btd': middle_point_btd,
        'condition_btd': condition_btd,
        'text': text,
        'time_step_1d': time_step_1d,
        'padding_mask_bt': padding_mask_bt,
        'spkr_emb_bd': spkr_emb_bd,
    }


def build_flow_export_inputs_v2(
    flow,
    bucket: int,
    *,
    batch_size: int = 1,
    token_len: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    bucket = int(bucket)
    if bucket <= 0:
        raise ValueError(f"bucket must be positive, got {bucket}")

    device = device or next(flow.parameters()).device
    dtype = dtype or next(flow.parameters()).dtype
    token_len = int(token_len or estimate_token_len(flow, bucket))
    middle_point_btd, _condition_btd, time_step_1d, padding_mask_bt, spkr_emb_bd = _build_common_inputs(
        flow, bucket, batch_size, dtype, device
    )
    text = torch.zeros(batch_size, token_len, device=device, dtype=torch.long)
    with torch.no_grad():
        precomputed_text_embed = flow.estimator.text_emb_layer(text, bucket)

    # Export V2 keeps speaker conditioning outside condition_btd so the concat dim
    # matches estimator.emb_concator.proj's real input expectation on current ckpt.
    condition_btd = torch.zeros(batch_size, bucket, flow.mel_dim, device=device, dtype=dtype)

    return {
        'middle_point_btd': middle_point_btd,
        'condition_btd': condition_btd,
        'precomputed_text_embed': precomputed_text_embed,
        'time_step_1d': time_step_1d,
        'padding_mask_bt': padding_mask_bt,
        'spkr_emb_bd': spkr_emb_bd,
    }
