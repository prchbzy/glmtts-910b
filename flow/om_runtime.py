from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import acl

from flow.export_utils import parse_buckets, select_flow_bucket


INPUT_ORDER_V2 = [
    'middle_point_btd',
    'condition_btd',
    'precomputed_text_embed',
    'time_step_1d',
    'padding_mask_bt',
    'spkr_emb_bd',
]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _pad_tensor(tensor: torch.Tensor, target_len: int, dim: int = 1, pad_value=0):
    cur = tensor.shape[dim]
    if cur == target_len:
        return tensor
    if cur > target_len:
        raise ValueError(f'tensor len {cur} exceeds target_len {target_len}')
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - cur
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=dim)

def _get_acl_context():
    result = acl.rt.get_context()
    if isinstance(result, tuple):
        if len(result) == 2:
            first, second = result
            # CosyVoice2 patch uses: context, ret_code = acl.rt.get_context()
            if isinstance(second, int):
                ctx, ret = first, second
                if ret != 0:
                    raise RuntimeError(f'acl.rt.get_context failed: ret={ret}')
                return ctx
            if isinstance(first, int):
                ret, ctx = first, second
                if ret != 0:
                    raise RuntimeError(f'acl.rt.get_context failed: ret={ret}')
                return ctx
        return result[0]
    return result


def _set_acl_context(ctx):
    result = acl.rt.set_context(ctx)
    if isinstance(result, tuple):
        ret = result[0]
    else:
        ret = result
    if isinstance(ret, int) and ret != 0:
        raise RuntimeError(f'acl.rt.set_context failed: ret={ret}')



class AisBenchOMSession:
    def __init__(self, model_path: str, device_id: int, py_ctx=None):
        from ais_bench.infer.interface import InferSession
        self.model_path = model_path
        self.device_id = int(device_id)
        torch.npu.set_device(self.device_id)
        self.py_ctx = py_ctx if py_ctx is not None else _get_acl_context()
        self.session = InferSession(device_id=self.device_id, model_path=model_path)
        self.om_ctx = _get_acl_context()

    def set_context(self):
        torch.npu.set_device(self.device_id)
        if hasattr(self.session, 'set_context'):
            result = self.session.set_context()
            if isinstance(result, tuple):
                ret = result[0]
            else:
                ret = result
            if isinstance(ret, int) and ret != 0:
                raise RuntimeError(f'InferSession.set_context failed: ret={ret}')
            return
        if self.om_ctx is not None:
            _set_acl_context(self.om_ctx)

    def restore_py_context(self):
        if self.py_ctx is not None:
            _set_acl_context(self.py_ctx)

    def infer(self, feeds):
        torch.npu.set_device(self.device_id)
        self.set_context()
        return self.session.infer(feeds=feeds, mode='static')

    def free_resource(self):
        self.set_context()
        self.session.free_resource()


class FlowOMEstimatorManager:
    def __init__(self, om_dir: str, buckets, prefix: str = 'flow_estimator_v2_b', device_id: int = 0):
        self.om_dir = Path(om_dir)
        self.buckets = parse_buckets(buckets)
        self.prefix = prefix
        self.device_id = int(device_id)
        self.sessions = {}
        self.model_paths = {}
        self._announced_buckets = set()
        self.primary_bucket = None

        torch.npu.set_device(self.device_id)
        self.py_ctx = _get_acl_context()

        for bucket in self.buckets:
            model_path = self.om_dir / f'{self.prefix}{bucket}.om'
            if model_path.exists():
                self.sessions[bucket] = AisBenchOMSession(str(model_path), self.device_id, py_ctx=self.py_ctx)
                self.model_paths[bucket] = str(model_path)
                if self.primary_bucket is None:
                    self.primary_bucket = bucket
                print(f'[flow_om] loaded bucket={bucket} model={model_path} device_id={self.device_id}')

        if not self.sessions:
            raise FileNotFoundError(
                f'no OM models found under {self.om_dir} for prefix={self.prefix}, buckets={self.buckets}'
            )
        self.buckets = sorted(self.sessions.keys())
        self.primary_bucket = self.primary_bucket if self.primary_bucket is not None else self.buckets[0]
        self._restore_primary_then_py()

    def _restore_primary_then_py(self):
        primary = self.sessions.get(self.primary_bucket)
        if primary is not None:
            primary.set_context()
        if self.py_ctx is not None:
            _set_acl_context(self.py_ctx)

    def select_bucket(self, seq_len: int) -> int:
        return select_flow_bucket(int(seq_len), self.buckets)

    def infer(
        self,
        middle_point_btd: torch.Tensor,
        condition_btd: torch.Tensor,
        precomputed_text_embed: torch.Tensor,
        time_step_1d: torch.Tensor,
        padding_mask_bt: torch.Tensor,
        spkr_emb_bd: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = int(middle_point_btd.shape[1])
        bucket = self.select_bucket(seq_len)
        if bucket not in self.sessions:
            raise KeyError(f'bucket {bucket} not loaded; available={self.buckets}')

        if bucket not in self._announced_buckets:
            print(
                '[flow_om] first infer bucket='
                f'{bucket} seq_len={seq_len} middle={tuple(middle_point_btd.shape)} '
                f'condition={tuple(condition_btd.shape)} text_embed={tuple(precomputed_text_embed.shape)}'
            )
            self._announced_buckets.add(bucket)

        middle_point_btd = _pad_tensor(middle_point_btd, bucket, dim=1, pad_value=0)
        condition_btd = _pad_tensor(condition_btd, bucket, dim=1, pad_value=0)
        precomputed_text_embed = _pad_tensor(precomputed_text_embed, bucket, dim=1, pad_value=0)
        padding_mask_bt = _pad_tensor(padding_mask_bt, bucket, dim=1, pad_value=False)

        feeds = [
            middle_point_btd.detach().cpu().float().numpy(),
            condition_btd.detach().cpu().float().numpy(),
            precomputed_text_embed.detach().cpu().float().numpy(),
            time_step_1d.detach().cpu().float().numpy(),
            padding_mask_bt.detach().cpu().bool().numpy(),
            spkr_emb_bd.detach().cpu().float().numpy(),
        ]
        session = self.sessions[bucket]
        outputs = session.infer(feeds)
        self._restore_primary_then_py()
        output = outputs[0]
        output_np = np.asarray(output)
        result_cpu = torch.from_numpy(output_np)
        result = result_cpu.to(device=middle_point_btd.device, dtype=middle_point_btd.dtype)
        return result[:, :seq_len, :]


def maybe_create_flow_om_manager():
    if not _env_flag('GLMTTS_FLOW_OM_ENABLE', False):
        return None
    try:
        __import__('ais_bench.infer.interface')
    except Exception as e:
        print(f'[flow_om] runtime unavailable, fallback to PyTorch: {e}')
        return None

    om_dir = os.environ.get('GLMTTS_FLOW_OM_DIR', 'exported/flow_om')
    buckets = os.environ.get('GLMTTS_FLOW_OM_BUCKETS', '256,512,768,1024')
    prefix = os.environ.get('GLMTTS_FLOW_OM_PREFIX', 'flow_estimator_v2_b')
    device_id = int(os.environ.get('GLMTTS_FLOW_OM_DEVICE_ID', '0'))
    try:
        torch.npu.set_device(device_id)
        manager = FlowOMEstimatorManager(om_dir=om_dir, buckets=buckets, prefix=prefix, device_id=device_id)
        print(f'[flow_om] enabled buckets={manager.buckets} dir={om_dir} prefix={prefix} device_id={device_id}')
        return manager
    except Exception as e:
        print(f'[flow_om] init failed, fallback to PyTorch: {e}')
        return None
