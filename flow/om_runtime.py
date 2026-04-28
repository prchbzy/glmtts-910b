from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
import acl

from flow.export_utils import parse_buckets, select_flow_bucket


INPUT_ORDER_V2 = [
    "middle_point_btd",
    "condition_btd",
    "precomputed_text_embed",
    "time_step_1d",
    "padding_mask_bt",
    "spkr_emb_bd",
]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ret_value(result: Any):
    """
    Return (value, ret) for ACL Python APIs that may return:
      - value
      - ret
      - (value, ret)
      - (ret, value)

    For pointer-like values, the ret code is usually the small integer 0.
    """
    if isinstance(result, tuple):
        if len(result) == 2:
            a, b = result
            # Most ACL Python APIs return (value, ret). The ret code is usually 0.
            if isinstance(b, int) and abs(b) < 1000000:
                return a, b
            # Some patched APIs return (ret, value).
            if isinstance(a, int) and abs(a) < 1000000:
                return b, a
            # Ambiguous pointer-like tuple: keep first as value, assume success.
            return a, 0
        if len(result) > 0:
            return result[0], 0
    return result, 0


def _ret_code(result: Any) -> int:
    """
    Extract return code for ACL APIs that may return:
      - ret
      - (dataset, ret)
      - (ret, dataset)

    Important: acl.mdl.add_dataset_buffer may return (dataset, ret).
    The dataset pointer is a huge integer and must NOT be treated as ret.
    """
    if isinstance(result, tuple):
        # Prefer a small integer ret code, especially the last item.
        for v in reversed(result):
            if isinstance(v, int) and abs(v) < 1000000:
                return v
        return 0
    if isinstance(result, int):
        # Pointer-like huge ints are not ret codes.
        if abs(result) >= 1000000:
            return 0
        return result
    return 0


def _check_ret(ret: int, op: str):
    if isinstance(ret, int) and ret != 0:
        raise RuntimeError(f"{op} failed: ret={ret}")


def _get_acl_context():
    ctx, ret = _ret_value(acl.rt.get_context())
    _check_ret(ret, "acl.rt.get_context")
    return ctx


def _set_acl_context(ctx):
    if ctx is None:
        return
    result = acl.rt.set_context(ctx)
    ret = result[0] if isinstance(result, tuple) else result
    _check_ret(ret, "acl.rt.set_context")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _pad_tensor(tensor: torch.Tensor, target_len: int, dim: int = 1, pad_value=0):
    cur = int(tensor.shape[dim])
    if cur == target_len:
        return tensor
    if cur > target_len:
        raise ValueError(f"tensor len {cur} exceeds target_len {target_len}")
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - cur
    pad_tensor = torch.full(
        pad_shape,
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad_tensor], dim=dim)


def _destroy_dataset(dataset):
    if dataset is None:
        return
    num_buffers = acl.mdl.get_dataset_num_buffers(dataset)
    for i in range(num_buffers):
        data_buf = acl.mdl.get_dataset_buffer(dataset, i)
        if data_buf is not None:
            acl.destroy_data_buffer(data_buf)
    acl.mdl.destroy_dataset(dataset)


def _create_dataset_from_tensors(tensors: list[torch.Tensor]):
    dataset = acl.mdl.create_dataset()
    buffers = []
    try:
        for tensor in tensors:
            # acl.create_data_buffer may return buf or (buf, ret), depending on CANN/Python binding.
            buf_result = acl.create_data_buffer(
                int(tensor.data_ptr()),
                int(_tensor_nbytes(tensor)),
            )
            buf, ret = _ret_value(buf_result)
            _check_ret(ret, "acl.create_data_buffer")
            buffers.append(buf)

            add_result = acl.mdl.add_dataset_buffer(dataset, buf)
            ret = _ret_code(add_result)
            _check_ret(ret, "acl.mdl.add_dataset_buffer")
        return dataset, buffers
    except Exception:
        # If creation fails halfway, destroy what we have allocated.
        _destroy_dataset(dataset)
        raise


class AclMdlOMSession:
    """
    Minimal ACL model session that executes OM in the current PyTorch NPU context.

    This class intentionally avoids ais_bench.InferSession, because ais_bench may
    manage a separate ACL context. Here input and output buffers are PyTorch NPU
    tensors, passed to acl.mdl.execute via their data_ptr().
    """

    def __init__(self, model_path: str, device_id: int):
        self.model_path = str(model_path)
        self.device_id = int(device_id)

        torch.npu.set_device(self.device_id)
        # IMPORTANT: capture the PyTorch-created/current ACL context.
        self.ctx = _get_acl_context()
        _set_acl_context(self.ctx)

        load_result = acl.mdl.load_from_file(self.model_path)
        self.model_id, ret = _ret_value(load_result)
        _check_ret(ret, f"acl.mdl.load_from_file({self.model_path})")

        self.desc = acl.mdl.create_desc()
        ret = _ret_code(acl.mdl.get_desc(self.desc, self.model_id))
        _check_ret(ret, "acl.mdl.get_desc")

        self.input_num = acl.mdl.get_num_inputs(self.desc)
        self.output_num = acl.mdl.get_num_outputs(self.desc)

        if self.input_num != len(INPUT_ORDER_V2):
            raise RuntimeError(
                f"OM input count mismatch: model has {self.input_num}, "
                f"expected {len(INPUT_ORDER_V2)}"
            )
        if self.output_num < 1:
            raise RuntimeError("OM has no outputs")

        print(
            f"[flow_acl_mdl] loaded model={self.model_path} "
            f"model_id={self.model_id} inputs={self.input_num} outputs={self.output_num} "
            f"device_id={self.device_id}"
        )

    def execute(self, input_tensors: list[torch.Tensor], output_tensors: list[torch.Tensor]):
        if len(input_tensors) != self.input_num:
            raise ValueError(f"expected {self.input_num} inputs, got {len(input_tensors)}")
        if len(output_tensors) != self.output_num:
            raise ValueError(f"expected {self.output_num} outputs, got {len(output_tensors)}")

        torch.npu.set_device(self.device_id)
        _set_acl_context(self.ctx)

        # Make sure PyTorch has finished writing input tensors before ACL reads them.
        torch.npu.synchronize()

        input_dataset = None
        output_dataset = None
        try:
            input_dataset, _ = _create_dataset_from_tensors(input_tensors)
            output_dataset, _ = _create_dataset_from_tensors(output_tensors)

            ret = _ret_code(acl.mdl.execute(self.model_id, input_dataset, output_dataset))
            _check_ret(ret, "acl.mdl.execute")

            # Make sure ACL finished writing output tensors before PyTorch reads them.
            torch.npu.synchronize()
            return output_tensors
        finally:
            _destroy_dataset(input_dataset)
            _destroy_dataset(output_dataset)

    def close(self):
        torch.npu.set_device(self.device_id)
        _set_acl_context(self.ctx)
        if getattr(self, "desc", None) is not None:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        if getattr(self, "model_id", None) is not None:
            ret = _ret_code(acl.mdl.unload(self.model_id))
            if isinstance(ret, int) and ret != 0:
                print(f"[flow_acl_mdl] unload warning model_id={self.model_id} ret={ret}")
            self.model_id = None


class FlowOMEstimatorManager:
    def __init__(
        self,
        om_dir: str,
        buckets,
        prefix: str = "flow_estimator_v2_b",
        device_id: int = 0,
    ):
        self.om_dir = Path(om_dir)
        self.buckets = parse_buckets(buckets)
        self.prefix = prefix
        self.device_id = int(device_id)
        self.sessions: dict[int, AclMdlOMSession] = {}
        self.model_paths: dict[int, str] = {}
        self._announced_buckets = set()
        self.last_profile = {}
        self.request_profile = {}

        torch.npu.set_device(self.device_id)
        self.py_ctx = _get_acl_context()

        for bucket in self.buckets:
            model_path = self.om_dir / f"{self.prefix}{bucket}.om"
            if model_path.exists():
                self.sessions[bucket] = AclMdlOMSession(str(model_path), self.device_id)
                self.model_paths[bucket] = str(model_path)
                print(
                    f"[flow_acl_mdl] bucket={bucket} model={model_path} "
                    f"device_id={self.device_id}"
                )

        if not self.sessions:
            raise FileNotFoundError(
                f"no OM models found under {self.om_dir} for prefix={self.prefix}, "
                f"buckets={self.buckets}"
            )

        self.buckets = sorted(self.sessions.keys())
        self.reset_profile()
        _set_acl_context(self.py_ctx)

    def select_bucket(self, seq_len: int) -> int:
        return select_flow_bucket(int(seq_len), self.buckets)

    def reset_profile(self):
        self.request_profile = {
            "om_calls": 0,
            "om_pad": 0.0,
            "om_input_to_numpy": 0.0,   # kept for compatibility; now means input_prepare
            "om_session_infer": 0.0,
            "om_output_to_torch_cpu": 0.0,  # kept for compatibility; always 0 in zero-copy
            "om_output_to_device": 0.0,     # kept for compatibility; always 0 in zero-copy
            "om_total": 0.0,
        }

    def get_profile(self):
        return dict(self.request_profile)

    def infer(
        self,
        middle_point_btd: torch.Tensor,
        condition_btd: torch.Tensor,
        precomputed_text_embed: torch.Tensor,
        time_step_1d: torch.Tensor,
        padding_mask_bt: torch.Tensor,
        spkr_emb_bd: torch.Tensor,
    ) -> torch.Tensor:
        infer_start = time.perf_counter()
        seq_len = int(middle_point_btd.shape[1])
        bucket = self.select_bucket(seq_len)

        if bucket not in self.sessions:
            raise KeyError(f"bucket {bucket} not loaded; available={self.buckets}")

        if bucket not in self._announced_buckets:
            print(
                "[flow_acl_mdl] first infer bucket="
                f"{bucket} seq_len={seq_len} middle={tuple(middle_point_btd.shape)} "
                f"condition={tuple(condition_btd.shape)} "
                f"text_embed={tuple(precomputed_text_embed.shape)}"
            )
            self._announced_buckets.add(bucket)

        pad_start = time.perf_counter()
        middle_point_btd = _pad_tensor(middle_point_btd, bucket, dim=1, pad_value=0)
        condition_btd = _pad_tensor(condition_btd, bucket, dim=1, pad_value=0)
        precomputed_text_embed = _pad_tensor(
            precomputed_text_embed, bucket, dim=1, pad_value=0
        )
        padding_mask_bt = _pad_tensor(padding_mask_bt, bucket, dim=1, pad_value=False)
        om_pad = time.perf_counter() - pad_start

        input_start = time.perf_counter()

        # Keep these tensors alive until acl.mdl.execute completes.
        # IMPORTANT: these must stay on NPU. Do NOT .cpu().numpy().
        device_tensors = [
            middle_point_btd.detach().contiguous().float(),
            condition_btd.detach().contiguous().float(),
            precomputed_text_embed.detach().contiguous().float(),
            time_step_1d.detach().contiguous().float(),
            padding_mask_bt.detach().contiguous().bool(),
            spkr_emb_bd.detach().contiguous().float(),
        ]

        # Output is written directly by OM into this PyTorch NPU tensor.
        output_tensor = torch.empty(
            (1, bucket, 80),
            dtype=device_tensors[0].dtype,
            device=device_tensors[0].device,
        ).contiguous()

        input_prepare = time.perf_counter() - input_start

        session = self.sessions[bucket]
        session_start = time.perf_counter()
        session.execute(device_tensors, [output_tensor])
        om_session_infer = time.perf_counter() - session_start

        result = output_tensor[:, :seq_len, :]

        om_total = time.perf_counter() - infer_start
        profile = {
            "bucket": bucket,
            "seq_len": seq_len,
            "om_calls": 1,
            "om_pad": om_pad,
            "om_input_to_numpy": input_prepare,
            "om_session_infer": om_session_infer,
            "om_output_to_torch_cpu": 0.0,
            "om_output_to_device": 0.0,
            "om_total": om_total,
        }
        self.last_profile = profile

        self.request_profile["om_calls"] = int(self.request_profile.get("om_calls", 0)) + 1
        self.request_profile["om_pad"] = float(self.request_profile.get("om_pad", 0.0)) + om_pad
        self.request_profile["om_input_to_numpy"] = (
            float(self.request_profile.get("om_input_to_numpy", 0.0)) + input_prepare
        )
        self.request_profile["om_session_infer"] = (
            float(self.request_profile.get("om_session_infer", 0.0)) + om_session_infer
        )
        self.request_profile["om_output_to_torch_cpu"] = float(
            self.request_profile.get("om_output_to_torch_cpu", 0.0)
        )
        self.request_profile["om_output_to_device"] = float(
            self.request_profile.get("om_output_to_device", 0.0)
        )
        self.request_profile["om_total"] = float(self.request_profile.get("om_total", 0.0)) + om_total

        return result

    def close(self):
        for session in self.sessions.values():
            try:
                session.close()
            except Exception as e:
                print(f"[flow_acl_mdl] close warning: {e}")
        if getattr(self, "py_ctx", None) is not None:
            _set_acl_context(self.py_ctx)


def maybe_create_flow_om_manager():
    if not _env_flag("GLMTTS_FLOW_OM_ENABLE", False):
        return None

    om_dir = os.environ.get("GLMTTS_FLOW_OM_DIR", "exported/flow_om")
    buckets = os.environ.get("GLMTTS_FLOW_OM_BUCKETS", "256,512,768,1024")
    prefix = os.environ.get("GLMTTS_FLOW_OM_PREFIX", "flow_estimator_v2_b")
    device_id = int(os.environ.get("GLMTTS_FLOW_OM_DEVICE_ID", "0"))

    try:
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            print("[flow_acl_mdl] torch.npu unavailable, fallback to PyTorch")
            return None
        torch.npu.set_device(device_id)
        manager = FlowOMEstimatorManager(
            om_dir=om_dir,
            buckets=buckets,
            prefix=prefix,
            device_id=device_id,
        )
        print(
            f"[flow_acl_mdl] enabled buckets={manager.buckets} "
            f"dir={om_dir} prefix={prefix} device_id={device_id}"
        )
        return manager
    except Exception as e:
        print(f"[flow_acl_mdl] init failed, fallback to PyTorch: {e}")
        return None
