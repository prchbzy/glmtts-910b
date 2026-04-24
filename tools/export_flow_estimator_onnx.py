import argparse
from pathlib import Path

import torch

from flow.export_utils import (
    DEFAULT_FLOW_BLOCK_PATTERN,
    FlowEstimatorExportWrapper,
    FlowEstimatorExportWrapperV2,
    build_flow_export_inputs,
    build_flow_export_inputs_v2,
    parse_buckets,
)
from glmtts_inference import DEVICE
from utils import yaml_util


def export_bucket(flow, out_path: Path, bucket: int, opset: int, is_causal: bool, block_pattern: list[int], wrapper_version: str):
    if wrapper_version == 'v2':
        wrapper = FlowEstimatorExportWrapperV2(
            flow=flow,
            is_causal=is_causal,
            block_pattern=block_pattern,
        ).to(DEVICE)
        dummy = build_flow_export_inputs_v2(flow, bucket=bucket, device=DEVICE)
        args = (
            dummy['middle_point_btd'],
            dummy['condition_btd'],
            dummy['precomputed_text_embed'],
            dummy['time_step_1d'],
            dummy['padding_mask_bt'],
            dummy['spkr_emb_bd'],
        )
        input_names = [
            'middle_point_btd',
            'condition_btd',
            'precomputed_text_embed',
            'time_step_1d',
            'padding_mask_bt',
            'spkr_emb_bd',
        ]
    else:
        wrapper = FlowEstimatorExportWrapper(
            flow=flow,
            is_causal=is_causal,
            block_pattern=block_pattern,
        ).to(DEVICE)
        dummy = build_flow_export_inputs(flow, bucket=bucket, device=DEVICE)
        args = (
            dummy['middle_point_btd'],
            dummy['condition_btd'],
            dummy['text'],
            dummy['time_step_1d'],
            dummy['padding_mask_bt'],
            dummy['spkr_emb_bd'],
        )
        input_names = [
            'middle_point_btd',
            'condition_btd',
            'text',
            'time_step_1d',
            'padding_mask_bt',
            'spkr_emb_bd',
        ]

    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            out_path,
            input_names=input_names,
            output_names=['dphi_dt'],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )


def main():
    parser = argparse.ArgumentParser(description='Export Flow estimator buckets to ONNX.')
    parser.add_argument('--flow_ckpt', default='ckpt/flow/flow.pt')
    parser.add_argument('--flow_config', default='ckpt/flow/config.yaml')
    parser.add_argument('--output_dir', default='exported/flow_onnx')
    parser.add_argument('--buckets', default='256,512,768,1024')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--is_causal', action='store_true')
    parser.add_argument('--wrapper_version', default='v2', choices=['v1', 'v2'])
    parser.add_argument('--block_pattern', default=','.join(str(v) for v in DEFAULT_FLOW_BLOCK_PATTERN))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flow = yaml_util.load_flow_model(args.flow_ckpt, args.flow_config, DEVICE)
    flow.eval()
    buckets = parse_buckets(args.buckets)
    block_pattern = [int(item.strip()) for item in args.block_pattern.split(',') if item.strip()]

    for bucket in buckets:
        out_path = output_dir / f'flow_estimator_{args.wrapper_version}_b{bucket}.onnx'
        print(f'[export] wrapper={args.wrapper_version} bucket={bucket} -> {out_path}')
        export_bucket(
            flow=flow,
            out_path=out_path,
            bucket=bucket,
            opset=args.opset,
            is_causal=args.is_causal,
            block_pattern=block_pattern,
            wrapper_version=args.wrapper_version,
        )

    print(f'[done] exported {len(buckets)} bucket(s) to {output_dir}')


if __name__ == '__main__':
    main()
