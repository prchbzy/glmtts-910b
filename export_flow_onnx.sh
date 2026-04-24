#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

FLOW_CKPT="${FLOW_CKPT:-ckpt/flow/flow.pt}"
FLOW_CONFIG="${FLOW_CONFIG:-ckpt/flow/config.yaml}"
FLOW_ONNX_DIR="${FLOW_ONNX_DIR:-exported/flow_onnx}"
FLOW_BUCKETS="${FLOW_BUCKETS:-256,512,768,1024}"
FLOW_WRAPPER_VERSION="${FLOW_WRAPPER_VERSION:-v2}"
FLOW_ONNX_OPSET="${FLOW_ONNX_OPSET:-17}"
FLOW_BLOCK_PATTERN="${FLOW_BLOCK_PATTERN:-3,3,3,3,3,3,3,3,3,3}"
FLOW_IS_CAUSAL="${FLOW_IS_CAUSAL:-0}"

CMD=(python3 tools/export_flow_estimator_onnx.py
  --flow_ckpt "$FLOW_CKPT"
  --flow_config "$FLOW_CONFIG"
  --output_dir "$FLOW_ONNX_DIR"
  --buckets "$FLOW_BUCKETS"
  --wrapper_version "$FLOW_WRAPPER_VERSION"
  --opset "$FLOW_ONNX_OPSET"
  --block_pattern "$FLOW_BLOCK_PATTERN")

if [ "$FLOW_IS_CAUSAL" = "1" ]; then
  CMD+=(--is_causal)
fi

echo "[export_flow_onnx] ckpt=$FLOW_CKPT config=$FLOW_CONFIG output=$FLOW_ONNX_DIR buckets=$FLOW_BUCKETS wrapper=$FLOW_WRAPPER_VERSION opset=$FLOW_ONNX_OPSET causal=$FLOW_IS_CAUSAL"
"${CMD[@]}"
