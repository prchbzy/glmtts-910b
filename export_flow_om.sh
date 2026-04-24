#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

source "${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

detect_soc_version() {
  if [ -n "${FLOW_OM_SOC_VERSION:-}" ]; then
    printf '%s
' "$FLOW_OM_SOC_VERSION"
    return 0
  fi

  if command -v npu-smi >/dev/null 2>&1; then
    local detected
    detected="$(npu-smi info 2>/dev/null | sed -n 's/.*Name[[:space:]]*:[[:space:]]*//p' | grep -E 'Ascend[0-9A-Za-z]+' | head -n1 | tr -d '')"
    if [ -n "$detected" ]; then
      printf '%s
' "$detected"
      return 0
    fi
  fi

  printf '%s
' 'Ascend910B1'
}

FLOW_OM_SOC_VERSION="$(detect_soc_version)"
FLOW_WRAPPER_VERSION="${FLOW_WRAPPER_VERSION:-v2}"
FLOW_BUCKETS="${FLOW_BUCKETS:-256,512,768,1024}"
FLOW_ONNX_DIR="${FLOW_ONNX_DIR:-exported/flow_onnx}"
FLOW_OM_DIR="${FLOW_OM_DIR:-exported/flow_om}"
FLOW_OM_PRECISION_MODE_V2="${FLOW_OM_PRECISION_MODE_V2:-origin}"
FLOW_OM_IMPLMODE="${FLOW_OM_IMPLMODE:-high_precision}"
FLOW_OM_LOG_LEVEL="${FLOW_OM_LOG_LEVEL:-error}"
mkdir -p "$FLOW_OM_DIR"

if command -v atc >/dev/null 2>&1; then
  echo "[export_flow_om] atc=$(atc --version 2>/dev/null | head -n1)"
else
  echo "[export_flow_om] error: atc not found in PATH" >&2
  exit 1
fi
echo "[export_flow_om] soc_version=$FLOW_OM_SOC_VERSION wrapper=$FLOW_WRAPPER_VERSION buckets=$FLOW_BUCKETS onnx_dir=$FLOW_ONNX_DIR om_dir=$FLOW_OM_DIR"

IFS=',' read -r -a BUCKET_ARRAY <<< "$FLOW_BUCKETS"

for bucket in "${BUCKET_ARRAY[@]}"; do
  bucket="${bucket// /}"
  [ -n "$bucket" ] || continue

  if [ "$FLOW_WRAPPER_VERSION" = "v2" ]; then
    model_path="$FLOW_ONNX_DIR/flow_estimator_v2_b${bucket}.onnx"
    output_path="$FLOW_OM_DIR/flow_estimator_v2_b${bucket}"
    input_shape="middle_point_btd:1,${bucket},80;condition_btd:1,${bucket},80;precomputed_text_embed:1,${bucket},512;time_step_1d:1;padding_mask_bt:1,${bucket};spkr_emb_bd:1,192"
  else
    model_path="$FLOW_ONNX_DIR/flow_estimator_v1_b${bucket}.onnx"
    output_path="$FLOW_OM_DIR/flow_estimator_v1_b${bucket}"
    text_len=$((bucket / 2))
    input_shape="middle_point_btd:1,${bucket},80;condition_btd:1,${bucket},80;text:1,${text_len};time_step_1d:1;padding_mask_bt:1,${bucket};spkr_emb_bd:1,192"
  fi

  echo "[export_flow_om] wrapper=$FLOW_WRAPPER_VERSION bucket=$bucket model=$model_path output=$output_path"
  atc --framework=5     --soc_version="$FLOW_OM_SOC_VERSION"     --model="$model_path"     --output="$output_path"     --input_format=ND     --input_shape="$input_shape"     --precision_mode_v2="$FLOW_OM_PRECISION_MODE_V2"     --op_select_implmode="$FLOW_OM_IMPLMODE"     --log="$FLOW_OM_LOG_LEVEL"
done

echo "[export_flow_om] done wrapper=$FLOW_WRAPPER_VERSION buckets=$FLOW_BUCKETS output_dir=$FLOW_OM_DIR"
