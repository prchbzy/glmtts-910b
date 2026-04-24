#!/bin/bash
root_dir=$(dirname "$(readlink -f "$0")")
cd "$root_dir" || exit

get_idle_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "${CUDA_VISIBLE_DEVICES:-0}"
        return
    fi
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | \
    awk '{print NR-1, $1 + $2}' | sort -nk2 | head -n1 | cut -d' ' -f1
}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    idle_gpu=$(get_idle_gpu)
    export CUDA_VISIBLE_DEVICES="$idle_gpu"
else
    idle_gpu="$CUDA_VISIBLE_DEVICES"
fi

export GLMTTS_API_PORT="${GLMTTS_API_PORT:-8048}"
export GLMTTS_API_USE_MONGO="${GLMTTS_API_USE_MONGO:-true}"
export GLMTTS_MONGO_URI="mongodb://root:123456@glmtts-mongo:27017/glmtts?authSource=admin"
export GLMTTS_MONGO_DB="${GLMTTS_MONGO_DB:-glmtts}"
export GLMTTS_MONGO_VOICE_COLLECTION="${GLMTTS_MONGO_VOICE_COLLECTION:-voices}"

export GLMTTS_GRADIO_LLM_BACKEND="${GLMTTS_GRADIO_LLM_BACKEND:-hf}"
export GLMTTS_GRADIO_HF_ATTN_IMPLEMENTATION="${GLMTTS_GRADIO_HF_ATTN_IMPLEMENTATION:-eager}"
export GLMTTS_GRADIO_HF_GRAPH_DECODE="${GLMTTS_GRADIO_HF_GRAPH_DECODE:-true}"
export GLMTTS_GRADIO_HF_GRAPH_MAX_CACHE_LEN="${GLMTTS_GRADIO_HF_GRAPH_MAX_CACHE_LEN:-1536}"
export GLMTTS_GRADIO_HF_GRAPH_WARMUP_ITERS="${GLMTTS_GRADIO_HF_GRAPH_WARMUP_ITERS:-2}"

export GLMTTS_FLOW_DTYPE="${GLMTTS_FLOW_DTYPE:-fp32}"
export GLMTTS_FLOW_OM_ENABLE="${GLMTTS_FLOW_OM_ENABLE:-1}"
export GLMTTS_FLOW_OM_DIR="${GLMTTS_FLOW_OM_DIR:-exported/flow_om}"
export GLMTTS_FLOW_OM_BUCKETS="${GLMTTS_FLOW_OM_BUCKETS:-256,512,768,1024}"
export GLMTTS_FLOW_OM_PREFIX="${GLMTTS_FLOW_OM_PREFIX:-flow_estimator_v2_b}"
export GLMTTS_FLOW_OM_DEVICE_ID="${GLMTTS_FLOW_OM_DEVICE_ID:-0}"
export GLMTTS_FLOW_GRAPH_ENABLE="${GLMTTS_FLOW_GRAPH_ENABLE:-0}"
export GLMTTS_FLOW_COMPILE_ENABLE="${GLMTTS_FLOW_COMPILE_ENABLE:-0}"

echo "Launching GLM-TTS API | GPU=$idle_gpu | port=$GLMTTS_API_PORT"
echo "Mongo enabled=$GLMTTS_API_USE_MONGO | db=$GLMTTS_MONGO_DB | collection=$GLMTTS_MONGO_VOICE_COLLECTION"
echo "Inference backend=$GLMTTS_GRADIO_LLM_BACKEND | attn=$GLMTTS_GRADIO_HF_ATTN_IMPLEMENTATION | graph_decode=$GLMTTS_GRADIO_HF_GRAPH_DECODE"
echo "Flow OM=$GLMTTS_FLOW_OM_ENABLE | dtype=$GLMTTS_FLOW_DTYPE | om_prefix=$GLMTTS_FLOW_OM_PREFIX | flow_graph=$GLMTTS_FLOW_GRAPH_ENABLE | flow_compile=$GLMTTS_FLOW_COMPILE_ENABLE"

python -m tools.api_app
