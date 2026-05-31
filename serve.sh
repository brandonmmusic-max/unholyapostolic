#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash with the unholyapostolic config
# (~365 tok/s single-user decode; 2x RTX PRO 6000 / sm_120 / TP2).
#
#   docker pull verdictai/unholyapostolic:latest
#   MODEL_PATH=/path/to/deepseek-v4-flash GPUS=0,1 ./serve.sh
#
# NOTE: the NATIVE Lightning Indexer is used on purpose — do NOT set
# VLLM_USE_B12X_SPARSE_INDEXER (the b12x sparse indexer corrupts >131K retrieval).
set -euo pipefail

IMAGE="${IMAGE:-verdictai/unholyapostolic:latest}"
MODEL_PATH="${MODEL_PATH:-/models/deepseek-v4-flash}"   # host path to the DSV4-Flash weights
GPUS="${GPUS:-0,1}"                                      # the two GPUs to use (TP2)
PORT="${PORT:-9201}"

docker run --rm --gpus "device=${GPUS}" \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_PATH}:/model:ro" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=2048 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP=1 \
  "${IMAGE}" \
  vllm serve /model \
    --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 136000 \
    --max-num-batched-tokens 12288 \
    --stream-interval 8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-flashinfer-autotune \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --default-chat-template-kwargs.thinking true \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
