#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash with the unholyapostolic config
# (~365 tok/s single-user decode; 2x RTX PRO 6000 / sm_120 / TP2).
#
#   docker pull verdictai/unholyapostolic:latest
#   MODEL_PATH=/path/to/deepseek-v4-flash GPUS=0,1 ./serve.sh
#
# ── EXACT REPRODUCTION ─────────────────────────────────────────────────────────
# This script + the image carry the FULL working env. There is ONE thing neither
# can carry: a HOST-level NVIDIA P2P override. The b12x PCIe all-reduce needs
# forced GPU<->GPU P2P on these no-NVLink cards. Without it you hit, during
# cudagraph capture:
#     torch.AcceleratorError: CUDA error: operation not permitted when stream is
#     capturing   (cudaErrorStreamCaptureUnsupported, in b12x .../pcie_oneshot.py)
# Fix it ONCE on the host (see README "Host prerequisites"):
#     echo 'options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"' \
#       | sudo tee /etc/modprobe.d/nvidia-p2p-override.conf
#     sudo update-initramfs -u && sudo reboot
# Validated on: driver 595.58.03 / CUDA 13.2, 2x RTX PRO 6000 Blackwell (sm_120),
# Resizable BAR enabled.
#
# To get unblocked WITHOUT b12x (slower, standard NCCL all-reduce, no host change):
#     add  -e VLLM_ENABLE_PCIE_ALLREDUCE=0
#
# NOTE: the NATIVE Lightning Indexer is used on purpose — do NOT set
# VLLM_USE_B12X_SPARSE_INDEXER (the b12x sparse indexer corrupts >131K retrieval).
# ───────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Pin by digest for byte-exact reproduction (see README for the current sha256):
#   IMAGE="verdictai/unholyapostolic@sha256:6fcfe17e80142e0ce182f816ce143d2d1803aca32a48ab63fb5f934004a362ad"
IMAGE="${IMAGE:-verdictai/unholyapostolic:latest}"
MODEL_PATH="${MODEL_PATH:-/models/deepseek-v4-flash}"   # host path to the DSV4-Flash weights
GPUS="${GPUS:-0,1}"                                      # the two GPUs to use (TP2)
PORT="${PORT:-9201}"

# The image already bakes every -e below as a default; they are repeated here so
# this script is self-contained and correct even against an older pull. The
# docker-runtime flags (--ipc/--shm-size/--ulimit/--gpus) CANNOT live in an image.
docker run --rm \
  --gpus "device=${GPUS}" \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_PATH}:/model:ro" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=2048 \
  -e VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_NET_GDR_LEVEL=SYS \
  -e CUTE_DSL_ARCH=sm_120a \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  "${IMAGE}" \
  vllm serve /model \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 136000 \
    --max-num-batched-tokens 12288 \
    --max-num-seqs 8 \
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
