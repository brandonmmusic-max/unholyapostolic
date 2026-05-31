# unholyapostolic — Version Readout (exact pinned stack)

Complete version + config manifest for the validated `unholyapostolic` run
(TP2 / 136K / `draft_sample_method=probabilistic`, **351 tok/s** single-user decode).
Everything below ships **inside the pinned image** unless flagged as a host prerequisite.

- **Published image (env-baked):** `verdictai/unholyapostolic@sha256:6fcfe17e80142e0ce182f816ce143d2d1803aca32a48ab63fb5f934004a362ad`
- The benchmark below ran on the functionally identical **pre-env-bake** image `sha256:807047f49f80…` (same vLLM / b12x / deep_gemm bits; the new digest only adds the baked `ENV`).
- OS: Ubuntu 22.04 · amd64 · image size 34.3 GB

## Build lineage
`FROM` the unholy-fusion sm_120 base (vLLM `dev/unholy-fusion` fork + b12x) → installed `deep_gemm-2.5.0+76e93aa` wheel (DeepGEMM PR #324) → baked runtime `ENV`. No vLLM recompile for the #324 swap.

## vLLM serve command
```bash
vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 136000 \
  --max-num-batched-tokens 12288 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --host 0.0.0.0 --port 9201 \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs.thinking true \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --stream-interval 8
```
Resolved server config: dtype `bfloat16` · quant `deepseek_v4_fp8` · KV `fp8` · TP2 · max_len 136000 · cudagraph `FULL_AND_PIECEWISE` · custom_ops `all` · MTP k=3 · reasoning style `parser_only_omit_effort` · native indexer (b12x sparse indexer OFF) · all-reduce dispatch `B12X_PCIE_ONESHOT` → `PYNCCL` fallback.

## Runtime env (baked into the image)
```text
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_USE_B12X_MOE=1
VLLM_ALLREDUCE_USE_SYMM_MEM=0
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=2048
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_LEVEL=SYS
NCCL_NET_GDR_LEVEL=SYS
CUDA_HOME=/usr/local/cuda
FLASHINFER_DISABLE_VERSION_CHECK=1
CUTE_DSL_ARCH=sm_120a
TORCH_CUDA_ARCH_LIST=12.0a
```
Container runtime flags (from `serve.sh`): IPC mode `host`, shm 32 GiB, ulimits `memlock=-1` / `stack=67108864`.

## Core versions (inside the image)
```text
python              3.12.13
vllm                0.1.dev1+gbfad804ed   (commit bfad804edd…, branch dev/unholy-fusion)
torch               2.11.0+cu130          (torch.version.cuda 13.0, nccl 2.28.9)
torchvision         0.26.0+cu130
triton              3.6.0
flashinfer-python   0.6.12                (flashinfer-cubin 0.6.11.post2)
b12x                0.15.3                (commit 87dc6e0034a21c6c58e1fb8e7fd1e47e28258ee8)
deep_gemm           2.5.0+76e93aa         (DeepGEMM PR #324, commit 76e93aa8ba…)
transformers        5.9.0
tokenizers          0.22.2
safetensors         0.7.0
nvidia-nccl-cu13    2.28.9
nvidia-cutlass-dsl  4.5.1
numpy               2.3.5   ·   pydantic 2.13.4   ·   fastapi 0.136.3
```
CUDA toolchain in image: `CUDA_VERSION=13.0.2` · cuBLAS 13.1.0.3 · cuDNN 9.19.0.56 · nvrtc 13.0.88 · cuSPARSELt 0.8.0.
The image digest pins the **full** environment exactly; for the complete inventory: `docker run --rm <image> /vllm/.venv/bin/pip freeze`.

### vLLM local patches (in the image — NOT reproducible via pip)
`/vllm` is HEAD `bfad804` (`dev/unholy-fusion`) with these tracked files modified:
```text
vllm/engine/arg_utils.py
vllm/model_executor/layers/fused_moe/b12x_moe.py
vllm/model_executor/warmup/kernel_warmup.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/single_type_kv_cache_manager.py
```
→ Reproduce by **pulling the pinned image**, not by `pip install`-ing these versions onto vanilla vLLM.

## Host environment (NOT in the image — prerequisites)
- NVIDIA driver **595.58.03**, CUDA 13.2 (driver), kernel Linux 6.18.x (Pop!_OS 24.04)
- 4× **NVIDIA RTX PRO 6000 Blackwell** (sm_120), PCIe gen5 x16, **300 W** power limit each; the served container pinned to 2 GPUs (TP2 → **600 W** for inference)
- GPU↔GPU topology: `NODE` (PCIe, no NVLink)
- **NVIDIA P2P override** (host modprobe — required for b12x; see README → Host prerequisites):
```text
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
```

## Model (`config.json`)
```text
architectures   ["DeepseekV4ForCausalLM"]   model_type deepseek_v4   torch_dtype bfloat16
max_position_embeddings 1048576   vocab_size 129280   hidden_size 4096   num_hidden_layers 43
num_attention_heads 64   num_key_value_heads 1   moe_intermediate_size 2048
n_routed_experts 256   n_shared_experts 1   num_experts_per_tok 6
q_lora_rank 1024   qk_rope_head_dim 64
rope_scaling  yarn · factor 16 · original_max_position_embeddings 65536 · beta_fast 32 · beta_slow 1
quantization  fp8 · e4m3 · ue8m0 scale · dynamic activation · weight_block_size [128,128]
generation_config  do_sample true · temperature 1.0 · top_p 1.0 · eos_token_id 1
```

## Headline results (this exact stack)
Prefill (single-prompt, integrated scout):
```text
8k     8,191 tok    1.006 s TTFT   8,145 tok/s
16k   16,250 tok    2.203 s TTFT   7,378 tok/s
32k   32,340 tok    4.530 s TTFT   7,138 tok/s
64k   64,555 tok    9.717 s TTFT   6,644 tok/s
128k 128,980 tok   21.359 s TTFT   6,039 tok/s
```
Aggregate sustained decode tok/s (`*` = capacity/warmup-limited, kept via `--show-capacity-limited-values`; `skip` = didn't fit KV):
```text
ctx\conc     1      2      4      8     16     32     64
0        351.2  675.4  310.4  451.1  433.4* 436.8* 438.1*
16k      341.9  651.1  241.1  417.2  412.4* 414.6* 422.6*
32k      340.1  646.5  303.1  406.6  400.1* 375.6* 397.3*
64k      333.0  630.5* 295.5  402.6* 421.2* 402.4*  skip
128k     319.7  603.2  325.0  402.6  379.4*  skip   skip
```
KV budget reported by vLLM: 2,580,736 tokens (10,081 blocks × 256).

**MTP acceptance** over the run: **mean 2.31 / median 1.92 / ~44%** (222/222 windows > 1.0, range 1.64–4.00). The 4.00 / 99.8% peak windows are the formulaic long-generation tail — **not** representative; cite the full-run mean (2.31).

## Benchmark harness
`llm_decode_bench` v0.4.24 (commit `1438ed34…`). Args: concurrency `1,2,4,8,16,32,64` × contexts `0,16384,32768,65536,131072`, `max_tokens=2048`, `duration=30 s/cell`, prefill `integrated_decode_scout`, `ignore_eos=true`, warmup 3 s @ 131072 ctx / C=1.
