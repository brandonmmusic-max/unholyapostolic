# unholyapostolic — DeepSeek-V4-Flash, consumer Blackwell (sm_120), TP2

A tuned serving stack for **DeepSeek-V4-Flash** on **2× RTX PRO 6000 (sm_120, PCIe, no NVLink)** that roughly **doubles single-user decode** vs the original Lucifer image — at a **300 W per-GPU cap**.

## Pull
```bash
docker pull verdictai/unholyapostolic:latest                    # Docker Hub
docker pull ghcr.io/brandonmmusic-max/unholyapostolic:latest    # GitHub Container Registry
```

## Run
```bash
MODEL_PATH=/path/to/deepseek-v4-flash GPUS=0,1 ./serve.sh
```
See [`serve.sh`](serve.sh) for the full command. It boots the 365-tok/s config (V2 model runner + MTP k=3 + native Lightning Indexer + parser-only reasoning + b12x PCIe all-reduce) and serves an OpenAI-compatible API on port 9201.

## Numbers (2× RTX PRO 6000, TP2, 300 W cap each)
| metric | original Lucifer (MTP baseline) | **unholyapostolic** |
|---|---|---|
| single-user decode (short ctx) | ~185 tok/s | **~365 tok/s** (~2×) |
| long-ctx decode @134K | — | **339–352 tok/s** |
| prefill @134K (native) | — | **~6.5K tok/s** |
| estonia retrieval (30-shot, byte-identical) | — | **28/30** (2 misses = stopping artifacts, not retrieval errors) |

Reference point: a ~2,100 W TP4 community run measured **39.4 tok/s/request** decode at concurrency-30; this TP2 stack does **~52 tok/s/request** — faster per-request decode on **half the GPUs and ~⅓ the power.**

## What got the ~2× jump (in order of impact)
1. **V2 model runner** (`VLLM_USE_V2_MODEL_RUNNER=1`) — the keystone. nsys profiling showed ~46% of each decode token was **GPU idle waiting on the host** (kernel-launch + piecewise-cudagraph gaps). The V2 runner collapses that per-sub-step host gap. Everything below only pays *because* of this.
2. **MTP k=3** (`--speculative-config.num_speculative_tokens 3`). On the V1 runner, k=3 *lost* (per-sub-step host overhead > speculation gain). On V2 the collapsed gap makes each extra draft sub-step nearly free, so **k=3 pays** — "hide the deeper MTP in the freed host budget." This + (1) is the bulk of the 2×. (k=4 does *not* pay: position-4 accept decays to 3.4%, mean accept-len flat ~2.4 — k=3 is the ceiling for DSV4's single trained MTP head.)
3. **Tuned serve config**: `--max-model-len 136000 --max-num-batched-tokens 12288 --gpu-memory-utilization 0.95 --stream-interval 8`, `custom_ops:["all"]`, cudagraph `FULL_AND_PIECEWISE`.
4. **Native Lightning Indexer** (do *not* set `VLLM_USE_B12X_SPARSE_INDEXER`) — the b12x sparse indexer **corrupts long-context retrieval** (>131K); native is both correct (28–30/30 estonia) *and* faster single-user.
5. **Parser-only reasoning** — the V2 runner rejects reasoning-budget enforcement, so `--reasoning-parser deepseek_v4` + `--default-chat-template-kwargs.thinking true`, and **no `--reasoning-config.*`**.
6. **DeepGEMM PR #324** — the SM120 "complete odd/large next_n / kPadOddN port" — swapped in as a **site-packages wheel, no vLLM recompile** (`deep_gemm 2.5.0+76e93aa`, torch 2.11). Keeps the 365 decode and runs the upstreamed SM120 indexer-scoring kernels; validated **retrieval-preserving** (28/30 byte-identical estonia).

## Build
The published image is ready to pull. To rebuild the #324 layer, see [`Dockerfile`](Dockerfile) — it documents the reproducible delta (DeepGEMM PR #324 swapped onto the unholy-fusion base; no vLLM recompile).

## Repo contents
- [`serve.sh`](serve.sh) — run the image with the 365-tok/s config
- [`Dockerfile`](Dockerfile) — the DeepGEMM PR #324 build layer
- `README.md` — this file

## Known issue (being fixed)
2/30 estonia shots over-reason to the 40,000-token cap (a stopping/reasoning-length quirk — not a retrieval or kernel error) → 28/30 instead of 30/30. A serving-level stopping fix is in progress; the result + stop flag will land here.
