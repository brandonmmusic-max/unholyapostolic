# unholyapostolic — DeepSeek-V4-Flash, consumer Blackwell (sm_120), TP2

A tuned serving stack for **DeepSeek-V4-Flash** on **2× RTX PRO 6000 (sm_120, PCIe, no NVLink)** that roughly **doubles single-user decode** vs the original Lucifer image — at a **300 W per-GPU cap**.

## Pull
```bash
docker pull verdictai/unholyapostolic:latest                    # Docker Hub
docker pull ghcr.io/brandonmmusic-max/unholyapostolic:latest    # GitHub Container Registry

# byte-exact pin (env-baked build, 2026-05-31) — same digest on both registries:
docker pull verdictai/unholyapostolic@sha256:6fcfe17e80142e0ce182f816ce143d2d1803aca32a48ab63fb5f934004a362ad
```

> **Validated stack:** driver **595.58.03** / CUDA **13.2**, 2× **RTX PRO 6000 Blackwell** (sm_120, PCIe, Resizable BAR on). The image bakes every required env var; `serve.sh` adds the docker-runtime flags. **One host step is required first → [Host prerequisites](#host-prerequisites-one-time).**

## Host prerequisites (one-time)
The b12x PCIe all-reduce needs **forced GPU↔GPU P2P**, which these no-NVLink cards only do with an NVIDIA driver override on the **host** (it can't live in the image). Without it you hit, during cudagraph capture:
```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
  (cudaErrorStreamCaptureUnsupported, in b12x/.../pcie_oneshot.py → all_reduce)
```
Set it once and reboot:
```bash
echo 'options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"' | sudo tee /etc/modprobe.d/nvidia-p2p-override.conf
sudo update-initramfs -u   # Debian/Ubuntu/Pop!_OS; dracut -f on Fedora/RHEL
sudo reboot
```
Verify after reboot: `nvidia-smi -q | grep -i bar1` shows BAR1 ≈ full VRAM. **No host access / don't want b12x?** add `-e VLLM_ENABLE_PCIE_ALLREDUCE=0` (standard NCCL all-reduce — a bit slower, boots anywhere).

## Run
```bash
MODEL_PATH=/path/to/deepseek-v4-flash GPUS=0,1 ./serve.sh
```
See [`serve.sh`](serve.sh) for the full command. It boots the 365-tok/s config (V2 model runner + MTP k=3 with **`draft_sample_method=probabilistic`** + native Lightning Indexer + parser-only reasoning + b12x PCIe all-reduce) and serves an OpenAI-compatible API on port 9201.

## Numbers (2× RTX PRO 6000, TP2, 300 W cap each)
| metric | original Lucifer (MTP baseline) | **unholyapostolic** |
|---|---|---|
| single-user decode (short ctx) | ~185 tok/s | **~365 tok/s** (~2×) |
| long-ctx decode @134K | — | **339–352 tok/s** |
| prefill @134K (native) | — | **~6.5K tok/s** |
| estonia retrieval (30-shot, byte-identical) | — | **28/30** (2 misses = stopping artifacts, not retrieval errors) |

Reference point: a ~2,100 W TP4 community run measured **39.4 tok/s/request** decode at concurrency-30; this TP2 stack does **~52 tok/s/request** — faster per-request decode on **half the GPUs and ~⅓ the power.**

## Decode vs temperature — MTP works under sampling
vLLM's default `draft_sample_method="greedy"` makes the MTP drafter argmax-only, so under sampling (temperature > 0) the rejection sampler accepts **0%** of drafts and decode collapses from ~365 to ~90 tok/s. The serve config sets **`draft_sample_method=probabilistic`** — it samples the draft at the request temperature and feeds real draft probabilities to the rejection sampler, so MTP works at **every** temperature (fixes the upstream vLLM bug [#16899](https://github.com/vllm-project/vllm/pull/16899) / [#40149](https://github.com/vllm-project/vllm/issues/40149), upstream fix [#40269](https://github.com/vllm-project/vllm/pull/40269)):

| temperature | default (greedy draft) | **`draft_sample_method=probabilistic`** |
|---|---|---|
| 0.0 (greedy) | 363.8 tok/s · accept 3.99 | **365.5 tok/s · accept 3.99** |
| 0.7 (typical chat) | **90.8 tok/s · accept 1.00 (0%)** | **357.5 tok/s · accept 3.99** |
| 1.0 | 90.9 tok/s · accept 1.00 (0%) | **357.7 tok/s · accept 3.99** |

*(single-user, k=3, 2× RTX PRO 6000 / TP2 / 300 W; accept = mean accepted length, max 4 at k=3. The 3.99 above is on a predictable probe prompt; on diverse content acceptance averages ~2.3 — see the full benchmark below — but the decode-speed recovery holds regardless.)*

## Full decode benchmark (`llm_decode_bench` v0.4.24, sampled @ temp 1.0, with the fix)
Full concurrency × context sweep. **`conc=1` is the clean single-user number**; higher-concurrency cells are power/KV-limited at the 300 W / TP2 envelope (warmup timeouts), so read them as a floor, not a scaling curve.

**Aggregate decode tok/s**

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | **351** | 675 | 310 | 451 | 433* | 437* | 438* |
| 16K | 342 | 651 | 241 | 417 | 412* | 415* | 423* |
| 32K | 340 | 647 | 303 | 407 | 400* | 376* | 397* |
| 64K | 333 | 631* | 296 | 403* | 421* | 402* | — |
| 128K | **320** | 603 | 325 | 403 | 379* | — | — |

**Prefill** (single-prompt, integrated scout): 8K → **8,145** · 16K → 7,378 · 32K → 7,138 · 64K → 6,644 · 128K → **6,039** tok/s (TTFT 1.0–21.4 s).

`*` = capacity-limited (warmup timed out); `—` = didn't fit the KV cache. **MTP acceptance** over the run: mean **2.31** / median 1.92 / ~44% (222 windows) — in line with estonia (2.4 / 46%), so representative of real content. Power: **2× 300 W = 600 W for inference** (the bench's whole-box panel reads all 4 GPUs on the machine — ignore its 1,200 W / 641 W totals).

<details><summary>Raw bench dashboard (TUI)</summary>

```
── unholyapostolic · llm_decode_bench v0.4.24 · deepseek-v4-flash · TP2 · temp 1.0 · 30s/cell ──
  Aggregate decode tok/s
   ctx\conc │   1 │   2 │   4 │   8 │  16 │  32 │  64
  ─────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────
   0        │ 351 │ 675 │ 310 │ 451 │ 433*│ 437*│ 438*
   16k      │ 342 │ 651 │ 241 │ 417 │ 412*│ 415*│ 423*
   32k      │ 340 │ 647 │ 303 │ 407 │ 400*│ 376*│ 397*
   64k      │ 333 │ 631*│ 296 │ 403*│ 421*│ 402*│   —
   128k     │ 320 │ 603 │ 325 │ 403 │ 379*│   — │   —
  Prefill tok/s   8k→8145  16k→7378  32k→7138  64k→6644  128k→6039
  MTP accept      mean 2.31 / median 1.92 / ~44%  (222 windows, k=3)
  Power           GPU 2,3 @ 300W cap (600W inference)
  Spec            mtp · k=3 · draft_sample_method=probabilistic
```

Full raw bench log (NVIDIA P2P panel · config · prefill · the matrix · power): [benchmarks/llm_decode_bench_tp2_136k.log](benchmarks/llm_decode_bench_tp2_136k.log)
</details>

## What got the ~2× jump (in order of impact)
1. **V2 model runner** (`VLLM_USE_V2_MODEL_RUNNER=1`) — the keystone. nsys profiling showed ~46% of each decode token was **GPU idle waiting on the host** (kernel-launch + piecewise-cudagraph gaps). The V2 runner collapses that per-sub-step host gap. Everything below only pays *because* of this.
2. **MTP k=3** (`--speculative-config.num_speculative_tokens 3`). On the V1 runner, k=3 *lost* (per-sub-step host overhead > speculation gain). On V2 the collapsed gap makes each extra draft sub-step nearly free, so **k=3 pays** — "hide the deeper MTP in the freed host budget." This + (1) is the bulk of the 2×. (k=4 does *not* pay: position-4 accept decays to 3.4%, mean accept-len flat ~2.4 — k=3 is the ceiling for DSV4's single trained MTP head.)
3. **Tuned serve config**: `--max-model-len 136000 --max-num-batched-tokens 12288 --gpu-memory-utilization 0.95 --stream-interval 8`, `custom_ops:["all"]`, cudagraph `FULL_AND_PIECEWISE`.
4. **Native Lightning Indexer** (do *not* set `VLLM_USE_B12X_SPARSE_INDEXER`) — the b12x sparse indexer **corrupts long-context retrieval** (>131K); native is both correct (28–30/30 estonia) *and* faster single-user.
5. **Parser-only reasoning** — the V2 runner rejects reasoning-budget enforcement, so `--reasoning-parser deepseek_v4` + `--default-chat-template-kwargs.thinking true`, and **no `--reasoning-config.*`**.
6. **DeepGEMM PR #324** — the SM120 "complete odd/large next_n / kPadOddN port" — swapped in as a **site-packages wheel, no vLLM recompile** (`deep_gemm 2.5.0+76e93aa`, torch 2.11). Keeps the 365 decode and runs the upstreamed SM120 indexer-scoring kernels; validated **retrieval-preserving** (28/30 byte-identical estonia).

## Build
The published image is ready to pull. [`Dockerfile`](Dockerfile) is the full reproducible recipe: DeepGEMM PR #324 swapped onto the unholy-fusion base (site-packages wheel, no vLLM recompile) **+ the baked runtime env** so a bare `docker run` is already correct. Pinned: `sha256:6fcfe17e…` (`:env-20260531`).

## Repo contents
- [`serve.sh`](serve.sh) — run the image with the full 365-tok/s config (all env + docker-runtime flags + host-prereq notes)
- [`Dockerfile`](Dockerfile) — DeepGEMM PR #324 layer + baked runtime env
- [`benchmarks/llm_decode_bench_tp2_136k.log`](benchmarks/llm_decode_bench_tp2_136k.log) — full raw `llm_decode_bench` v0.4.24 TUI (TP2 / 136K run)
- [`VERSION_READOUT.md`](VERSION_READOUT.md) — exact pinned version + config manifest (vLLM / torch / b12x / deep_gemm / CUDA, host prereqs, model config, results)
- `README.md` — this file

## Known quirk (investigated — benign, concurrency-only)
2/30 estonia shots over-reason to the 40,000-token cap → 28/30 instead of 30/30. This is a **concurrency-30 stress artifact, not a single-user or retrieval problem**: estonia runs the *same* prompt 30× at once, and batch-composition numeric noise occasionally tips 2 of the 30 runs into a reasoning-vacillation loop that never closes `</think>` (the retrieval itself is correct — it's a stopping/reasoning-length quirk, not a wrong answer). Investigated thoroughly: no clean serving-level fix reaches 30/30 without side-effects (reasoning-budget cap, EOS handling, repetition penalty, and lower MTP all failed; best partial 29/30 via `MAX_NUM_SEQS=30`). **At single-user it doesn't manifest**, so it doesn't affect the headline use case.
