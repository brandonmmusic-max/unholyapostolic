# unholyapostolic — DeepSeek-V4-Flash serving stack for consumer Blackwell (sm_120, TP2)
#
# The published image (verdictai/unholyapostolic / ghcr.io/brandonmmusic-max/unholyapostolic)
# is complete — you normally just `docker pull` it and run serve.sh.
#
# This Dockerfile documents the *reproducible delta*: DeepGEMM PR #324 swapped onto the
# "unholy-fusion" DSV4-Flash base (vLLM apostolic-purification fork + b12x 0.15.3 + torch 2.11
# + flashinfer, built for sm_120). deep_gemm is a site-packages wheel, so swapping #324 needs
# NO vLLM recompile.

ARG BASE_IMAGE=unholy-fusion-base:latest    # the vLLM + b12x sm_120 base (the Lucifer-v2runner stack)
FROM ${BASE_IMAGE}

# DeepGEMM PR #324 — SM120 "complete odd/large next_n (kPadOddN) port for paged MQA logits"
# (deepseek-ai/DeepGEMM refs/pull/324/head). Built against the image's torch 2.11.
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends git cuda-nvrtc-dev-13-0 || true; \
    tmp="$(mktemp -d)"; \
    git clone --recursive --shallow-submodules https://github.com/deepseek-ai/DeepGEMM.git "$tmp/deepgemm"; \
    cd "$tmp/deepgemm"; \
    git fetch origin refs/pull/324/head && git checkout FETCH_HEAD; \
    pip uninstall -y deep_gemm 2>/dev/null || true; \
    python3 setup.py bdist_wheel; \
    pip install dist/*.whl; \
    python3 -c "import deep_gemm; print('deep_gemm', deep_gemm.__version__)"; \
    rm -rf "$tmp"

# Run with serve.sh (the 365-tok/s config: V2 runner + MTP k=3 + native indexer + parser-only reasoning).
