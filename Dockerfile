# Conversion dependencies come directly from the official NeMo image.
# CSCS uses a copy of this image; no derived image build is required.
ARG BASE_IMAGE=nvcr.io/nvidia/nemo:26.08.00@sha256:ac012c8d5b7b72fe60ca53e2519175fa8c27966b2d2f8efe6d1ed0559aff4ba0
FROM ${BASE_IMAGE}

WORKDIR /workspace

# NeMo provides Transformers 5.12.1 and tokenizers 0.22.2. Keep its package
# stack unchanged. The repository launchers export the checkpoint-compatible
# Megatron source through PYTHONPATH and reject fallback to bundled MCore 0.19.
