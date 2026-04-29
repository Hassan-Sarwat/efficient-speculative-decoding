# Unified image: Unsloth (training) + vLLM (speculative decoding inference).
# Base on the official vLLM 0.20 image — already has CUDA 13 runtime + torch 2.11
# matched to the vLLM C extension. Layer Unsloth and the rest of the deps on top.
FROM vllm/vllm-openai:v0.19.1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# vllm is already installed in the base image; install everything else.
# --no-deps for vllm prevents reinstalling torch and risking a CUDA mismatch.
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY src/ /app/src/
COPY tests/ /app/tests/

RUN mkdir -p /app/models /app/data /app/outputs && chmod +x /app/scripts/*.sh

# Override the upstream vLLM entrypoint so the container is generic.
ENTRYPOINT []
CMD ["bash"]
