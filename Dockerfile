# Use Unsloth's optimized CUDA 12.1 image (PyTorch 2.2)
FROM unslothai/unsloth:cu121-torch220

# Set working directory
WORKDIR /app

# Install vLLM and other dependencies
# Note: We pin vLLM to ensure compatibility with the CUDA version
RUN pip install --no-cache-dir \
    "vllm>=0.4.0" \
    "datasets" \
    "accelerate" \
    "wandb" \
    "scipy"

# Create output directories for models
RUN mkdir -p /app/models/target /app/models/draft /app/data

# Copy your python scripts into the container
COPY train_target.py /app/
COPY train_draft.py /app/
COPY benchmark.py /app/

# Default command (can be overridden)
CMD ["bash"]