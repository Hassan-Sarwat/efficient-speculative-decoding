# Efficient Speculative Decoding (Distillation & Training)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-wip-orange)

This project implements a **Speculative Decoding** training pipeline. It distills knowledge from a large "Teacher" model (Qwen 2.5 14B) into a smaller, faster "Draft" model (Qwen 2.5 0.5B). This technique significantly accelerates inference while maintaining high response quality.

## 🚀 Deployment Guide (Miniforge)

This project uses **Miniforge** to manage environments, ensuring a conflict-free setup for both Unsloth (training) and vLLM (serving).

### 1. Hardware Requirements
*   **Recommended**: A single **RTX 3090 / 4090** (24GB VRAM) is sufficient for this pipeline.
*   **RunPod Note**: Our experiments are typically performed on **A40s** as they are often more cost-effective on RunPod.
*   **VRAM**: Please note that the total VRAM required will be mentioned at the end of the experiment; expect an update if you verify usage manually.
*   **Disk**: >= 100GB.

### 2. Setup Environment
Connect via SSH/Terminal and run:

```bash
# 1. Clone the Repository
git clone https://github.com/Hassan-Sarwat/efficient-speculative-decoding.git
cd efficient-speculative-decoding

# 2. Configure Secrets
echo "WANDB_API_KEY=vb..." > .env

# 3. Setup Environments
# This script installs Miniforge (if missing) and sets up 'env_train' and 'env_serve'
bash setup_envs.sh
```

### 3. Start Training
The training pipeline uses Unsloth (installed automatically by the setup script). Unsloth handles its own PyTorch dependencies.

```bash
# Activates 'env_train' automatically
bash scripts/train_native.sh
```

### 4. Verify & Benchmark
To run the benchmarks/serving:

```bash
# Activates 'env_serve' automatically
bash scripts/serve_native.sh
```

> [!NOTE]
> **Docker Alternative**: This repository contains Docker configurations (`Dockerfile.train`, `Dockerfile.serve`, `docker-compose.yml`) for those who prefer containerized deployments. However, these are currently **Work In Progress (WIP)** and the images may not be fully functional. Use the Miniforge approach above for the most stable experience.

## 📂 Project Structure

```plaintext
.
├── configs/               # YAML Configurations for models
│   ├── target_14b.yaml    # Teacher Config
│   └── draft_0-5b.yaml    # Student Config
├── scripts/
│   ├── setup_envs.sh      # Environment setup (Miniforge)
│   ├── train_native.sh    # Training runner
│   └── serve_native.sh    # Inference/Benchmark runner
├── train.py               # Universal Fine-tuning script (Unsloth)
├── benchmark.py           # Speculative Decoding Benchmark (vLLM)
├── Dockerfile.train       # (WIP) Training Image
├── Dockerfile.serve       # (WIP) Inference Image
└── docker-compose.yml     # (WIP) Orchestration
```
