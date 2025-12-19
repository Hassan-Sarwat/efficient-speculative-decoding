# Efficient Speculative Decoding (Distillation & Training)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-ready-green)

This project implements a **Speculative Decoding** training pipeline. It distills knowledge from a large "Teacher" model (Qwen 2.5 14B) into a smaller, faster "Draft" model (Qwen 2.5 0.5B). This technique significantly accelerates inference while maintaining high response quality.

Designed for specialized deployment on **RunPod**, leveraging **Docker-in-Docker** to solve PyTorch version conflicts between training (Unsloth) and serving (vLLM) environments.

## 🏗 Architecture

The system is composed of two Dockerized services orchestrated by `docker-compose`:

1.  **Trainer (`service: trainer`)**:
    *   **Base**: Unsloth (PyTorch 2.4/2.6)
    *   **Role**: Fine-tunes both the Target (14B) and Draft (0.5B) models.
    *   **Output**: Saves distilled adapters to `/app/models`.
2.  **Server (`service: server`)**:
    *   **Base**: vLLM (PyTorch 2.5.1)
    *   **Role**: Loads the trained models and runs high-performance benchmarks.

## 🚀 RunPod Deployment Guide

This guide assumes you are starting from a fresh RunPod GPU instance.

### 1. Rent a GPU
*   **Template**: Select **"RunPod Pytorch 2.4 General Cuda 12.1"** (or any modern Ubuntu + Nvidia Docker template).
*   **GPU**: A single **RTX 3090 / 4090** (24GB VRAM) is sufficient for 4-bit training.
*   **Disk**: >= 100GB (Models are heavy!).

### 2. Setup Environment
Connect via SSH or Web Terminal and run:

```bash
# 1. Update APT and install Docker Compose (if missing)
apt-get update && apt-get install -y docker-compose

# 2. Clone the Repository
git clone https://github.com/Hassan-Sarwat/efficient-speculative-decoding.git
cd efficient-speculative-decoding

# 3. Configure Secrets
# Create a .env file for your WandB Key (do NOT commit this!)
echo "WANDB_API_KEY=vb..." > .env
```

### 3. Start Training
Launch the entire pipeline. This will build the Docker images and run the training script.

```bash
# Build and Run
docker-compose up trainer
```

> **Note**: The first run will take time as it downloads the base models (Qwen 14B/0.5B).

### 4. Verify & Benchmark
Once usage drops (training finished), the `server` service can be started to benchmark the results.

```bash
docker-compose up server
```

## 📂 Project Structure

```plaintext
.
├── configs/               # YAML Configurations for models
│   ├── target_14b.yaml    # Teacher Config
│   └── draft_0-5b.yaml    # Student Config
├── scripts/
│   └── train_pipeline.sh  # Main orchestration script
├── train.py               # Universal Fine-tuning script (Unsloth)
├── benchmark.py           # Speculative Decoding Benchmark (vLLM)
├── Dockerfile.train       # Training Environment
├── Dockerfile.serve       # Inference Environment
└── docker-compose.yml     # Service Orchestration
```

## 🛠 Local Development

To run locally (requires NVIDIA GPU + Linux/WSL2):

1.  Ensure you have **NVIDIA Container Toolkit** installed.
2.  Run `git config core.autocrlf input` to prevent Windows line-ending issues.
3.  Execute `docker-compose up trainer`.
