#!/bin/bash
set -e

echo "--- 🛠️  Setting up RunPod Environment ---"

# 1. Install Docker Engine (if missing)
if ! command -v docker &> /dev/null; then
    echo "[1/3] Installing Docker Engine..."
    apt-get update
    apt-get install -y docker.io
else
    echo "[1/3] Docker Engine already installed."
fi

# 2. Install Docker Compose V2 (Standalone)
echo "[2/3] Installing Docker Compose V2..."
curl -SL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 3. Start Docker Daemon (Background)
# RunPod often doesn't have systemd, so we start dockerd manually
echo "[3/3] Starting Docker Daemon..."
if ! pidof dockerd > /dev/null; then
    nohup dockerd > /var/log/dockerd.log 2>&1 &
    echo "    ...Waiting for Docker to initialize..."
    sleep 5
else
    echo "    ...Docker Daemon already running."
fi

echo "--- ✅ Setup Complete! ---"
echo "You can now run: docker-compose up trainer"
