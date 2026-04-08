#!/bin/bash
set -e  # Stop if any pipeline fails

echo "💤 Starting Nightly Training Queue..."
start_time=$(date)

# 1. Easy Scenarios
echo "👉 Running: CoD - Easy"
bash scripts/train_pipeline.sh -t cod -s easy

echo "👉 Running: CoT - Easy"
bash scripts/train_pipeline.sh -t cot -s easy

echo "👉 Running: Untrained - Easy"
bash scripts/untrained_pipeline.sh easy

# 2. Medium Scenarios
echo "👉 Running: CoD - Medium"
bash scripts/train_pipeline.sh -t cod -s medium

echo "👉 Running: CoT - Medium"
bash scripts/train_pipeline.sh -t cot -s medium

echo "👉 Running: Untrained - Medium"
bash scripts/untrained_pipeline.sh medium

# 3. Hard Scenarios
echo "👉 Running: CoD - Hard"
bash scripts/train_pipeline.sh -t cod -s hard

echo "👉 Running: CoT - Hard"
bash scripts/train_pipeline.sh -t cot -s hard

echo "👉 Running: Untrained - Hard"
bash scripts/untrained_pipeline.sh hard

end_time=$(date)
echo "✅ All jobs completed successfully!"
echo "Start: $start_time"
echo "End:   $end_time"