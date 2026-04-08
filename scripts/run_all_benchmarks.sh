#!/bin/bash
set -e  # Stop if any pipeline fails

echo "========================================================"
echo "Starting Full Evaluation Suite (9 Scenarios)"
echo "========================================================"
START_TIME=$(date +%s)
START_DATE=$(date)

# Define a function aligned with the formatting of train_pipeline.sh
run_benchmark() {
    local type=$1
    local scenario=$2
    local mode=$3

    echo ""
    echo "=========================================="
    echo "Evaluating: $type - $scenario"
    echo "=========================================="
    local step_start=$(date +%s)

    # Note: For 'base' (Untrained), the script will automatically fallback 
    # to the base model since no adapter is present.
    bash scripts/benchmark_pipeline.sh -t "$type" -s "$scenario" -m "$mode"
    
    local step_end=$(date +%s)
    local step_duration=$((step_end - step_start))
    echo "------------------------------------------"
    echo "Completed $type ($scenario) in ${step_duration}s"
    echo "------------------------------------------"
}

# ---------------------------------------------------------
# 1. Easy Scenarios (MATH Level 1-2 / GSM8k)
# ---------------------------------------------------------
run_benchmark "cot" "easy" "both"
run_benchmark "cod" "easy" "both"
run_benchmark "base" "easy" "baseline"

# ---------------------------------------------------------
# 2. Medium Scenarios (MATH Level 3)
# ---------------------------------------------------------
run_benchmark "cot" "medium" "both"
run_benchmark "cod" "medium" "both"
run_benchmark "base" "medium" "baseline"

# ---------------------------------------------------------
# 3. Hard Scenarios (MATH Level 4-5)
# ---------------------------------------------------------
run_benchmark "cot" "hard" "both"
run_benchmark "cod" "hard" "both"
run_benchmark "base" "hard" "baseline"

# ---------------------------------------------------------
# Final Summary
# ---------------------------------------------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================"
echo "EVALUATION QUEUE COMPLETED SUCCESSFULLY"
echo "========================================================"
echo "Start: $START_DATE"
echo "End:   $(date)"
echo "Total Duration: ${DURATION}s"
echo "All evaluation results & unified metrics saved in outputs/"
echo "========================================================"
