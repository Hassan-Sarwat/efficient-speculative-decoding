# Chain of Thought (CoT) & Chain of Draft (CoD) Data Generation

This folder contains scripts to generate data using the `gemini-3.0-preview-pro` model on Vertex AI.
The pipeline generates two datasets simultaneously:
1.  **Chain of Thought (CoT)**: Contains the full, high-level reasoning (`<thought>...</thought>`). Saved as `cot_*.jsonl`.
2.  **Chain of Draft (CoD)**: Contains concise, summarized reasoning (`<draft>...</draft>`). Saved as `cod_*.jsonl`.

## Difficulty Pipelines
We generate **1000 samples** for each difficulty level, resulting in 6000 total samples (1k CoT + 1k CoD per difficulty).

### 1. Easy (GSM8K)
Uses the `gsm8k` dataset. No complex filters needed as the dataset itself is the target.
```bash
python data_generation/launch_generation.py \
  --dataset gsm8k \
  --file_suffix easy \
  --limit 1000
```

### 2. Medium (Competition Math L1/L2)
Uses `qwedsacf/competition_math`, filtered to lower levels of Algebra/Precalc.
```bash
python data_generation/launch_generation.py \
  --filter "level=Level 1,Level 2" \
  --filter "type=Algebra,Intermediate Algebra,Precalculus" \
  --file_suffix medium \
  --limit 1000
```

### 3. Hard (Competition Math L3/L4)
Uses `qwedsacf/competition_math`, filtered to higher levels.
```bash
python data_generation/launch_generation.py \
  --filter "level=Level 3,Level 4" \
  --filter "type=Algebra,Intermediate Algebra,Precalculus" \
  --file_suffix hard \
  --limit 1000
```

---

## Pipeline Explained

Here is the step-by-step process of how data flows from the dataset to the final JSONL files.

### Step 1: Launch Generation (`launch_generation.py`)
**Goal**: Select questions and send them to Google's Batch API.

1.  **Launch**: Run `launch_generation.py` with your desired dataset and filters.
2.  **State File**: This creates a state file in `tmp/` (e.g., `tmp/batch_state_gsm8k_easy.json`). This file tracks your batches for this specific suffix.

### Step 2: process Results (`process_results.py`)
**Goal**: Check status, download, and summarize.

**Important**: You must run this script with the **same arguments** (dataset and suffix) used in launch.

**Usage:**
If you launched with `--dataset gsm8k --file_suffix easy`:
```bash
python data_generation/process_results.py --dataset gsm8k --file_suffix easy
```

**What it does:**
1.  **Check Status**: It tracks the progress of batches found in the specific state file (e.g., `batch_state_gsm8k_easy.json`).
2.  **Download & Save CoT**: Saves full reasoning to `data/cot_{suffix}.jsonl`.
3.  **Summarize for CoD**: It **automatically** submits a new batch job (marked as type 'summarization') to summarize the full reasoning.
4.  **Save CoD**: You run `process_results.py` again later. When the summarization job is done, it saves summaries to `data/cod_{suffix}.jsonl`.

## CLI Arguments

### `launch_generation.py`
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | HuggingFace dataset name | `qwedsacf/competition_math` |
| `--limit` | Number of NEW samples to generate | `1000` |
| `--filter` | Filter dataset (only for comp_math). | None |
| `--file_suffix` | Output suffix (`cot_{suffix}.jsonl`) | None |
| `--dry-run` | Prepare batch file but do not submit | `False` |

### `process_results.py`
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | **REQUIRED**: Must match the launch argument. | `qwedsacf/competition_math` |
| `--file_suffix` | **REQUIRED**: Must match the launch argument. | None |
| `--output_dir` | Directory to save final output | `data` |
| `--temp_dir` | Directory where state files are stored | `tmp` |
