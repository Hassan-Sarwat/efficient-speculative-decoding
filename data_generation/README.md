# Chain of Thought (CoT) & Chain of Draft (CoD) Data Generation

This folder contains scripts to generate data using the `gemini-3-pro-preview` model on Vertex AI (Google Batch API).

The pipeline generates two datasets independently:
1.  **Chain of Thought (CoT)**: Contains the full, high-level reasoning (`<thought>...</thought>`). Saved as `cot_*.jsonl`.
2.  **Chain of Draft (CoD)**: Contains concise, summarized reasoning (`<draft>...</draft>`). Saved as `cod_*.jsonl`.

## Prerequisites

Ensure you have the following installed:
```bash
pip install google-genai==1.56.0 datasets tqdm python-dotenv
```

Set your API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Running the Pipeline

We generally target **1000 samples** per run. The `--chain` argument determines whether you are generating CoT (`thought`) or CoD (`draft`).

### 1. Easy Scenario (GSM8K)
For the GSM8K dataset, no complex filtering is required.

**Generate Chain of Thought (CoT):**
```bash
python data_generation/launch_generation.py \
  --chain thought \
  --dataset "gsm8k" \
  --file_suffix "easy" \
  --limit 1000
```

**Generate Chain of Draft (CoD):**
```bash
python data_generation/launch_generation.py \
  --chain draft \
  --dataset "gsm8k" \
  --file_suffix "easy" \
  --limit 1000
```
> **Note:** Defaults to `thought` if `--chain` is omitted.

**Launch Generation Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--chain` | Type of chain: `thought` or `draft` | `thought` |
| `--dataset` | Hugging Face dataset name | `qwedsacf/competition_math` |
| `--filter` | Filter string (e.g., `level=Level 1,Level 2`) | `None` |
| `--file_suffix` | Output suffix (`cot_{suffix}.jsonl`) | `None` |
| `--limit` | Max number of samples to process | `1000` |
| `--dry-run` | Prepare batch file but do not submit | `False` |
| `--auto_fill` | Auto-select "Fill Gap" if existing < limit | `False` |
| `--auto_extend` | Auto-select "Extend" if existing data found | `False` |

**Process Results (Check Status & Download):**
```bash
python data_generation/process_results.py \
  --chain thought \
  --dataset "gsm8k" \
  --file_suffix "easy"
```
Or for draft:
```bash
python data_generation/process_results.py \
  --chain draft \
  --dataset "gsm8k" \
  --file_suffix "easy"
```

### 2. Medium Scenario (Competition Math)
For "Medium" difficulty, we filter the `qwedsacf/competition_math` dataset for Algebra and Precalculus (Levels 1-3).

**Generate CoT:**
```bash
python data_generation/launch_generation.py \
  --chain thought \
  --dataset "qwedsacf/competition_math" \
  --filter "level=Level 1,Level 2,Level 3" \
  --filter "type=Algebra,Intermediate Algebra,Precalculus" \
  --file_suffix "medium" \
  --limit 1000
```

**Generate CoD:**
```bash
python data_generation/launch_generation.py \
  --chain draft \
  --dataset "qwedsacf/competition_math" \
  --filter "level=Level 1,Level 2,Level 3" \
  --filter "type=Algebra,Intermediate Algebra,Precalculus" \
  --file_suffix "medium" \
  --limit 1000
```

### 3. Hard Scenario (Competition Math)
Filter for Level 4, 5.

**Generate CoT:**
```bash
python data_generation/launch_generation.py \
  --chain thought \
  --dataset "qwedsacf/competition_math" \
  --filter "level=Level 4,Level 5" \
  --filter "type=Algebra,Intermediate Algebra,Precalculus,Number Theory" \
  --file_suffix "hard" \
  --limit 1000
```

---

## Technical Details

### Scripts Structure
- **`launch_generation.py`**: Handles dataset loading, filtering, and submitting Batch API jobs. Creates a local state file in `tmp/`.
- **`process_results.py`**: Checks batch status and downloads results. Saving logic handled here.
- **`result_processor.py`**: Contains helper logic for parsing JSON results.

### Output Files
- **CoT**: `data/cot_{dataset}_{suffix}.jsonl`
- **CoD**: `data/cod_{dataset}_{suffix}.jsonl`

## Data Analysis

We provide a script to verify generated data against the ground truth.

**Usage:**
```bash
python data_generation/analyze_data.py \
  --dataset "qwedsacf/competition_math" \
  --suffix "medium"
```
