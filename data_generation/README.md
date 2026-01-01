# Chain of Thought (CoT) & Chain of Draft (CoD) Data Generation

This folder contains scripts to generate data using the `gemini-3-pro-preview` model on Vertex AI (Google Batch API).

The pipeline generates two datasets simultaneously:
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

We generally target **1000 samples** per run.

### 1. Easy Scenario (GSM8K)
For the GSM8K dataset, no complex filtering is required.

**Launch Generation:**
```bash
python data_generation/launch_generation.py \
  --dataset "gsm8k" \
  --file_suffix "easy" \
  --limit 1000
```
> **Note:** If you already have existing samples (e.g. 900), the script will ask if you want to **Fill** the gap (generate 100 more) or **Extend** (generate 1000 new ones). use `--auto_fill` or `--auto_extend` to skip the prompt.

**Launch Generation Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Hugging Face dataset name | `None` |
| `--filter` | Filter string (e.g., `level=Level 1,Level 2`) | `None` |
| `--file_suffix` | Output suffix (`cot_{suffix}.jsonl`) | `None` |
| `--limit` | Max number of samples to process | `None` |
| `--dry-run` | Prepare batch file but do not submit | `False` |
| `--auto_fill` | Auto-select "Fill Gap" if existing < limit | `False` |
| `--auto_extend` | Auto-select "Extend" if existing data found | `False` |

**Process Results (Download & Summarize):**
```bash
python data_generation/process_results.py \
  --dataset "gsm8k" \
  --file_suffix "easy"
```

### 2. Medium Scenario (Competition Math)
For "Medium" difficulty, we filter the `qwedsacf/competition_math` dataset for Algebra and Precalculus (Levels 1-3).

**Launch Generation:**
```bash
python data_generation/launch_generation.py \
  --dataset "qwedsacf/competition_math" \
  --filter "level=Level 1,Level 2,Level 3" \
  --filter "type=Algebra,Intermediate Algebra,Precalculus" \
  --file_suffix "medium" \
  --limit 1000
```

**Process Results:**
```bash
python data_generation/process_results.py \
  --dataset "qwedsacf/competition_math" \
  --file_suffix "medium"
```

### 3. Hard Scenario (Competition Math)
For "Hard" difficulty, we filter for higher levels (Level 4, 5).

**Launch Generation:**
```bash
python data_generation/launch_generation.py \
  --dataset "qwedsacf/competition_math" \
  --filter "level=Level 4,Level 5" \
  --filter "type=Algebra,Intermediate Algebra,Precalculus,Number Theory" \
  --file_suffix "hard" \
  --limit 1000
```

**Process Results:**
```bash
python data_generation/process_results.py \
  --dataset "qwedsacf/competition_math" \
  --file_suffix "hard"
```

---

## Technical Details

### Scripts Structure
- **`launch_generation.py`**: Handles dataset loading, filtering, and submitting Batch API jobs. Creates a local state file in `tmp/`.
- **`process_results.py`**: Checks batch status, downloads results, tracks metrics (token usage/cost), and orchestrates the summarization step.
- **`result_processor.py`**: Contains the core logic for parsing JSON results, updating metrics, and formatting data samples.

### Workflow
1.  **Launch**: Submits a `generation` batch job.
2.  **Process (Pass 1)**: 
    - Downloads `generation` results.
    - Saves **CoT** samples locally.
    - Identifies samples needing summarization and automatically submits a `summarization` batch job.
3.  **Process (Pass 2)**:
    - Downloads `summarization` results.
    - Saves **CoD** samples locally.

### Cost Estimation
The `process_results.py` script will output a cost estimate based on token usage at the end of execution.
