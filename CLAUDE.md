# CLAUDE.md

Project-specific guidance for Claude Code working in this repo. Auto-loaded into every conversation.

## Project

Research framework comparing **Chain of Thought (CoT)** vs **Chain of Draft (CoD)** reasoning for speculative-decoding efficiency. Trains a Qwen3-14B target + Qwen3-0.6B draft on math reasoning (GSM8K / MATH) with LoRA, then benchmarks vLLM speculative decoding (target + draft).

## Environment (critical pins)

- **Single unified venv at `env/`** (Python 3.11). Set up via `bash scripts/uv_setup_env.sh`. Holds both Unsloth (training) and vLLM (inference). Don't reintroduce separate train/serve venvs — the unified env is intentional.
- **`vllm==0.19.1` is pinned**. Do **not** bump to 0.20 without resolving the conflict: Unsloth 2026.4.8 hard-pins `torch<2.11.0` and vLLM 0.20 requires `torch==2.11.0`. vLLM 0.19.1 still has the V1 engine, draft-model speculative decoding, and `llm.get_metrics()` — all our code needs.
- Install via PyPI only: `uv pip install -r requirements.txt --index-strategy unsafe-best-match`. Do **not** preinstall `torch` from a non-cu128 index — vLLM 0.19.1's wheel pulls `torch + nvidia-*-cu128` itself.
- Activate before running anything: `source env/bin/activate` (Linux/macOS) or `env/Scripts/activate` (Windows). Pipeline scripts source/deactivate this for you.

## Pipeline entrypoints

All four live in `scripts/`. Type `{cot,cod}` × scenario `{easy,medium,hard}` (easy=GSM8K, medium=MATH lvl 1-2, hard=MATH lvl 3-4).

| Script | Args | Purpose |
|---|---|---|
| `train_pipeline.sh` | `-t {cot\|cod} -s {easy\|medium\|hard}` | Train target → distill synthetic data via vLLM → train draft. Saves LoRA adapters only (no merge). |
| `benchmark_pipeline.sh` | `-t … -s … [-m baseline\|speculative\|both] [-n N] [-k]` | Ephemerally merges adapters → runs benchmark → deletes temp models. `-m` defaults to `both`. `-n` caps sample count. `-k` keeps merged models on disk. |
| `untrained_pipeline.sh` | `<scenario>` | Baseline using untrained base models. |
| `run_queue.sh` | (none) | Runs all 9 jobs (3 scenarios × {cot,cod,untrained}) sequentially. |

## Naming conventions

- LoRA adapters: `models/{target,draft}_{type}_{scenario}/`
- Untrained baseline: `models/draft_untrained_{scenario}/`
- Ephemeral merged models: `models/temp_merged_{target,draft}_{type}_{scenario}/`
- Training data: `data/processed/{type}_{scenario}.jsonl`
- Distilled data: `data/distilled/{type}_{scenario}.jsonl`
- Test data: `data/tests/{scenario}_test.jsonl`
- Benchmark outputs: `outputs/{run_name}_{scenario}.csv`, `outputs/metrics_{run_name}.json`, `outputs/comparison_{run_name}.txt`

## Models & configs

- Target base: `Qwen/Qwen3-14B`
- Draft base: `Qwen/Qwen3-0.6B` (note: 0.6B, not 0.5B)
- Configs: [configs/target_14b.yaml](configs/target_14b.yaml), [configs/draft_0-6b.yaml](configs/draft_0-6b.yaml).
- [src/train.py](src/train.py) accepts a YAML config and supports CLI overrides for `--data_file`, `--final_save_path`, `--output_dir`, `--run_name`, `--wandb_project`.
- W&B project hardcoded as `peft_cob`.

## Secrets / .env

Both training and data-generation scripts read `.env`:
- `WANDB_API_KEY` — used by [src/train.py](src/train.py) for logging to the `peft_cob` project.
- `GEMINI_API_KEY` — used by [data_generation/launch_generation.py](data_generation/launch_generation.py) for Google Gemini Batch API calls.

Copy `.env.example` → `.env` and populate before running.

## Library-API gotchas

These are load-bearing — a fresh session will trip on them:

- **`trl 0.24`**: use `processing_class=` (not `tokenizer=`); `assistant_only_loss=True` in `SFTConfig` replaces the removed `DataCollatorForCompletionOnlyLM`. Chat template is auto-applied to the `messages` column.
- **Unsloth import order**: import `unsloth` **before** `transformers` / `peft` / `trl` — it patches them at import time.
- **vLLM V1 spec-decode metrics**: call `llm.get_metrics()` **before** `del llm`. Offline mode (`LLM.generate()`) does not populate per-request `RequestOutput.metrics` — TTFT is unavailable, ITL is synthesized from wall-time.
- **vLLM speculative_config**: `{"method": "draft_model", "model": draft_path, "num_speculative_tokens": K}` (K=5 in [tests/benchmark.py](tests/benchmark.py)).
- **Tokenizer alignment**: target/draft must share an identical vocab for spec decoding. [tests/benchmark.py](tests/benchmark.py) validates this at startup.
- **Distillation is resumable**: [src/distill_data.py](src/distill_data.py) skips rows already present in the output file. Re-running is safe.

## Answer extraction

[src/answer_utils.py](src/answer_utils.py) is shared by distillation validation and benchmark accuracy. Extraction cascade:

- MATH (medium / hard): `\boxed{}` → `####` → last number
- GSM8K (easy): `####` → last number

A high last-number fallback rate signals format non-compliance from the model.

## Don't

- Don't add backwards-compat shims for removed `trl` / `unsloth` / `vllm` APIs — pin a working version in `requirements.txt` instead.
- Don't reintroduce separate train/serve venvs.
- Don't bump `vllm` past 0.19.1 without addressing the Unsloth `torch<2.11.0` constraint.
- Don't preinstall `torch` from a non-cu128 index before installing requirements.
