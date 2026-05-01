#!/usr/bin/env python3
"""
Benchmark with Speculative Decoding Metrics Extraction

vLLM 0.11+ (V1 engine only — V0 was removed in 0.11; tested against 0.19.1).
Spec-decode metrics are read via `llm.get_metrics()` which exposes Prometheus
counters:
  vllm:spec_decode_num_drafts
  vllm:spec_decode_num_draft_tokens
  vllm:spec_decode_num_accepted_tokens
  vllm:spec_decode_num_accepted_tokens_per_pos  (Vector, length K)
"""

import time
import sys
import os

from dataclasses import dataclass, asdict
from typing import List, Optional
import gc
import csv
import argparse
import json
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.v1.metrics.reader import Counter as VllmCounter, Vector as VllmVector
from datasets import load_dataset
from transformers import AutoTokenizer
# Add src directory to path so answer_utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from answer_utils import extract_answer, check_equality, classify_extraction_method


def _query_vram_gb(device_index: int = 0) -> float:
    """Query current GPU VRAM usage via pynvml then nvidia-smi.

    torch.cuda.max_memory_allocated() returns 0 when vLLM runs its engine in a
    subprocess (the spawn start method), so we query the driver directly instead.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 3)
    except Exception as e:
        print(f"[VRAM] pynvml failed ({e}), trying nvidia-smi...")

    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
            if lines:
                return float(lines[0]) / 1024  # MiB → GiB
        print(f"[VRAM] nvidia-smi returned non-zero exit ({r.returncode}): {r.stderr.strip()}")
    except Exception as e:
        print(f"[VRAM] nvidia-smi failed ({e}), falling back to torch allocator...")

    return torch.cuda.max_memory_allocated() / (1024 ** 3)


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for blog analysis"""
    # Primary Metrics
    avg_tokens_generated: float
    total_tokens_generated: int
    acceptance_rate: float

    # Supporting Metrics
    accuracy: float
    total_samples: int
    avg_ttft_ms: float
    avg_itl_ms: float
    tokens_per_second: float
    total_duration_sec: float
    peak_vram_gb: float

    # Experiment Config
    experiment_name: str
    use_speculative: bool
    num_speculative_tokens: int

    # Speculative-specific
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_drafts: int = 0
    system_efficiency: float = 0.0

    # Per-position acceptance distribution (length K, one entry per draft position).
    # Reveals whether the draft is uniformly good or degrades after the first few tokens.
    accepts_per_position: Optional[List[int]] = None
    emits_per_position: Optional[List[int]] = None

    # Extraction method breakdown for predictions: {"boxed": N, "separator": N, "last_number": N, "empty": N}
    extraction_method_counts: Optional[dict] = None

    def to_dict(self):
        return asdict(self)

    def print_summary(self):
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Avg Tokens: {self.avg_tokens_generated:.1f}")
        print(f"Total Tokens: {self.total_tokens_generated}")
        if self.use_speculative and self.acceptance_rate > 0:
            print(f"Acceptance Rate: {self.acceptance_rate:.2%}")
            print(f"   Draft Tokens: {self.num_draft_tokens}")
            print(f"   Accepted Tokens: {self.num_accepted_tokens}")
            # In V1 this field stores mean acceptance length (1 + accepted/drafts),
            # i.e. average tokens emitted per draft round. >1 is a speedup.
            print(f"   Mean accept length: {self.system_efficiency:.3f}")
            if self.accepts_per_position:
                accepts = self.accepts_per_position
                denom = self.num_drafts if self.num_drafts > 0 else None
                if accepts and denom:
                    print(f"   Per-position acceptance (rate = accepts[i] / num_drafts):")
                    for i, count in enumerate(accepts):
                        rate = count / denom
                        bar = "#" * int(rate * 30)
                        print(f"      pos {i}: {count:7d} ({rate:6.2%}) {bar}")
        elif self.use_speculative:
            print(f"Acceptance Rate: Not captured (check logs)")
        print(f"Accuracy: {self.accuracy:.2%}")
        print(f"Tokens/sec: {self.tokens_per_second:.1f}")
        print(f"ITL: {self.avg_itl_ms:.2f}ms/token")
        print(f"Peak VRAM: {self.peak_vram_gb:.2f}GB")
        if self.extraction_method_counts:
            total = sum(self.extraction_method_counts.values())
            print(f"Answer Extraction ({total} samples):")
            for method in ("boxed", "separator", "last_number", "empty"):
                count = self.extraction_method_counts.get(method, 0)
                bar = "#" * int(count / max(total, 1) * 20)
                print(f"   {method:12s}: {count:4d} ({count/total:5.1%}) {bar}")
            fallback = self.extraction_method_counts.get("last_number", 0)
            if fallback / total > 0.2:
                print(f"   [WARNING] High last_number fallback ({fallback/total:.0%}) — model may not follow expected format")
        print("="*60 + "\n")


# =============================================================================
# Speculative Decoding Metrics Extraction (vLLM V1)
# =============================================================================
#
# In V1, vLLM exposes spec-decode counters via `llm.get_metrics()`, which
# returns a list of metric objects with a Prometheus-style schema. The relevant
# names (no "_total" suffix at the metric level — that's added per-sample by
# prometheus_client):
#
#   vllm:spec_decode_num_drafts                   Counter
#   vllm:spec_decode_num_draft_tokens             Counter
#   vllm:spec_decode_num_accepted_tokens          Counter
#   vllm:spec_decode_num_accepted_tokens_per_pos  Vector (length K)
#
# `llm.get_metrics()` MUST be called while the engine is alive — i.e. BEFORE
# `del llm`. The metric objects are typed (Counter/Vector); we read `.value`
# for scalars and `.values` for vectors.

def extract_spec_metrics_v1(llm: LLM, num_spec_tokens: int) -> dict | None:
    """
    Read spec-decode counters via vLLM V1's `llm.get_metrics()`.

    Uses the official API pattern from vllm/examples/offline_inference/spec_decode.py:
    iterate all metrics, accumulate with += (handles multi-engine DP), access
    Counter.value and Vector.values directly.

    MUST be called BEFORE `del llm`. Returns None if no spec-decode metrics
    are present (e.g. baseline run with `use_speculative=False`).
    """
    try:
        metrics = llm.get_metrics()
    except AttributeError:
        print("[METRICS] llm.get_metrics() not available — vLLM version may be too old")
        return None
    except Exception as e:
        print(f"[METRICS] llm.get_metrics() raised: {e}")
        return None

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepts_per_pos = [0] * num_spec_tokens
    found_any = False

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts" and isinstance(metric, VllmCounter):
            num_drafts += metric.value
            found_any = True
        elif metric.name == "vllm:spec_decode_num_draft_tokens" and isinstance(metric, VllmCounter):
            num_draft_tokens += metric.value
            found_any = True
        elif metric.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(metric, VllmCounter):
            num_accepted_tokens += metric.value
            found_any = True
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and isinstance(metric, VllmVector):
            for pos in range(min(len(metric.values), num_spec_tokens)):
                accepts_per_pos[pos] += metric.values[pos]
            found_any = True

    if not found_any:
        return None

    if num_draft_tokens <= 0:
        return {
            "acceptance_rate": 0.0,
            "system_efficiency": 1.0,
            "num_accepted_tokens": num_accepted_tokens,
            "num_draft_tokens": num_draft_tokens,
            "num_drafts": num_drafts,
            "accepts_per_pos": accepts_per_pos,
        }

    acceptance_rate = num_accepted_tokens / num_draft_tokens
    mean_accept_len = 1.0 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0

    print(f"\n[METRICS] Extracted from V1 llm.get_metrics():")
    print(f"   Drafts:          {num_drafts}")
    print(f"   Draft tokens:    {num_draft_tokens}")
    print(f"   Accepted tokens: {num_accepted_tokens}")
    print(f"   Acceptance rate: {acceptance_rate:.4f}")
    print(f"   Mean accept len: {mean_accept_len:.3f}")

    return {
        "acceptance_rate": acceptance_rate,
        "system_efficiency": mean_accept_len,
        "num_accepted_tokens": num_accepted_tokens,
        "num_draft_tokens": num_draft_tokens,
        "num_drafts": num_drafts,
        "accepts_per_pos": accepts_per_pos,
    }


def dump_metrics_diagnostic(llm: LLM):
    """Diagnostic: print all spec_decode-related metrics from llm.get_metrics()."""
    try:
        metrics = llm.get_metrics()
    except Exception as e:
        print(f"[DIAGNOSTIC] llm.get_metrics() failed: {e}")
        return
    print("\n[DIAGNOSTIC] llm.get_metrics() spec_decode entries:")
    found = False
    for m in metrics:
        name = getattr(m, "name", "<noname>")
        if "spec_decode" in name:
            found = True
            attrs = {a: getattr(m, a, None) for a in ("value", "values", "count")
                     if hasattr(m, a)}
            print(f"   {name!r}  type={type(m).__name__}  attrs={attrs}")
    if not found:
        print("   (no spec_decode metrics present)")


def run_benchmark_pass(name, data, stop_tokens, tokenizer, scenario, use_speculative=False,
                      target_base="Qwen/Qwen3-14B", target_adapter=None,
                      draft_base="Qwen/Qwen3-0.6B", draft_adapter=None,
                      csv_writer=None, run_id="", enable_lora=False):
    """Run a single benchmark pass with metric capture"""

    print(f"\n{'='*40}")
    print(f"RUNNING BENCHMARK: {name}")
    print(f"{'='*40}")

    torch.cuda.reset_peak_memory_stats()

    os.makedirs("outputs", exist_ok=True)

    # 1. Determine model paths
    if target_adapter and os.path.exists(target_adapter):
        target_model_path = target_base
        print(f"Target: Using adapter {target_adapter} with base {target_base}")
    else:
        target_model_path = target_base
        target_adapter = None
        print(f"Target: Using merged model {target_base}")

    # 2. Prepare Draft Model
    speculative_model_path = None
    if use_speculative:
        if draft_adapter:
            if not os.path.exists(draft_adapter):
                print(f"ERROR: Draft model not found at {draft_adapter}")
                return None
            if not os.path.exists(os.path.join(draft_adapter, "config.json")):
                print(f"ERROR: Invalid draft model at {draft_adapter}")
                return None

            speculative_model_path = draft_adapter
            print(f"Using pre-merged draft model: {speculative_model_path}")
        else:
            speculative_model_path = draft_base
            print(f"Using base draft model: {speculative_model_path}")

    # 3. Configure vLLM
    llm_kwargs = {
        "model": target_model_path,
        "enable_lora": bool(target_adapter),
        "max_lora_rank": 64 if target_adapter else None,
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "max_num_seqs": 16,
        # V1 supports prefix caching with spec decode (V0 required it disabled).
        "enable_prefix_caching": True,
        "seed": 42,
        # Offline LLM disables stat logging by default, which makes
        # llm.get_metrics() raise "Stat logging disabled". Must be False
        # to capture spec-decode acceptance counters.
        "disable_log_stats": False,
    }

    llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

    # 4. Add speculative config (V1 schema requires "method")
    num_spec_tokens = 0
    if use_speculative and speculative_model_path:
        print(f"Speculative Decoding: ENABLED")
        print(f"Draft Model: {speculative_model_path}")

        num_spec_tokens = 5
        llm_kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": speculative_model_path,
            "num_speculative_tokens": num_spec_tokens,
        }
    else:
        print(f"Speculative Decoding: DISABLED (Baseline)")

    # 5. Initialize vLLM
    print(f"\nInitializing vLLM...")
    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Verify actual K value (vLLM may silently override).
    # V1 surfaces this on vllm_config.speculative_config; attribute name has
    # historically been `num_spec_tokens` and/or `num_speculative_tokens`.
    if use_speculative:
        try:
            sc = llm.llm_engine.vllm_config.speculative_config
            actual_k = None
            if sc is not None:
                actual_k = getattr(sc, "num_spec_tokens", None)
                if actual_k is None:
                    actual_k = getattr(sc, "num_speculative_tokens", None)
            if actual_k is not None and actual_k != num_spec_tokens:
                print(f"[WARNING] Requested K={num_spec_tokens}, vLLM using K={actual_k}")
                num_spec_tokens = actual_k
            elif actual_k is not None:
                print(f"[OK] Speculative tokens K={actual_k} confirmed")
        except AttributeError:
            pass

    # 6. Attach target adapter
    target_lora_request = None
    if target_adapter:
        print(f"Attaching Target LoRA Adapter: {target_adapter}")
        target_lora_request = LoRARequest("target_adapter", 1, target_adapter)

    # 7. Configure sampling
    params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        stop=stop_tokens
    )

    # 8. Format prompts
    print("Formatting Prompts...")
    prompts = []
    question_key = "question" if "question" in data.column_names else "instruction"
    answer_key = "answer" if "answer" in data.column_names else "output"

    if question_key not in data.column_names:
        raise ValueError(f"Dataset must contain '{question_key}' column")

    _FORMAT_SYSTEM_MESSAGE = (
        "Solve the problem step by step. "
        "Conclude your response with '####' followed by the final answer on the same line, "
        "for example: #### 42"
    )
    for q in data[question_key]:
        messages = [
            {"role": "system", "content": _FORMAT_SYSTEM_MESSAGE},
            {"role": "user", "content": q},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    # 9. Run generation
    print(f"Starting Generation on {len(prompts)} samples...")

    start_time = time.time()
    outputs = llm.generate(prompts, params, lora_request=target_lora_request)
    end_time = time.time()

    duration = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / duration

    max_vram = _query_vram_gb()

    # =========================================================================
    # 10. Extract spec metrics BEFORE del llm (engine must be alive)
    # =========================================================================
    acceptance_rate = 0.0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    num_drafts_total = 0
    system_efficiency = 0.0
    accepts_per_position = None
    emits_per_position = None

    if use_speculative:
        spec_metrics_dict = extract_spec_metrics_v1(llm, num_spec_tokens)

        if spec_metrics_dict is not None:
            acceptance_rate = spec_metrics_dict["acceptance_rate"]
            num_draft_tokens = spec_metrics_dict["num_draft_tokens"]
            num_accepted_tokens = spec_metrics_dict["num_accepted_tokens"]
            num_drafts_total = spec_metrics_dict["num_drafts"]
            system_efficiency = spec_metrics_dict["system_efficiency"]
            accepts_per_position = spec_metrics_dict.get("accepts_per_pos")
            if accepts_per_position:
                print(f"[METRICS] Per-position accepts: {accepts_per_position}")
        else:
            print("[WARNING] Could not extract spec decode metrics via llm.get_metrics()")
            dump_metrics_diagnostic(llm)

        # V1 does not expose a separate "emits_per_position" counter; per-position
        # acceptance from accepts_per_pos is the canonical view.
        emits_per_position = None

    # 11. Calculate latency metrics
    #
    # vLLM V1's offline `LLM.generate()` does NOT populate per-request
    # `RequestOutput.metrics` (it's None — see vllm-project/vllm#26298, "closed
    # as not planned"). Per-request TTFT/ITL is therefore unavailable in
    # offline mode; we fall back to a batch-aggregate ITL estimate computed
    # from total wall time and total generated tokens. TTFT cannot be
    # estimated this way and is reported as 0.
    print("Generation Complete!")
    ttft_list: list[float] = []
    itl_list: list[float] = []
    metrics_available = 0
    metrics_missing = 0

    for o in outputs:
        m = getattr(o, "metrics", None)

        if m is None:
            metrics_missing += 1
            continue

        metrics_available += 1

        if getattr(m, "first_token_time", None) is not None and getattr(m, "arrival_time", None) is not None:
            ttft_ms = (m.first_token_time - m.arrival_time) * 1000
            ttft_list.append(ttft_ms)

        generated_token_count = len(o.outputs[0].token_ids)
        if (
            generated_token_count > 1
            and getattr(m, "finished_time", None) is not None
            and getattr(m, "first_token_time", None) is not None
        ):
            gen_time = m.finished_time - m.first_token_time
            itl_ms = (gen_time / (generated_token_count - 1)) * 1000
            itl_list.append(itl_ms)

    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0
    avg_itl = sum(itl_list) / len(itl_list) if itl_list else 0.0

    if metrics_missing == len(outputs):
        # Expected on V1 offline: synthesize an aggregate ITL from total tokens
        # and wall-clock duration. TTFT stays 0.
        if total_tokens > 0:
            avg_itl = (duration / total_tokens) * 1000
        print(
            f"   Per-request metrics unavailable (V1 offline behaviour); "
            f"using aggregate ITL = duration / total_tokens"
        )
    else:
        if metrics_missing > 0:
            print(f"   Metrics unavailable for {metrics_missing}/{len(outputs)} samples")
        print(f"   TTFT computed for {len(ttft_list)}/{metrics_available} samples")
        print(f"   ITL computed for {len(itl_list)}/{metrics_available} samples")

    print(f"Duration:   {duration:.2f}s")
    print(f"Throughput: {tps:.2f} tokens/s")
    print(f"Avg ITL:    {avg_itl:.2f} ms/token")
    print(f"Peak VRAM:  {max_vram:.2f} GB")

    # 12. Evaluate accuracy
    score = 0
    from collections import Counter
    pred_methods: Counter = Counter()
    gt_methods: Counter = Counter()
    print("\nEVALUATION SAMPLES (First 1):")

    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data[answer_key][i]

        pred_ext = extract_answer(gen_text, scenario)
        gt_ext = extract_answer(ground_truth, scenario)
        pred_methods[classify_extraction_method(gen_text, scenario)] += 1
        gt_methods[classify_extraction_method(ground_truth, scenario)] += 1

        is_correct = check_equality(pred_ext, gt_ext)
        score += 1 if is_correct else 0

        if i < 1:
            print(f"\n--- Sample {i} ---")
            print(f"GT Raw: {ground_truth[-50:]}")
            print(f"GT Ext: {gt_ext}")
            print(f"Pred Ext: {pred_ext}")
            print(f"Correct? {is_correct}")

        if csv_writer:
            csv_writer.writerow([
                run_id, name, data[question_key][i], ground_truth,
                gen_text, gt_ext, pred_ext, is_correct,
                len(output.outputs[0].token_ids)
            ])

    final_acc = (score / len(data)) * 100
    print(f"\nAccuracy: {final_acc}%")

    avg_output_tokens = total_tokens / len(data)
    print(f"Avg Output Length: {avg_output_tokens:.1f} tokens/sample")

    total_eval = len(outputs)
    print(f"\nExtraction methods — Predictions:")
    for method in ("boxed", "separator", "last_number", "empty"):
        count = pred_methods.get(method, 0)
        print(f"   {method:12s}: {count:4d} ({count/total_eval:5.1%})")
    print(f"Extraction methods — Ground Truth:")
    for method in ("boxed", "separator", "last_number", "empty"):
        count = gt_methods.get(method, 0)
        print(f"   {method:12s}: {count:4d} ({count/total_eval:5.1%})")
    if pred_methods.get("last_number", 0) / total_eval > 0.2:
        print(f"[WARNING] >20% of predictions extracted via last_number fallback")

    # 13. Cleanup (AFTER metric extraction — this is critical)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    metrics = BenchmarkMetrics(
        avg_tokens_generated=avg_output_tokens,
        total_tokens_generated=total_tokens,
        acceptance_rate=acceptance_rate,
        accuracy=final_acc / 100,
        total_samples=len(data),
        avg_ttft_ms=avg_ttft,
        avg_itl_ms=avg_itl,
        tokens_per_second=tps,
        total_duration_sec=duration,
        peak_vram_gb=max_vram,
        experiment_name=name,
        use_speculative=use_speculative,
        num_speculative_tokens=num_spec_tokens,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        num_drafts=num_drafts_total,
        system_efficiency=system_efficiency,
        accepts_per_position=accepts_per_position,
        emits_per_position=emits_per_position,
        extraction_method_counts=dict(pred_methods),
    )

    metrics_path = f"outputs/metrics_{name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    metrics.print_summary()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Speculative Decoding Benchmark")
    parser.add_argument("--scenario", type=str, required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--target-base-model", type=str, required=True)
    parser.add_argument("--target-adapter", type=str)
    parser.add_argument("--draft-base-model", type=str)
    parser.add_argument("--merged-draft-model", type=str)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--use-speculative", action="store_true")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--run-both", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None,
                    help="Number of samples to run. Defaults to full dataset.")

    args = parser.parse_args()

    target_base = args.target_base_model
    target_adapter = args.target_adapter
    draft_base = args.draft_base_model or "Qwen/Qwen3-0.6B"
    draft_adapter = args.merged_draft_model

    # Load dataset
    print("Loading Dataset...")
    data = load_dataset("json", data_files=args.data_path, split="train")

    if args.num_samples is not None:
        original_size = len(data)
        data = data.select(range(min(args.num_samples, len(data))))
        print(f"Using {len(data)}/{original_size} samples (--num-samples={args.num_samples})")
    else:
        print(f"Using full dataset: {len(data)} samples")

    print(f"Loading Tokenizer from {target_base}...")
    tokenizer = AutoTokenizer.from_pretrained(target_base, trust_remote_code=True)
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]

    # Speculative decoding requires target and draft to share vocabulary.
    # Fail fast at startup rather than producing silently-wrong acceptance metrics.
    print(f"Verifying target/draft tokenizer alignment ({draft_base})...")
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_base, trust_remote_code=True)
    target_vocab = tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    if target_vocab != draft_vocab:
        only_target = set(target_vocab) - set(draft_vocab)
        only_draft = set(draft_vocab) - set(target_vocab)
        raise RuntimeError(
            f"Target ({target_base}) and draft ({draft_base}) tokenizer vocabularies differ. "
            f"Speculative decoding requires identical vocabularies.\n"
            f"  Target vocab size: {len(target_vocab)}, Draft vocab size: {len(draft_vocab)}\n"
            f"  Tokens only in target: {len(only_target)}, only in draft: {len(only_draft)}"
        )
    print(f"[OK] Vocabularies match ({len(target_vocab)} tokens)")
    del draft_tokenizer

    os.makedirs("outputs", exist_ok=True)
    out_csv_path = f"outputs/{args.run_name}_{args.scenario}.csv"
    print(f"Logging details to {out_csv_path}")

    run_baseline = args.run_both or not args.use_speculative
    run_spec = args.run_both or args.use_speculative

    all_metrics = {}

    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "config_name", "question", "ground_truth", "prediction",
                        "gt_extracted", "pred_extracted", "is_correct", "tokens"])

        if run_baseline:
            print("\n" + "="*60)
            print("RUNNING BASELINE (No Speculative Decoding)")
            print("="*60)

            baseline_name = f"{args.run_name}_baseline"
            baseline_metrics = run_benchmark_pass(
                name=baseline_name,
                data=data,
                stop_tokens=stop_tokens,
                tokenizer=tokenizer,
                scenario=args.scenario,
                use_speculative=False,
                target_base=target_base,
                target_adapter=target_adapter,
                draft_base=draft_base,
                draft_adapter=None,
                csv_writer=writer,
                run_id=baseline_name,
                enable_lora=args.enable_lora
            )

            if baseline_metrics:
                all_metrics['baseline'] = baseline_metrics.to_dict()

        if run_spec:
            print("\n" + "="*60)
            print("RUNNING SPECULATIVE DECODING")
            print("="*60)

            spec_name = f"{args.run_name}_speculative"
            spec_metrics = run_benchmark_pass(
                name=spec_name,
                data=data,
                stop_tokens=stop_tokens,
                tokenizer=tokenizer,
                scenario=args.scenario,
                use_speculative=True,
                target_base=target_base,
                target_adapter=target_adapter,
                draft_base=draft_base,
                draft_adapter=draft_adapter,
                csv_writer=writer,
                run_id=spec_name,
                enable_lora=args.enable_lora
            )

            if spec_metrics:
                all_metrics['speculative'] = spec_metrics.to_dict()

    if all_metrics:
        unified_metrics_path = f"outputs/metrics_{args.run_name}.json"
        with open(unified_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nUnified metrics saved to: {unified_metrics_path}")

        if 'baseline' in all_metrics and 'speculative' in all_metrics:
            baseline = all_metrics['baseline']
            spec = all_metrics['speculative']

            speedup = spec['tokens_per_second'] / baseline['tokens_per_second'] if baseline['tokens_per_second'] > 0 else 0
            latency_reduction = (baseline['total_duration_sec'] - spec['total_duration_sec']) / baseline['total_duration_sec'] * 100
            token_reduction = (baseline['avg_tokens_generated'] - spec['avg_tokens_generated']) / baseline['avg_tokens_generated'] * 100

            print("\n" + "="*60)
            print("COMPARISON: Baseline vs Speculative Decoding")
            print("="*60)
            print(f"Throughput Speedup: {speedup:.2f}x")
            print(f"Latency Reduction: {latency_reduction:.1f}%")
            print(f"Accuracy Delta: {(spec['accuracy'] - baseline['accuracy'])*100:.2f}%")
            print(f"Token Reduction: {token_reduction:.1f}%")

            if spec['acceptance_rate'] > 0:
                print(f"Acceptance Rate: {spec['acceptance_rate']:.2%}")
                print(f"   Draft Tokens: {spec['num_draft_tokens']}")
                print(f"   Accepted Tokens: {spec['num_accepted_tokens']}")
                print(f"   Mean accept length: {spec['system_efficiency']:.3f}")
            else:
                print(f"Acceptance Rate: Not available (check logs)")

            print("="*60)

            comparison_path = f"outputs/comparison_{args.run_name}.txt"
            with open(comparison_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("COMPARISON: Baseline vs Speculative Decoding\n")
                f.write("="*60 + "\n")
                f.write(f"Throughput Speedup: {speedup:.2f}x\n")
                f.write(f"Latency Reduction: {latency_reduction:.1f}%\n")
                f.write(f"Accuracy Delta: {(spec['accuracy'] - baseline['accuracy'])*100:.2f}%\n")
                f.write(f"Token Reduction: {token_reduction:.1f}%\n")
                if spec['acceptance_rate'] > 0:
                    f.write(f"Acceptance Rate: {spec['acceptance_rate']:.2%}\n")
                    f.write(f"Draft Tokens: {spec['num_draft_tokens']}\n")
                    f.write(f"Accepted Tokens: {spec['num_accepted_tokens']}\n")
                    f.write(f"Mean accept length: {spec['system_efficiency']:.3f}\n")
            print(f"Comparison summary saved to: {comparison_path}")

    print("\nBenchmark complete with acceptance rate metrics!")


if __name__ == "__main__":
    main()