#!/usr/bin/env python3
"""
Benchmark with Speculative Decoding Metrics Extraction
Captures vLLM's logged metrics from stderr
"""

import time
import sys
import os

from dataclasses import dataclass, asdict
import gc
import csv
import argparse
import json
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from transformers import AutoTokenizer
# Add src directory to path so answer_utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
os.environ["VLLM_USE_V1"] = "0"
from answer_utils import extract_answer, check_equality


@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for blog analysis"""
    # Primary Metrics
    avg_tokens_generated: float
    total_tokens_generated: int
    acceptance_rate: float  # NOW ACCESSIBLE!
    
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
    
    # Speculative-specific (extracted from logs)
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    system_efficiency: float = 0.0
    
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
            print(f"   System Efficiency: {self.system_efficiency:.2%}")
        elif self.use_speculative:
            print(f"Acceptance Rate: Not captured (check logs)")
        print(f"Accuracy: {self.accuracy:.2%}")
        print(f"Tokens/sec: {self.tokens_per_second:.1f}")
        print(f"ITL: {self.avg_itl_ms:.2f}ms/token")
        print(f"Peak VRAM: {self.peak_vram_gb:.2f}GB")
        print("="*60 + "\n")


def _get_prometheus_counter(metric_name: str) -> float:
    """Read a Prometheus counter value from the vLLM registry."""
    from prometheus_client import REGISTRY
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            total = 0.0
            for sample in metric.samples:
                if sample.name.endswith('_total') or sample.name == metric_name:
                    total += sample.value
            return total
    raise ValueError(f"Metric '{metric_name}' not found in Prometheus registry")

def snapshot_spec_metrics() -> dict:
    """
    Snapshot current speculative decoding Prometheus counters.
    Call before and after generation, then subtract to get the delta.
    Returns dict with counter values, or None if metrics not available.
    """
    try:
        return {
            'num_draft_tokens': _get_prometheus_counter('vllm:spec_decode_num_draft_tokens_total'),
            'num_accepted_tokens': _get_prometheus_counter('vllm:spec_decode_num_accepted_tokens_total'),
        }
    except ValueError as e:
        print(f"[DEBUG] Prometheus metric not found: {e}")
        return None
    
def run_benchmark_pass(name, data, stop_tokens, tokenizer, scenario, use_speculative=False, 
                      target_base="Qwen/Qwen3-14B", target_adapter=None, 
                      draft_base="Qwen/Qwen3-0.6B", draft_adapter=None, 
                      csv_writer=None, run_id="", enable_lora=False):
    """Run a single benchmark pass with metric capture"""
    
    print(f"\n{'='*40}")
    print(f"RUNNING BENCHMARK: {name}")
    print(f"{'='*40}")

    # Reset Peak Memory Stats
    torch.cuda.reset_peak_memory_stats()
    
    # Prepare log file path
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
        "gpu_memory_utilization": 0.85,  # Slightly lower to accommodate CUDA graph memory
        # "enforce_eager": True,  # REMOVED - enables CUDA graphs for accurate benchmarking
        "max_model_len": 4096,
        "max_num_seqs": 16,
        "enable_prefix_caching": False,
    }
    
    llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

    # 4. Add speculative config
    num_spec_tokens = 0
    if use_speculative and speculative_model_path:
        print(f"Speculative Decoding: ENABLED")
        print(f"Draft Model: {speculative_model_path}")
        
        num_spec_tokens = 5
        llm_kwargs["speculative_config"] = {
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

    # 6. Attach target adapter
    target_lora_request = None
    if target_adapter:
        print(f"Attaching Target LoRA Adapter: {target_adapter}")
        target_lora_request = LoRARequest("target_adapter", 1, target_adapter)
    
    # 7. Configure sampling
    params = SamplingParams(
        temperature=0, 
        max_tokens=512,
        stop=stop_tokens
    )

    # 8. Format prompts
    print("Formatting Prompts...")
    prompts = []
    question_key = "question" if "question" in data.column_names else "instruction"
    answer_key = "answer" if "answer" in data.column_names else "output"

    if question_key not in data.column_names:
        raise ValueError(f"Dataset must contain '{question_key}' column")

    for q in data[question_key]:
        messages = [{"role": "user", "content": q}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    # 9. Run generation with Prometheus metric snapshots
    print(f"Starting Generation on {len(prompts)} samples...")

    # Snapshot BEFORE generation
    before_snap = snapshot_spec_metrics() if use_speculative else None

    start_time = time.time()
    outputs = llm.generate(prompts, params, lora_request=target_lora_request)
    end_time = time.time()

    # Snapshot AFTER generation and compute delta
    acceptance_rate = 0.0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    system_efficiency = 0.0

    if use_speculative and before_snap is not None:
        after_snap = snapshot_spec_metrics()
        if after_snap is not None:
            num_draft_tokens = int(after_snap['num_draft_tokens'] - before_snap['num_draft_tokens'])
            num_accepted_tokens = int(after_snap['num_accepted_tokens'] - before_snap['num_accepted_tokens'])
            if num_draft_tokens > 0:
                acceptance_rate = num_accepted_tokens / num_draft_tokens
                # system_efficiency = accepted / (accepted + draft steps used)
                num_steps = num_draft_tokens / num_spec_tokens
                system_efficiency = num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0            
            print(f"\nPrometheus metrics captured successfully!")
            print(f"   Draft tokens:    {num_draft_tokens}")
            print(f"   Accepted tokens: {num_accepted_tokens}")
            print(f"   Acceptance rate: {acceptance_rate:.3f}")
            print(f"   System efficiency: {system_efficiency:.3f}")
        else:
            print(f"\n[WARNING] Could not read post-generation Prometheus metrics")
    elif use_speculative:
        print(f"\n[WARNING] Prometheus spec metrics not available — counters not registered")

    
    duration = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / duration
    
    # Peak VRAM
    max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # 10. Calculate latency metrics
    print("Generation Complete!")
    ttft_list = []
    itl_list = []
    metrics_available = 0
    metrics_missing = 0

    for o in outputs:
        m = o.metrics
        
        # Guard against None metrics (can happen in edge cases)
        if m is None:
            metrics_missing += 1
            continue
        
        metrics_available += 1
        
        if m.first_token_time is not None and m.arrival_time is not None:
            ttft_seconds = m.first_token_time - m.arrival_time
            ttft_ms = ttft_seconds * 1000
            ttft_list.append(ttft_ms)
            
        generated_token_count = len(o.outputs[0].token_ids)
        if generated_token_count > 1 and m.finished_time is not None and m.first_token_time is not None:
            gen_time_seconds = m.finished_time - m.first_token_time
            itl_seconds = gen_time_seconds / (generated_token_count - 1)
            itl_ms = itl_seconds * 1000
            itl_list.append(itl_ms)

    # Log metrics availability for transparency
    if metrics_missing > 0:
        print(f"   ⚠ Metrics unavailable for {metrics_missing}/{len(outputs)} samples")
    print(f"   TTFT computed for {len(ttft_list)}/{metrics_available} samples")
    print(f"   ITL computed for {len(itl_list)}/{metrics_available} samples")

    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0
    avg_itl = sum(itl_list) / len(itl_list) if itl_list else 0.0

    print(f"Duration:   {duration:.2f}s")
    print(f"Throughput: {tps:.2f} tokens/s")
    print(f"Avg ITL:    {avg_itl:.2f} ms/token") 
    print(f"Peak VRAM:  {max_vram:.2f} GB")

    # 12. Evaluate accuracy
    score = 0
    print("\nEVALUATION SAMPLES (First 1):")
    
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data[answer_key][i]
        
        pred_ext = extract_answer(gen_text, scenario)
        gt_ext = extract_answer(ground_truth, scenario)
        
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
    
    # 13. Cleanup
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
        system_efficiency=system_efficiency,
    )

    # Save metrics to JSON
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
    
    # Resolve model paths
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
    
    # Load tokenizer
    print(f"Loading Tokenizer from {target_base}...")
    tokenizer = AutoTokenizer.from_pretrained(target_base, trust_remote_code=True)
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    
    # Setup Output CSV
    os.makedirs("outputs", exist_ok=True)
    out_csv_path = f"outputs/{args.run_name}_{args.scenario}.csv" 
    
    print(f"Logging details to {out_csv_path}")
    
    # Determine which benchmarks to run
    run_baseline = args.run_both or not args.use_speculative
    run_spec = args.run_both or args.use_speculative
    
    all_metrics = {}
    
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "config_name", "question", "ground_truth", "prediction", 
                        "gt_extracted", "pred_extracted", "is_correct", "tokens"])
        
        # Run baseline if requested
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
        
        # Run speculative if requested
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
    
    # Save unified metrics JSON
    if all_metrics:
        unified_metrics_path = f"outputs/metrics_{args.run_name}.json"
        with open(unified_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nUnified metrics saved to: {unified_metrics_path}")
        
        # Calculate and display speedup if both runs completed
        if 'baseline' in all_metrics and 'speculative' in all_metrics:
            baseline = all_metrics['baseline']
            spec = all_metrics['speculative']
            
            speedup =  spec['tokens_per_second'] / baseline['tokens_per_second']  if spec['tokens_per_second'] > 0 else 0
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
                print(f"   System Efficiency: {spec['system_efficiency']:.2%}")
            else:
                print(f"Acceptance Rate: Not available (check logs)")
            
            print("="*60)
            
            # Save comparison summary
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
                    f.write(f"System Efficiency: {spec['system_efficiency']:.2%}\n")
            print(f"Comparison summary saved to: {comparison_path}")
    
    print("\nBenchmark complete with acceptance rate metrics!")


if __name__ == "__main__":
    main()