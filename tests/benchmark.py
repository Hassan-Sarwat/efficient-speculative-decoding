import re
import time
import gc
import csv
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple, Optional, Dict

# 1. ROBUST PARSING FUNCTIONS (Ported from data_generation/analysis.ipynb)

def resolve_fractions(text: str) -> str:
    """
    Resolves both LaTeX fractions (\\frac{1}{2}) AND plain text fractions (1/2) to decimals.
    Examples: 
      "\\frac{1}{2}" -> "0.5"
      "1/2"          -> "0.5"
      "1,000/4"      -> "250.0"
    """
    if not text: return text
    text = str(text)
    
    # --- Helper: Perform Division ---
    def calculate_div(n_str, d_str):
        try:
            n = float(n_str.replace(',', '').strip())
            d = float(d_str.replace(',', '').strip())
            if d == 0: return None
            return str(n / d)
        except ValueError:
            return None

    # --- Pass 1: LaTeX Fractions (\frac{a}{b}) ---
    def repl_latex(m):
        val = calculate_div(m.group(2), m.group(3))
        return val if val is not None else m.group(0)

    # Pattern matches \frac{...}{...} or \dfrac{...}{...}
    text = re.sub(r'\\(d?)frac\{([^{}]+)\}\{([^{}]+)\}', repl_latex, text)

    # --- Pass 2: Plain Text Fractions (a/b) ---
    def repl_plain(m):
        val = calculate_div(m.group(1), m.group(2))
        return val if val is not None else m.group(0)

    # Pattern matches: number / number
    # Handles negatives (-1/2) and commas (1,000/2)
    # We use a lookahead (?!\d) to ensure we don't cut off numbers, though straightforward matching works well here.
    plain_pattern = r'(-?\d+(?:,\d+)*)\s*/\s*(-?\d+(?:,\d+)*)'
    text = re.sub(plain_pattern, repl_plain, text)

    return text

def extract_boxed_content(text: str) -> Optional[str]:
    if not text: return None
    idx = text.rfind("\\boxed{")
    if idx == -1: return None
    start_idx = idx + 7
    balance = 1
    for i in range(start_idx, len(text)):
        char = text[i]
        if char == "{": balance += 1
        elif char == "}":
            balance -= 1
            if balance == 0: return text[start_idx:i]
    return None

def clean_competition_math_answer(text: str) -> str:
    if not text: return ""
    text = text.replace("$", "")
    text = text.replace(",", "").strip()
    return text

def extract_answer(text: str, scenario: str) -> str:
    """
    Extracts answer based on scenario (easy=gsm8k, medium/hard=math).
    """
    if not text: return ""
    text = str(text)

    is_math = scenario in ['medium', 'hard']
    is_gsm8k = scenario == 'easy'

    # Prioritize boxed for math, #### for gsm8k, but have fallbacks
    if is_math:
        boxed = extract_boxed_content(text)
        if boxed: return clean_competition_math_answer(boxed)
        if "####" in text: return text.split("####")[-1].strip()
        return text.strip()

    if is_gsm8k:
        if "####" in text: return text.split("####")[-1].strip()
        # Fallback if no separator found (unlikely for gsm8k but possible)
        return text.strip()

    # Generic Fallback
    if "####" in text: return text.split("####")[-1].strip()
    boxed = extract_boxed_content(text)
    if boxed: return boxed
    return text.strip()

def normalize_string(text: str) -> str:
    if not text: return ""
    text = str(text).strip()
    text = text.replace(",", "")
    if text.endswith("."): text = text[:-1]
    return text

def parse_number(text: str):
    clean_text = text.replace(",", "")
    pattern = r'(-?\d+\.?\d*|-?\.\d+)(%)?'
    match = re.search(pattern, clean_text)
    if match:
        try:
            return float(match.group(1)), bool(match.group(2))
        except ValueError:
            pass
    return None, False

def check_equality(pred: str, gt: str) -> bool:
    s1, s2 = normalize_string(pred), normalize_string(gt)
    if s1 == s2: return True
    
    v1, p1 = parse_number(pred)
    v2, p2 = parse_number(gt)
    if v1 is None or v2 is None: return False
    
    def is_close(a, b): return abs(a - b) < 1e-6
    
    if p1 == p2: return is_close(v1, v2)
    if p1 and not p2: return is_close(v1, v2) or is_close(v1/100.0, v2)
    if p2 and not p1: return is_close(v2, v1) or is_close(v2/100.0, v1)
    return False

# 2. METRICS EXTRACTION HELPER
def get_speculative_metrics(llm_instance):
    """
    Robustly extracts speculative scheduler metrics for vLLM 0.6.4+
    """
    engine = getattr(llm_instance, 'llm_engine', None)
    if not engine: return None
    
    model_executor = getattr(engine, 'model_executor', None)
    
    # 3. Find the Model Runner
    if hasattr(model_executor, 'driver_worker'):
        driver = model_executor.driver_worker
    else:
        driver = model_executor
        
    model_runner = getattr(driver, 'model_runner', None)
    spec_scheduler = getattr(model_runner, 'spec_scheduler', None)
    
    if spec_scheduler:
        return spec_scheduler.metrics
    else:
        return None

# 3. BENCHMARK PASS FUNCTION
def run_benchmark_pass(name, data, stop_tokens, tokenizer, scenario, use_speculative=False, 
                      target_model="models/target", draft_model=None, csv_writer=None, run_id="", enable_lora=False):
    print(f"\n{'='*40}")
    print(f"🚀 RUNNING BENCHMARK: {name}")
    print(f"{'='*40}")

    # Reset Peak Memory Stats
    torch.cuda.reset_peak_memory_stats()
    
    # --- SENIOR ENGINEER FIX: Enforce Latency Regime ---
    # We limit max_num_seqs to simulate real-time chat traffic (low concurrency).
    # If we let vLLM batch 256+ requests, it becomes compute-bound and speculation fails.
    llm_kwargs = {
        "model": target_model, 
        "tensor_parallel_size": 1,
        "enforce_eager": True, # Match distill_data settings for compatibility
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.95, # Optimized memory usage
        "quantization": "bitsandbytes", # 4-bit loading
        "load_format": "bitsandbytes",
        "enable_lora": enable_lora,
        "max_lora_rank": 64,
    }

    if use_speculative:
        print(f"🔹 Speculative Decoding: ENABLED")
        print(f"🔹 Draft Model: {draft_model}")
        llm_kwargs["speculative_model"] = draft_model
        llm_kwargs["num_speculative_tokens"] = 5
    else:
        print(f"🔹 Speculative Decoding: DISABLED (Target Only)")

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return None

    params = SamplingParams(
        temperature=0, 
        max_tokens=512,
        stop=stop_tokens
    )

    print("🔨 Formatting Prompts...")
    prompts = []
    # Limit dataset size for quick debugging if needed, currently full set
    for q in data['question']:
        messages = [{"role": "user", "content": q}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    print(f"⏳ Starting Generation on {len(prompts)} samples...")
    start_time = time.time()
    outputs = llm.generate(prompts, params)
    end_time = time.time()
    
    duration = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / duration
    latency_per_req = (duration / len(prompts)) * 1000 # ms
    
    # PEAK VRAM
    max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB

    print(f"✅ Generation Complete!")
    ttft_list = []
    itl_list = []
    
    for o in outputs:
        m = o.metrics 
        
        # 1. Time to First Token
        if m.first_token_time and m.arrival_time:
            ttft = m.first_token_time - m.arrival_time
            ttft_list.append(ttft)
            
        # 2. Inter-Token Latency (Generation Time / Generated Tokens)
        generated_token_count = len(o.outputs[0].token_ids)
        if generated_token_count > 1 and m.finished_time and m.first_token_time:
            gen_time = m.finished_time - m.first_token_time
            itl = gen_time / (generated_token_count - 1)
            itl_list.append(itl)

    # Calculate Averages (convert to ms)
    avg_ttft = (sum(ttft_list) / len(ttft_list)) * 1000 if ttft_list else 0.0
    avg_itl = (sum(itl_list) / len(itl_list)) * 1000 if itl_list else 0.0

    print(f"⏱️ Duration:   {duration:.2f}s")
    print(f"⚡ Throughput: {tps:.2f} tokens/s")
    print(f"🚀 Avg TTFT:   {avg_ttft:.2f} ms") 
    print(f"🌊 Avg ITL:    {avg_itl:.2f} ms/token") 
    print(f"🧠 Peak VRAM:  {max_vram:.2f} GB")
    print(f"🐢 Latency:    {latency_per_req:.2f} ms/req")

    # Extract Speculative Metrics
    acceptance_rate = None
    if use_speculative:
        metrics = get_speculative_metrics(llm)
        if metrics:
            try:
                acceptance_rate = metrics.acceptance_rate
                print(f"🎯 Aggregated Acceptance Rate: {acceptance_rate:.2%}")
            except AttributeError:
                print(f"⚠️ Metrics found but could not parse acceptance_rate: {metrics}")
        else:
            print("⚠️ Speculative metrics not found via scheduler probing.")

    # Accuracy Eval
    score = 0
    print("\n🔍 EVALUATION SAMPLES (First 1):")
    
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data['answer'][i]
        
        # New extraction Logic
        pred_ext = extract_answer(gen_text, scenario)
        gt_ext = extract_answer(ground_truth, scenario) # Assuming GT also needs extracting (often it is pre-cleaned but safer to run)
        
        # Note: In analysis.ipynb, extract_answer(..., is_ground_truth=True) logic was:
        # if is_gsm8k: split(####)[-1]
        # if is_math: boxed or split(####)[-1]
        # Our updated extract_answer handles both based on content availability but let's trust it.
        # Actually, if the ground truth in dataset is just the final number, extract might return emptiness if it looks for ####.
        # Let's trust the ported extract_answer handles the standard dataset format (which usually has ####).
        
        is_correct = check_equality(pred_ext, gt_ext)
        score += 1 if is_correct else 0
        
        if i < 1:
            print(f"\n--- Sample {i} ---")
            print(f"📝 GT Raw: {ground_truth[-50:]}")
            print(f"📝 GT Ext: {gt_ext}")
            print(f"🤖 Pred Ext: {pred_ext}")
            print(f"✅ Correct? {is_correct}")

        if csv_writer:
            csv_writer.writerow([
                run_id,
                name,
                data['question'][i],
                ground_truth,
                gen_text,
                gt_ext,
                pred_ext,
                is_correct,
                len(output.outputs[0].token_ids)
            ])

    final_acc = (score / len(data)) * 100
    print(f"\n🏆 Accuracy: {final_acc}%")
    
    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "duration": duration,
        "accuracy": final_acc,
        "tps": tps,
        "acceptance_rate": acceptance_rate,
        "latency": latency_per_req,
        "max_vram": max_vram,
        "total_tokens": total_tokens,
        "ttft": avg_ttft,
        "itl": avg_itl
    }

# 4. MAIN
def main():
    parser = argparse.ArgumentParser(description="Run Speculative Decoding Benchmark")
    parser.add_argument("--scenario", type=str, required=True, choices=["easy", "medium", "hard"], help="Dataset scenario")
    parser.add_argument("--target-model", type=str, required=True, help="Path to target model")
    parser.add_argument("--draft-model", type=str, help="Path to draft model (optional)")
    parser.add_argument("--data-path", type=str, help="Path to test dataset jsonl (optional)")
    parser.add_argument("--use-speculative", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--enable-lora", action="store_true", help="Enable LoRA adapters")
    parser.add_argument("--run-name", type=str, default="benchmark", help="Name for the run (used in CSV)")
    
    args = parser.parse_args()

    print("📥 Loading Dataset & Tokenizer (Shared)...")
    
    # Load Dataset
    if args.data_path:
        # Load local jsonl
        data = load_dataset("json", data_files=args.data_path, split="train")
    else:
        # Fallback to standard logic if no path provided
        if args.scenario == "easy":
             data = load_dataset("gsm8k", "main", split="test")
        else:
             print("❌ Must provide --data-path for medium/hard scenarios (local files)")
             return

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    
    # Setup Output CSV
    os.makedirs("outputs", exist_ok=True)
    out_csv_path = f"outputs/{args.run_name}_{args.scenario}.csv" 
    
    # We want to append to a master CSV? Or just one per run?
    # User said: "outputs/{type}_{scenario}.csv"
    # We'll open in 'w' mode for this specific run invocation. 
    # If the caller runs multiple times, they should provide different run-names or we handle aggregation in the caller script.
    
    print(f"📝 Logging details to {out_csv_path}")
    
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "config_name", "question", "ground_truth", "prediction", "gt_extracted", "pred_extracted", "is_correct", "tokens"])
        
        metrics = run_benchmark_pass(
            name=args.run_name, 
            data=data, 
            stop_tokens=stop_tokens, 
            tokenizer=tokenizer, 
            scenario=args.scenario,
            use_speculative=args.use_speculative,
            target_model=args.target_model,
            draft_model=args.draft_model,
            csv_writer=writer,
            run_id=args.run_name,
            enable_lora=args.enable_lora
        )

    if metrics:
        print("\n📊 Run Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        # Save summary metrics to a separate file for easier aggregation
        summary_path = f"outputs/metrics_{args.run_name}.csv"
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())
        print(f"✅ Summary metrics saved to {summary_path}")

if __name__ == "__main__":
    main()