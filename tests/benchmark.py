import re
import time
import gc
import csv
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from transformers import AutoTokenizer
from answer_utils import extract_answer, check_equality


# METRICS EXTRACTION HELPER
def get_speculative_metrics(llm_instance):
    """
    Robustly extracts speculative scheduler metrics for vLLM 0.6.4+
    """
    engine = getattr(llm_instance, 'llm_engine', None)
    if not engine: return None
    
    model_executor = getattr(engine, 'model_executor', None)
    
    # Find the Model Runner
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


# BENCHMARK PASS FUNCTION
def run_benchmark_pass(name, data, stop_tokens, tokenizer, scenario, use_speculative=False, 
                      target_base="Qwen/Qwen2.5-14B-Instruct", target_adapter=None, 
                      draft_base="Qwen/Qwen2.5-0.5B-Instruct", draft_adapter=None, 
                      csv_writer=None, run_id="", enable_lora=False):
    print(f"\n{'='*40}")
    print(f"🚀 RUNNING BENCHMARK: {name}")
    print(f"{'='*40}")

    # Reset Peak Memory Stats
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Prepare Target Model (always use base model + LoRA adapter at runtime)
    target_model_path = target_base

    # 2. Prepare Draft Model (expect pre-merged model)
    speculative_model_path = None
    if use_speculative:
        if draft_adapter:
            # Verify the merged model exists
            if not os.path.exists(draft_adapter):
                print(f"❌ ERROR: Draft model not found at {draft_adapter}")
                print(f"    Did you run train_pipeline.sh to merge the draft model?")
                return None
            if not os.path.exists(os.path.join(draft_adapter, "config.json")):
                print(f"❌ ERROR: Invalid draft model at {draft_adapter} (missing config.json)")
                return None
                
            speculative_model_path = draft_adapter  
            print(f"🔹 Using pre-merged draft model: {speculative_model_path}")
        else:
            speculative_model_path = draft_base
            print(f"🔹 Using base draft model (untrained): {speculative_model_path}")

    # 3. Configure vLLM for Target Model
    llm_kwargs = {
        "model": target_model_path,
        "enable_lora": True,
        "max_lora_rank": 64,
        "dtype": "float16",  # FP16 for LoRA compatibility
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
    }

    # 4. Add speculative config if enabled
    if use_speculative and speculative_model_path:
        print(f"🔹 Speculative Decoding: ENABLED")
        print(f"🔹 Draft Model: {speculative_model_path}")
        
        # vLLM 0.9.1 API: Use speculative_config as a dictionary parameter
        llm_kwargs["speculative_config"] = {
            "model": speculative_model_path,
            "num_speculative_tokens": 5,
        }
    else:
        print(f"🔹 Speculative Decoding: DISABLED (Target Only)")
    
    # 5. Initialize vLLM
    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 6. Attach target adapter if provided (runtime LoRA for target only)
    target_lora_request = None
    if target_adapter:
        print(f"🔗 Attaching Target LoRA Adapter: {target_adapter}")
        target_lora_request = LoRARequest("target_adapter", 1, target_adapter)
    
    # 7. Configure sampling
    params = SamplingParams(
        temperature=0, 
        max_tokens=512,
        stop=stop_tokens
    )

    # 8. Format prompts
    print("🔨 Formatting Prompts...")
    prompts = []
    question_key = "question" if "question" in data.column_names else "instruction"
    answer_key = "answer" if "answer" in data.column_names else "output"

    if question_key not in data.column_names:
        raise ValueError(f"Dataset must contain '{question_key}' column. Found: {data.column_names}")

    for q in data[question_key]:
        messages = [{"role": "user", "content": q}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    # 9. Run generation
    print(f"⏳ Starting Generation on {len(prompts)} samples...")
    
    start_time = time.time()
    outputs = llm.generate(prompts, params, lora_request=target_lora_request)
    end_time = time.time()
    
    duration = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / duration
    latency_per_req = (duration / len(prompts)) * 1000  # ms
    
    # Peak VRAM
    max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

    # 10. Calculate latency metrics
    print(f"✅ Generation Complete!")
    ttft_list = []
    itl_list = []
    
    for o in outputs:
        m = o.metrics 
        
        if m.first_token_time and m.arrival_time:
            ttft = m.first_token_time - m.arrival_time
            ttft_list.append(ttft)
            
        generated_token_count = len(o.outputs[0].token_ids)
        if generated_token_count > 1 and m.finished_time and m.first_token_time:
            gen_time = m.finished_time - m.first_token_time
            itl = gen_time / (generated_token_count - 1)
            itl_list.append(itl)

    avg_ttft = (sum(ttft_list) / len(ttft_list)) * 1000 if ttft_list else 0.0
    avg_itl = (sum(itl_list) / len(itl_list)) * 1000 if itl_list else 0.0

    print(f"⏱️  Duration:   {duration:.2f}s")
    print(f"⚡ Throughput: {tps:.2f} tokens/s")
    print(f"🚀 Avg TTFT:   {avg_ttft:.2f} ms") 
    print(f"🌊 Avg ITL:    {avg_itl:.2f} ms/token") 
    print(f"🧠 Peak VRAM:  {max_vram:.2f} GB")
    print(f"🐢 Latency:    {latency_per_req:.2f} ms/req")

    # 11. Extract speculative metrics
    acceptance_rate = None
    if use_speculative:
        metrics = get_speculative_metrics(llm)
        if metrics:
            try:
                acceptance_rate = metrics.acceptance_rate
                print(f"🎯 Acceptance Rate: {acceptance_rate:.2%}")
            except AttributeError:
                print(f"⚠️  Metrics found but could not parse acceptance_rate: {metrics}")
        else:
            print("⚠️  Speculative metrics not found via scheduler probing.")

    # 12. Evaluate accuracy
    score = 0
    print("\n🔍 EVALUATION SAMPLES (First 1):")
    
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data[answer_key][i]
        
        pred_ext = extract_answer(gen_text, scenario)
        gt_ext = extract_answer(ground_truth, scenario)
        
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
                data[question_key][i],
                ground_truth,
                gen_text,
                gt_ext,
                pred_ext,
                is_correct,
                len(output.outputs[0].token_ids)
            ])

    final_acc = (score / len(data)) * 100
    print(f"\n🏆 Accuracy: {final_acc}%")
    
    # Calculate additional metrics for analysis
    avg_output_tokens = total_tokens / len(data)
    print(f"📏 Avg Output Length: {avg_output_tokens:.1f} tokens/sample")
    
    # 13. Cleanup
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
        "avg_output_tokens": avg_output_tokens,
        "ttft": avg_ttft,
        "itl": avg_itl,
        "num_samples": len(data),
        "use_speculative": use_speculative,
    }


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Run Speculative Decoding Benchmark")
    parser.add_argument("--scenario", type=str, required=True, choices=["easy", "medium", "hard"], help="Dataset scenario")
    
    # Target Model Args
    parser.add_argument("--target-base-model", type=str, required=True, help="Base Model for Target (e.g. Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--target-adapter", type=str, help="Path to Target LoRA adapter (optional)")
    
    # Draft Model Args (Speculative)
    parser.add_argument("--draft-base-model", type=str, help="Base Model for Draft (optional)")
    parser.add_argument("--draft-adapter", type=str, help="Path to MERGED draft model (not LoRA adapter)")
    
    # Legacy args support (for backwards compatibility)
    parser.add_argument("--target-model", type=str, help="Legacy: Path to target model/adapter")
    parser.add_argument("--draft-model", type=str, help="Legacy: Path to draft model/adapter")

    parser.add_argument("--data-path", type=str, help="Path to test dataset jsonl (optional)")
    parser.add_argument("--use-speculative", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--enable-lora", action="store_true", help="Enable LoRA adapters support")
    parser.add_argument("--run-name", type=str, default="benchmark_run", help="Run name for CSV")
    
    args = parser.parse_args()
    
    # Use new args (backwards compatibility with legacy args)
    target_base = args.target_base_model
    target_adapter = args.target_adapter
    draft_base = args.draft_base_model
    draft_adapter = args.draft_adapter
    
    # Load Dataset
    print("📥 Loading Dataset...")
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

    # Initialize Tokenizer (from base model)
    print(f"📥 Loading Tokenizer from {target_base}...")
    tokenizer = AutoTokenizer.from_pretrained(target_base, trust_remote_code=True)
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    
    # Setup Output CSV
    os.makedirs("outputs", exist_ok=True)
    out_csv_path = f"outputs/{args.run_name}_{args.scenario}.csv" 
    
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
            target_base=target_base,
            target_adapter=target_adapter,
            draft_base=draft_base,
            draft_adapter=draft_adapter,
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