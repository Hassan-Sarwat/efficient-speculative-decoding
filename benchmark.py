import re
import time
import gc
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. ROBUST PARSING FUNCTION
def extract_answer(generation, expected_answer):
    """
    Scans the entire generated text for the *last* number.
    Strategy: Find the last numerical value in the generated text.
    If explicit markers are found, narrow the search scope to the text following them.
    """
    text_to_search = generation
    
    if "####" in generation:
        text_to_search = generation.split("####")[-1]
    elif "The answer is" in generation:
        text_to_search = generation.split("The answer is")[-1]

    # Look for numbers (integers or floats) in the text segment
    # logic: -? (optional negative) \d+ (digits) (\.\d+)? (optional decimal part)
    # We strip commas first to handle 70,000 -> 70000
    clean_text = text_to_search.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', clean_text)
    
    if numbers:
        pred_str = numbers[-1]
        # Remove any trailing dots (e.g. "15." -> "15") often captured if sentence ends with dot
        if pred_str.endswith('.'):
             pred_str = pred_str[:-1]
    else:
        return 0.0

    # Parse Expected
    expected_str = expected_answer.split("####")[-1].strip().replace(',', '')
    # Expected might be "18" or "18." or "$18"
    # Just extract the number
    exp_matches = re.findall(r'-?\d+\.?\d*', expected_str)
    if exp_matches:
        expected_val = exp_matches[-1]
        if expected_val.endswith('.'):
            expected_val = expected_val[:-1]
    else:
        return 0.0
    
    try:
        return 1.0 if float(pred_str) == float(expected_val) else 0.0
    except ValueError:
        return 0.0

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
def run_benchmark_pass(name, data, stop_tokens, tokenizer, use_speculative=False):
    print(f"\n{'='*40}")
    print(f"🚀 RUNNING BENCHMARK: {name}")
    print(f"{'='*40}")

    # Reset Peak Memory Stats
    torch.cuda.reset_peak_memory_stats()

    spec_model = "models/draft_padded" if use_speculative else None
    
    # --- SENIOR ENGINEER FIX: Enforce Latency Regime ---
    # We limit max_num_seqs to simulate real-time chat traffic (low concurrency).
    # If we let vLLM batch 256+ requests, it becomes compute-bound and speculation fails.
    llm_kwargs = {
        "model": "models/target", 
        "tensor_parallel_size": 1,
        "enforce_eager": False,
        "max_num_seqs": 32,  # <--- CRITICAL FIX for consistent speedups
    }

    if use_speculative:
        print(f"🔹 Speculative Decoding: ENABLED")
        print(f"🔹 Draft Model: {spec_model}")
        llm_kwargs["speculative_model"] = spec_model
        llm_kwargs["num_speculative_tokens"] = 5
    else:
        print(f"🔹 Speculative Decoding: DISABLED (Target Only)")

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        # Return extra Nones for the new metrics
        return None, 0.0, 0.0, None, 0.0, 0.0, 0, 0.0, 0.0

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
    # Reduce log noise, only print first sample
    print("\n🔍 EVALUATION SAMPLES (First 1):")
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data['answer'][i]
        is_correct = extract_answer(gen_text, ground_truth)
        score += is_correct
        if i < 1:
            print(f"\n--- Sample {i} ---")
            print(f"📝 GT: {ground_truth.split('####')[-1].strip()}")
            print(f"🤖 Gen (Tail): ...{gen_text[-100:].replace(chr(10), ' ')}") 
            print(f"✅ Correct? {is_correct}")

    final_acc = (score / len(data)) * 100
    print(f"\n🏆 Accuracy: {final_acc}%")
    
    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- UPDATED RETURN SIGNATURE ---
    return duration, final_acc, tps, acceptance_rate, latency_per_req, max_vram, total_tokens, avg_ttft, avg_itl

# 4. MAIN
def main():
    print("📥 Loading Dataset & Tokenizer (Shared)...")
    # Load full test set (~1319 samples)
    data = load_dataset("gsm8k", "main", split="test") 
    tokenizer = AutoTokenizer.from_pretrained("models/target")
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]

    # --- Run 1: Target Only ---
    (dur_base, acc_base, tps_base, _, lat_base, vram_base, 
     tokens_base, ttft_base, itl_base) = run_benchmark_pass(
        "Standard (Target Only)", 
        data, 
        stop_tokens, 
        tokenizer, 
        use_speculative=False
    )

    # --- Run 2: Speculative Decoding ---
    (dur_spec, acc_spec, tps_spec, ar_spec, lat_spec, vram_spec, 
     tokens_spec, ttft_spec, itl_spec) = run_benchmark_pass(
        "Speculative (Target + Draft)", 
        data, 
        stop_tokens, 
        tokenizer, 
        use_speculative=True
    )

    # --- Final Report ---
    print(f"\n{'='*40}")
    print("📊 FINAL BENCHMARK REPORT")
    print(f"{'='*40}")
    
    # Speedup
    if dur_base and dur_spec:
        speedup = dur_base / dur_spec
        print(f"🚀 Speedup Factor:   {speedup:.2f}x")
    else:
        print("❌ Could not calculate speedup.")

    # Metrics Bundle
    # Format: (Standard) -> (Speculative)
    print(f"\n⏱️ Duration:        {dur_base:.2f}s -> {dur_spec:.2f}s")
    print(f"⚡ Throughput:      {tps_base:.2f} -> {tps_spec:.2f} tok/s")
    print(f"🐢 Latency:         {lat_base:.2f} -> {lat_spec:.2f} ms/req")
    print(f"🌊 Avg ITL:         {itl_base:.2f} -> {itl_spec:.2f} ms/token")
    print(f"🚀 Avg TTFT:        {ttft_base:.2f} -> {ttft_spec:.2f} ms")
    print(f"🧠 Peak VRAM:       {vram_base:.2f} -> {vram_spec:.2f} GB")
    print(f"🔢 Total Tokens:    {tokens_base} -> {tokens_spec}")
    print(f"🏆 Accuracy:        {acc_base:.2f}% -> {acc_spec:.2f}%")

    # Acceptance Rate
    if ar_spec is not None:
        print(f"\n🎯 Speculative Acceptance Rate: {ar_spec:.2f}") 
        if isinstance(ar_spec, float):
             print(f"   (approx {ar_spec:.1%})")
    else:
        print("\n❓ Speculative Acceptance Rate: Unknown")

if __name__ == "__main__":
    main()