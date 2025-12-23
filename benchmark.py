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
    """
    gen_text = generation.strip()
    
    if "####" in gen_text:
        pred = gen_text.split("####")[-1].strip()
    elif "The answer is" in gen_text:
        pred = gen_text.split("The answer is")[-1].strip()
    else:
        numbers = re.findall(r'-?\d+\.?\d*', gen_text)
        if numbers:
            pred = numbers[-1]
        else:
            return 0.0 

    pred = re.sub(r'[^\d\.]', '', pred)
    expected = re.sub(r'[^\d\.]', '', expected_answer.split("####")[-1])
    
    try:
        is_match = float(pred) == float(expected)
        if not is_match:
            print(f"[DEBUG] Fail: PredRaw='{pred}' PredFloat={float(pred)} ExpRaw='{expected}' ExpFloat={float(expected)}")
            print(f"[DEBUG] GenTail='{gen_text[-50:]}'")
        return 1.0 if is_match else 0.0
    except ValueError:
        print(f"[DEBUG] ValueError: Pred='{pred}' Expected='{expected}'")
        return 0.0

# 2. METRICS EXTRACTION HELPER
def get_speculative_metrics(llm_instance):
    """
    Robustly extracts speculative scheduler metrics for vLLM 0.6.4+
    """
    # 1. Access the engine
    engine = getattr(llm_instance, 'llm_engine', None)
    if not engine:
        return None
    
    # 2. Dig into the Model Executor
    model_executor = getattr(engine, 'model_executor', None)
    
    # 3. Find the Model Runner
    if hasattr(model_executor, 'driver_worker'):
        driver = model_executor.driver_worker
    else:
        driver = model_executor
        
    model_runner = getattr(driver, 'model_runner', None)
    
    # 4. Extract the Speculative Scheduler Metrics
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
    
    llm_kwargs = {
        "model": "models/target", 
        "tensor_parallel_size": 1,
        "enforce_eager": False
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
        return None, 0.0, 0.0, None, 0.0, 0.0, 0

    params = SamplingParams(
        temperature=0, 
        max_tokens=512,
        stop=stop_tokens
    )

    print("🔨 Formatting Prompts...")
    prompts = []
    for q in data['question']:
        messages = [{"role": "user", "content": q}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    print("⏳ Starting Generation...")
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
        # vLLM provides 'metrics' in the output object
        # timestamp fields: arrival_time, first_token_time, finished_time
        m = o.metrics 
        
        # 1. Time to First Token
        if m.first_token_time and m.arrival_time:
            ttft = m.first_token_time - m.arrival_time
            ttft_list.append(ttft)
            
        # 2. Inter-Token Latency (Generation Time / Generated Tokens)
        # Note: We subtract 1 token because the first token is covered by TTFT
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
    print(f"🚀 Avg TTFT:   {avg_ttft:.2f} ms") # <--- NEW
    print(f"🌊 Avg ITL:    {avg_itl:.2f} ms/token") # <--- NEW
    print(f"🧠 Peak VRAM:  {max_vram:.2f} GB")
    print(f"🐢 Latency:    {latency_per_req:.2f} ms/req")

    # Extract Speculative Metrics
    acceptance_rate = None
    if use_speculative:
        metrics = get_speculative_metrics(llm)
        if metrics:
            try:
                # Try accessing acceptance_rate directly if available
                acceptance_rate = metrics.acceptance_rate
                print(f"🎯 Aggregated Acceptance Rate: {acceptance_rate:.2%}")
            except AttributeError:
                # Fallback calculation if possible, or just print metrics object
                print(f"⚠️ Metrics found but could not parse acceptance_rate: {metrics}")
        else:
            print("⚠️ Speculative metrics not found via scheduler probing.")

    # Accuracy Eval
    score = 0
    print("\n🔍 EVALUATION SAMPLES:")
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data['answer'][i]
        is_correct = extract_answer(gen_text, ground_truth)
        score += is_correct
        if i < 2:
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
    
    return duration, final_acc, tps, acceptance_rate, latency_per_req, max_vram, total_tokens

# 4. MAIN
def main():
    print("📥 Loading Dataset & Tokenizer (Shared)...")
    data = load_dataset("gsm8k", "main", split="test[:20]") 
    tokenizer = AutoTokenizer.from_pretrained("models/target")
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]

    # --- Run 1: Target Only ---
    dur_base, acc_base, tps_base, _, lat_base, vram_base, tokens_base = run_benchmark_pass(
        "Standard (Target Only)", 
        data, 
        stop_tokens, 
        tokenizer, 
        use_speculative=False
    )

    # --- Run 2: Speculative Decoding ---
    dur_spec, acc_spec, tps_spec, ar_spec, lat_spec, vram_spec, tokens_spec = run_benchmark_pass(
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
        print(f"� Speedup Factor:   {speedup:.2f}x")
    else:
        print("❌ Could not calculate speedup.")

    # Metrics Bundle
    # Format: (Standard) -> (Speculative)
    print(f"\n⏱️ Duration:        {dur_base:.2f}s -> {dur_spec:.2f}s")
    print(f"⚡ Throughput:      {tps_base:.2f} -> {tps_spec:.2f} tok/s")
    print(f"🐢 Latency:         {lat_base:.2f} -> {lat_spec:.2f} ms/req")
    print(f"🧠 Peak VRAM:       {vram_base:.2f} -> {vram_spec:.2f} GB")
    print(f"🔢 Total Tokens:    {tokens_base} -> {tokens_spec}")
    print(f"🏆 Accuracy:        {acc_base}% -> {acc_spec}%")

    # Acceptance Rate
    if ar_spec is not None:
        print(f"\n🎯 Speculative Acceptance Rate: {ar_spec:.2f}") # Formatted as float 0-1 usually, or convert to %
        if isinstance(ar_spec, float):
             print(f"   (approx {ar_spec:.1%})")
    else:
        print("\n❓ Speculative Acceptance Rate: Unknown")

if __name__ == "__main__":
    main()