import re
from vllm import LLM, SamplingParams
from datasets import load_dataset

# 1. ROBUST PARSING FUNCTION
def extract_answer(generation, expected_answer):
    """
    Scans the entire generated text for the *last* number.
    This fixes 0% accuracy when models forget '####'.
    """
    # Clean up generation
    gen_text = generation.strip()
    
    # Strategy A: Look for strict "####" delimiter (GSM8K standard)
    if "####" in gen_text:
        pred = gen_text.split("####")[-1].strip()
    # Strategy B: Look for "The answer is"
    elif "The answer is" in gen_text:
        pred = gen_text.split("The answer is")[-1].strip()
    # Strategy C: Brute force - find the LAST number in the text
    else:
        # Find all numbers (integers or decimals)
        numbers = re.findall(r'-?\d+\.?\d*', gen_text)
        if numbers:
            pred = numbers[-1]
        else:
            return 0.0 # No number found

    # Clean punctuation (e.g., "16." -> "16")
    pred = re.sub(r'[^\d\.]', '', pred)
    
    # Compare with expected (also cleaned)
    expected = re.sub(r'[^\d\.]', '', expected_answer.split("####")[-1])
    
    try:
        return 1.0 if float(pred) == float(expected) else 0.0
    except ValueError:
        return 0.0

# 2. RUN BENCHMARK
def run_debug_benchmark():
    # Load limited set for debugging
    data = load_dataset("gsm8k", "main", split="test[:10]") 
    
    # ⚠️ CRITICAL: Set stop tokens for Qwen to prevent infinite generation
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    
    llm = LLM(
        model="/app/models/target", # Path to your 14B or 7B model
        speculative_model="/app/models/draft", # Path to 0.5B
        num_speculative_tokens=10,
        tensor_parallel_size=1,
        enforce_eager=False
    )
    
    params = SamplingParams(
        temperature=0, 
        max_tokens=512,
        stop=stop_tokens # VLLM needs this explicitly!
    )

    # Format prompts (ChatML)
    prompts = [
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" 
        for q in data['question']
    ]

    print("🚀 Generating...")
    outputs = llm.generate(prompts, params)

    score = 0
    print("\n🔍 DEBUG LOGS (First 3 samples):")
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data['answer'][i]
        
        is_correct = extract_answer(gen_text, ground_truth)
        score += is_correct
        
        if i < 3: # Print first 3 to see what's happening
            print(f"\n--- Sample {i} ---")
            print(f"📝 GT: {ground_truth.split('####')[-1].strip()}")
            print(f"🤖 Gen (Tail): ...{gen_text[-100:].replace(chr(10), ' ')}") # Print last 100 chars
            print(f"✅ Correct? {is_correct}")

    print(f"\n🏆 Final Accuracy: {(score / len(data)) * 100}%")

if __name__ == "__main__":
    run_debug_benchmark()