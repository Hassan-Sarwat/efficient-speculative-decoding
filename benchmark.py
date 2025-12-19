import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer  # Required for correct chat templating

# 1. ROBUST PARSING FUNCTION (Kept your robust version)
def extract_answer(generation, expected_answer):
    """
    Scans the entire generated text for the *last* number.
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
        numbers = re.findall(r'-?\d+\.?\d*', gen_text)
        if numbers:
            pred = numbers[-1]
        else:
            return 0.0 

    # Clean punctuation
    pred = re.sub(r'[^\d\.]', '', pred)
    expected = re.sub(r'[^\d\.]', '', expected_answer.split("####")[-1])
    
    try:
        return 1.0 if float(pred) == float(expected) else 0.0
    except ValueError:
        return 0.0

# 2. RUN BENCHMARK
def run_debug_benchmark():
    print("⏳ Loading Dataset...")
    # Load limited set for debugging
    data = load_dataset("gsm8k", "main", split="test[:20]") 
    
    print("⏳ Loading Tokenizer & Model...")
    # ⚠️ CRITICAL CHANGE: Load tokenizer to get the exact system prompt
    tokenizer = AutoTokenizer.from_pretrained("/app/models/target")
    
    # Set stop tokens for Qwen to prevent infinite generation
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    
    llm = LLM(
        model="/app/models/target", 
        speculative_model="/app/models/draft", 
        num_speculative_tokens=10,
        tensor_parallel_size=1,
        enforce_eager=False
    )
    
    params = SamplingParams(
        temperature=0, 
        max_tokens=512,
        stop=stop_tokens
    )

    # ⚠️ CRITICAL CHANGE: Use apply_chat_template instead of manual f-strings
    print("🔨 Formatting Prompts with System Prompt...")
    prompts = []
    for q in data['question']:
        messages = [{"role": "user", "content": q}]
        # This injects: "<|im_start|>system\nYou are Qwen...\n<|im_start|>user..."
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    print("🚀 Generating...")
    outputs = llm.generate(prompts, params)

    score = 0
    print("\n🔍 DEBUG LOGS (First 3 samples):")
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        ground_truth = data['answer'][i]
        
        is_correct = extract_answer(gen_text, ground_truth)
        score += is_correct
        
        if i < 3: 
            print(f"\n--- Sample {i} ---")
            print(f"📝 GT: {ground_truth.split('####')[-1].strip()}")
            # Print the tail to see if '####' is being generated now
            print(f"🤖 Gen (Tail): ...{gen_text[-100:].replace(chr(10), ' ')}") 
            print(f"✅ Correct? {is_correct}")

    final_acc = (score / len(data)) * 100
    print(f"\n🏆 Final Accuracy: {final_acc}%")

if __name__ == "__main__":
    run_debug_benchmark()