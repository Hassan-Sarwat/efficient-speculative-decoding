# distill_untrained.py
import argparse
import json
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
import gc
import torch

# Specific System Prompt for the Untrained Baseline
UNTRAINED_SYSTEM_PROMPT = """You are a helpful AI assistant.
Please answer the user's question.
IMPORTANT: You must provide your final answer at the end of your response, strictly immediately after the separator ####."""

def main():
    parser = argparse.ArgumentParser(description="Distill data from an untrained base model using vLLM")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-14B", help="Base HF model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save distilled .jsonl file")
    args = parser.parse_args()

    # 1. Check for existing progress
    existing_instructions = set()
    if os.path.exists(args.output_file):
        print(f"Found existing output file at {args.output_file}. Resuming...")
        with open(args.output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "instruction" in data:
                            existing_instructions.add(data["instruction"])
                    except: pass
    print(f"Already processed {len(existing_instructions)} items.")

    print(f"Initializing vLLM with Base: {args.base_model}")
    
    llm = LLM(
        model=args.base_model,
        dtype="float16",  # FP16 instead of quantization
        enable_lora=False,
        gpu_memory_utilization=0.95,  # Use more VRAM on A40
        max_model_len=4096,  # Allow longer reasoning chains
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    tokenizer = llm.get_tokenizer()

    print(f"Loading Data from {args.input_file}...")
    dataset = load_dataset("json", data_files=args.input_file, split="train")

    # Prepare Prompts with the Untrained System Prompt
    prompts = []
    instructions_map = []
    
    for item in dataset:
        inst = item["instruction"]
        if inst not in existing_instructions:
            # Inject the specific system prompt here
            messages = [
                {"role": "system", "content": UNTRAINED_SYSTEM_PROMPT},
                {"role": "user", "content": inst}
            ]
            
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            instructions_map.append(inst)

    if not prompts:
        print("All items already processed.")
        return

    print(f"Generating {len(prompts)} responses...")

    # Sampling params equivalent to your unsloth config (temp 0.7, top_p 0.9)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )

    # Generate in batches for better memory management
    batch_size = 64
    all_outputs = []
    
    print(f"Processing in batches of {batch_size}...")
    for i in range(0, len(prompts), batch_size):
        batch_end = min(i + batch_size, len(prompts))
        batch_prompts = prompts[i:batch_end]
        
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(batch_outputs)
        
        print(f"Processed {batch_end}/{len(prompts)} samples")

    # Save Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, "a") as f:
        for instruction, output in zip(instructions_map, all_outputs):
            generated_text = output.outputs[0].text
            new_entry = {
                "instruction": instruction,
                "input": "",
                "output": generated_text
            }
            f.write(json.dumps(new_entry) + "\n")

    print(f"Distillation Complete! Saved to {args.output_file}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared")

if __name__ == "__main__":
    main()