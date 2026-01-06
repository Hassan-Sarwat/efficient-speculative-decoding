import argparse
import json
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Distill data using a target model")
    parser.add_argument("--base_model", type=str, required=True, help="Base HF model (e.g. Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save distilled .jsonl file")
    args = parser.parse_args()

    print(f"Initializing vLLM with Base: {args.base_model}")
    
    # Check if adapter exists
    use_adapter = args.adapter_path and os.path.exists(os.path.join(args.adapter_path, "adapter_config.json"))
    if args.adapter_path and not use_adapter:
        print(f"Adapter path provided but invalid/empty: {args.adapter_path}. Using Base Model only.")

    # Initialize vLLM
    llm = LLM(
        model=args.base_model,
        enable_lora=True if use_adapter else False,
        max_lora_rank=64,
        gpu_memory_utilization=0.95,
        quantization="bitsandbytes", # 4-bit loading
        load_format="bitsandbytes",
        enforce_eager=True # often helps with LoRA compatibility
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024
    )

    # Load Data
    print(f"Loading Data from {args.input_file}...")
    dataset = load_dataset("json", data_files=args.input_file, split="train")

    # Check Existing
    existing_instructions = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_instructions.add(json.loads(line)["instruction"])
                    except: pass

    # Prepare Prompts
    prompts = []
    instructions_map = []
    
    for item in dataset:
        inst = item["instruction"]
        if inst not in existing_instructions:
            messages = [{"role": "user", "content": inst}]
            prompt_text = llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            instructions_map.append(inst)

    if not prompts:
        print("All items already processed.")
        return

    print(f"Generating {len(prompts)} responses...")
    
    lora_request = None
    if use_adapter:
        # 'adapter_path' serves as both the name and the path here for simplicity
        lora_request = LoRARequest("custom_adapter", 1, args.adapter_path)

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "a") as f:
        for instruction, output in zip(instructions_map, outputs):
            generated_text = output.outputs[0].text
            new_entry = {
                "instruction": instruction,
                "input": "",
                "output": generated_text
            }
            f.write(json.dumps(new_entry) + "\n")

    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()