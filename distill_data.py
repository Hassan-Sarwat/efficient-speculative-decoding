from unsloth import FastLanguageModel
from datasets import load_dataset
import json
import torch
from tqdm import tqdm

import os

def distill():
    # Configuration
    MODEL_PATH = "/models/target"  # Must match final_save_path in target config
    INPUT_FILE = "data/cob_data.jsonl"
    OUTPUT_FILE = "data/cod_distilled_data.jsonl"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True

    # 1. Check for existing progress
    processed_instructions = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"� Found existing output file at {OUTPUT_FILE}. Resuming...")
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "instruction" in data:
                            processed_instructions.add(data["instruction"])
                    except json.JSONDecodeError:
                        continue
    print(f"✅ Already processed {len(processed_instructions)} items.")

    print(f"�🚀 Loading Target Model from {MODEL_PATH}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )
    except Exception as e:
        print(f"⚠️ Failed to load from {MODEL_PATH}. Make sure the target model is trained and saved.")
        raise e

    FastLanguageModel.for_inference(model)

    print(f"📂 Loading Data from {INPUT_FILE}...")
    dataset = load_dataset("json", data_files=INPUT_FILE, split="train")

    # Filter out already processed instructions
    # Using a list comprehension to preserve order of remaining items
    instructions_to_process = [
        inst for inst in dataset["instruction"] 
        if inst not in processed_instructions
    ]

    if not instructions_to_process:
        print("🎉 All items have already been processed!")
        return

    print(f"🔥 Generating Responses for {len(instructions_to_process)} samples (Skipped {len(processed_instructions)})...")
    
    # Process one by one (or small batches) to avoid padding complexity with Unsloth's optimized RoPE
    # For distillation, correctness > speed, and unsloth is reasonably fast.
    
    # Open file in append mode once, or open repeatedly. 
    # Opening repeatedly is safer for crashes but slightly slower. Given the generation time is dominant, it's negligible.
    
    for instruction in tqdm(instructions_to_process):
        messages = [{"role": "user", "content": instruction}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Decode only the new tokens
        generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        new_entry = {
            "instruction": instruction,
            "input": "",
            "output": generated_text
        }

        # Incremental Save
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(new_entry) + "\n")

    print(f"✅ Distillation Complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    distill()
