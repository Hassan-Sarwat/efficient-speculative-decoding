from unsloth import FastLanguageModel
from datasets import load_dataset
import json
import torch
from tqdm import tqdm

def distill():
    # Configuration
    MODEL_PATH = "/app/models/target"  # Must match final_save_path in target config
    INPUT_FILE = "data/cob_data.jsonl"
    OUTPUT_FILE = "data/cod_distilled_data.jsonl"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True

    print(f"🚀 Loading Target Model from {MODEL_PATH}...")
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

    original_instructions = dataset["instruction"]
    
    # Store results
    new_data = []

    print(f"🔥 Generating Responses for {len(original_instructions)} samples...")
    
    # Process one by one (or small batches) to avoid padding complexity with Unsloth's optimized RoPE
    # For distillation, correctness > speed, and unsloth is reasonably fast.
    
    for instruction in tqdm(original_instructions):
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
        
        new_data.append({
            "instruction": instruction,
            "input": "",
            "output": generated_text
        })

    print(f"💾 Saving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for entry in new_data:
            f.write(json.dumps(entry) + "\n")

    print("✅ Distillation Complete!")

if __name__ == "__main__":
    distill()
