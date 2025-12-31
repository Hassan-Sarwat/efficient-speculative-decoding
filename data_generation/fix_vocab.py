import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# Paths
DRAFT_PATH = "models/draft"
TARGET_PATH = "models/target"
OUTPUT_PATH = "models/draft_padded"

def fix_vocab_mismatch():
    print(f"🔍 Checking vocab sizes...")
    
    # 1. Get Target Vocab Size from Config (Lightweight)
    with open(os.path.join(TARGET_PATH, "config.json"), "r") as f:
        target_config = json.load(f)
    target_vocab = target_config["vocab_size"]
    print(f"   Target (14B) Vocab: {target_vocab}")

    # 2. Load Draft Model
    print(f"⏳ Loading Draft Model from {DRAFT_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_PATH)
    
    current_vocab = model.config.vocab_size
    print(f"   Draft (0.5B) Vocab: {current_vocab}")

    # 3. Resize if needed
    if current_vocab != target_vocab:
        print(f"⚠️  Mismatch detected! Resizing {current_vocab} -> {target_vocab}...")
        
        # Resize embeddings and head
        model.resize_token_embeddings(target_vocab)
        
        # Explicitly zero out the new embedding rows to avoid noise
        # (Though tokenizer won't use them, it's good practice)
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        input_embeddings[current_vocab:] = 0
        output_embeddings[current_vocab:] = 0
        
        print(f"✅ Resized and zero-initialized padding tokens.")
        
        # 4. Save
        print(f"💾 Saving fixed model to {OUTPUT_PATH}...")
        model.save_pretrained(OUTPUT_PATH)
        tokenizer.save_pretrained(OUTPUT_PATH)
        print("🎉 Done! Use 'models/draft_padded' in your benchmark.")
    else:
        print("✅ Vocab sizes already match. No action needed.")

if __name__ == "__main__":
    fix_vocab_mismatch()