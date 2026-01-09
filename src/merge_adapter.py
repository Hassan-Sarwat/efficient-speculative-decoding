import torch
import argparse
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model(model_type, scenario, category):
    # 1. Configuration Map
    # Define base models for each category
    BASE_MODELS = {
        "target": "Qwen/Qwen2.5-14B-Instruct",
        "draft": "Qwen/Qwen2.5-0.5B-Instruct"
    }

    # Construct Paths dynamically
    # Example: models/target_cod_easy
    adapter_name = f"{category}_{model_type}_{scenario}"
    adapter_path = os.path.join("models", adapter_name)
    
    # Example: models/merged/target_cod_easy
    output_dir = os.path.join("models", "merged", adapter_name)
    
    base_model_id = BASE_MODELS.get(category)
    if not base_model_id:
        raise ValueError(f"Invalid category: {category}. Must be 'target' or 'draft'.")

    print(f"\n{'='*60}")
    print(f"🔄 MERGE PROCESS STARTED")
    print(f"{'='*60}")
    print(f"🔹 Type:      {model_type.upper()}")
    print(f"🔹 Scenario:  {scenario.upper()}")
    print(f"🔹 Category:  {category.upper()}")
    print(f"🔹 Adapter:   {adapter_path}")
    print(f"🔹 Output:    {output_dir}")
    print(f"{'='*60}\n")

    # Check if adapter exists
    if not os.path.exists(adapter_path):
        print(f"❌ Error: Adapter path not found: {adapter_path}")
        return

    # 2. Load Base Model (CPU Offload for Safety)
    print(f"📦 Loading Base Model: {base_model_id}...")
    # We use device_map="cpu" to ensure the merge happens in system RAM 
    # to avoid OOM on the GPU, especially for the 14B model.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # 3. Load and Merge Adapter
    print(f"🔗 Loading LoRA Adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path, device_map="cpu")
        print("⚡ Merging weights (this may take a moment)...")
        model = model.merge_and_unload()
    except Exception as e:
        print(f"❌ Failed to load/merge adapter: {e}")
        return

    # 4. Save Merged Model
    print(f"💾 Saving merged model to {output_dir}...")
    if os.path.exists(output_dir):
        print(f"   (Overwriting existing directory)")
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Success! Model saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base models.")
    
    parser.add_argument("--type", type=str, required=True, choices=["cot", "cod"], 
                        help="Model type: 'cot' (Chain of Thought) or 'cod' (Chain of Draft)")
    
    parser.add_argument("--scenario", type=str, required=True, choices=["easy", "medium", "hard"], 
                        help="Difficulty scenario")
    
    parser.add_argument("--category", type=str, required=True, choices=["target", "draft"], 
                        help="Model category: 'target' (14B) or 'draft' (0.5B)")

    args = parser.parse_args()
    
    merge_model(args.type, args.scenario, args.category)