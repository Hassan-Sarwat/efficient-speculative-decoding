import os
import json
from datasets import load_dataset

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving {len(data)} samples to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def prepare_easy():
    print("--- Processing Easy (GSM8K) ---")
    try:
        # Try loading test split directly
        dataset = load_dataset("gsm8k", "main", split="test")
    except ValueError:
        print("⚠️ 'test' split not found via direct load. Checking keys...")
        ds_dict = load_dataset("gsm8k", "main")
        if 'test' in ds_dict:
            dataset = ds_dict['test']
        else:
            print("Fallback to train split (WARNING)")
            dataset = ds_dict['train']
            
    # Normalize to list of dicts
    samples = [{"question": row['question'], "answer": row['answer']} for row in dataset]
    
    # Limit
    if len(samples) > 1000:
        print(f"Limiting Easy samples from {len(samples)} to 1000.")
        samples = samples[:1000]
        
    save_jsonl(samples, "data/tests/easy_test.jsonl")

def prepare_math(scenario, levels, types, raw_exclusion_path):
    print(f"--- Processing {scenario.capitalize()} (Competition Math) ---")
    
    # 1. Load Exclusion Set (Raw Data used for Training)
    exclusion_problems = set()
    if os.path.exists(raw_exclusion_path):
        print(f"Loading exclusion data from {raw_exclusion_path}...")
        with open(raw_exclusion_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    p = item.get('problem') or item.get('question')
                    if p:
                        exclusion_problems.add(p.strip())
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(exclusion_problems)} samples to exclude.")
    else:
        print(f"⚠️ Warning: Exclusion file {raw_exclusion_path} not found. No samples will be excluded.")

    # 2. Load Dataset (Train Split as requested to simulate test set from remaining data)
    dataset = load_dataset("qwedsacf/competition_math", split="train")
    
    filtered_samples = []
    skipped_exclusion = 0
    skipped_filter = 0
    
    for row in dataset:
        if len(filtered_samples) >= 1000:
            break
            
        # Check Level & Type
        if row['level'] not in levels or row['type'] not in types:
            skipped_filter += 1
            continue
            
        prob = row['problem'].strip()
        
        # Check Exclusion
        if prob in exclusion_problems:
            skipped_exclusion += 1
            continue
            
        filtered_samples.append({
            "question": row['problem'], 
            "answer": row['solution'],
            "level": row['level'],
            "type": row['type']
        })
        
    print(f"Skipped {skipped_filter} by filter, {skipped_exclusion} by exclusion.")
    print(f"Remaining valid test samples: {len(filtered_samples)}")
    save_jsonl(filtered_samples, f"data/tests/{scenario}_test.jsonl")

def main():
    # Easy
    prepare_easy()
    
    # Common Types for Medium/Hard
    math_types = ["Algebra", "Intermediate Algebra", "Precalculus"]
    
    # Medium
    prepare_math(
        "medium", 
        levels=["Level 1", "Level 2"], 
        types=math_types,
        raw_exclusion_path="data/raw/qwedsacf_competition_math_medium.jsonl"
    )
    
    # Hard
    prepare_math(
        "hard", 
        levels=["Level 3", "Level 4"], 
        types=math_types,
        raw_exclusion_path="data/raw/qwedsacf_competition_math_hard.jsonl"
    )
    print("\n✅ Test data preparation complete!")

if __name__ == "__main__":
    main()
