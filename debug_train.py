import json
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

# Load your exact training data
data_file = "data/processed/cod_medium.jsonl"
dataset = load_dataset("json", data_files=data_file, split="train")

# Split same way as training (10% val)
dataset_split = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
train_dataset = dataset_split["train"]

print(f"Total training samples: {len(train_dataset)}")

# Calculate which samples are in step 35
# batch_size=2, grad_accum=4, so effective_batch=8
effective_batch_size = 8
step_35_start = 35 * effective_batch_size  # 280
step_35_end = step_35_start + effective_batch_size  # 288

print(f"\nStep 35 processes samples {step_35_start} to {step_35_end-1}")
print("=" * 60)

# Analyze those specific samples
for idx in range(step_35_start, min(step_35_end, len(train_dataset))):
    item = train_dataset[idx]
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    
    # Format exactly as training does
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    tokens = tokenizer.encode(text)
    length = len(tokens)
    
    print(f"\nSample {idx}:")
    print(f"  Token length: {length}")
    print(f"  Instruction preview: {instruction[:100]}...")
    print(f"  Output preview: {output[:100]}...")
    
    if length > 1536:
        print(f"  ⚠️  EXCEEDS 1536 tokens!")
    if length > 2048:
        print(f"  ❌ EXCEEDS 2048 tokens!")

# Also check overall distribution
all_lengths = []
for item in train_dataset:
    messages = [
        {"role": "user", "content": item.get("instruction", "")},
        {"role": "assistant", "content": item.get("output", "")}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    all_lengths.append(len(tokenizer.encode(text)))

print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)
print(f"Max length: {max(all_lengths)}")
print(f"95th percentile: {sorted(all_lengths)[int(len(all_lengths)*0.95)]}")
print(f"Samples > 1536: {sum(1 for l in all_lengths if l > 1536)}")
print(f"Samples > 2048: {sum(1 for l in all_lengths if l > 2048)}")