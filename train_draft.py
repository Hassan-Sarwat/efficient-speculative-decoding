import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Configuration
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "/app/models/draft"

def train_draft():
    print(f"🚀 Loading Draft Model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = False, # Small enough to load full usually, but unsloth handles it
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",
    )

    # Load SAME dataset as target to align their knowledge
    dataset = load_dataset("gsm8k", "main", split="train")
    
    def formatting_prompts_func(examples):
        texts = []
        for question, answer in zip(examples["question"], examples["answer"]):
            text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("🔥 Starting Draft Training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 8, # Higher batch size for small model
            gradient_accumulation_steps = 2,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "checkpoints_draft",
            optim = "adamw_8bit",
        ),
    )
    trainer.train()

    print("💾 Merging and Saving Draft Model...")
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "merged_16bit")
    print(f"✅ Draft Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_draft()