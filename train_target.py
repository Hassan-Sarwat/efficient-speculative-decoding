import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
# SWITCHED TO 7B as discussed for better speedup demonstration
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct" 
OUTPUT_DIR = "/app/models/target"

def train_target():
    print(f"🚀 Loading Target Model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True, # Load in 4bit for faster training
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True, 
        random_state = 3407,
    )

    # Load Data (Assuming GSM8K or your generated 'Chain of Draft' data)
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Simple formatting function
    def formatting_prompts_func(examples):
        texts = []
        for question, answer in zip(examples["question"], examples["answer"]):
            # Standard Qwen Chat Format
            text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("🔥 Starting Training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Short run for demo/debugging
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "checkpoints",
            optim = "adamw_8bit",
        ),
    )
    trainer.train()

    print("💾 Merging and Saving Target Model...")
    # Merge LoRA back into base model for vLLM compatibility
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "merged_16bit")
    print(f"✅ Target Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    import torch
    train_target()