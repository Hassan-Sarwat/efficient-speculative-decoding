# finetune.py (Replaces train_target.py and train_draft.py)
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser
import yaml
from dotenv import load_dotenv
import wandb

# Setup Logging (Senior engineers don't use print for status)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "HuggingFace model ID"})
    data_file: str = field(metadata={"help": "Path to local .jsonl file"})
    final_save_path: str = field(default=None, metadata={"help": "Where to save the merged model (separate from checkpoints)"})
    max_seq_length: int = field(default=2048)
    load_in_4bit: bool = field(default=True)
    is_draft_model: bool = field(default=False, metadata={"help": "Smaller batch size/config for draft?"})
    
    # LoRA Config (Moved from hardcoded to Config)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0)
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

def main():
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
        config_path = sys.argv[1]
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Parse dict into dataclasses
        parser = HfArgumentParser((ModelArguments, TrainingArguments))
        model_args, training_args = parser.parse_dict(config_dict)
    else:
        # Fallback to CLI args (good for quick debugging)
        parser = HfArgumentParser((ModelArguments, TrainingArguments))
        model_args, training_args = parser.parse_args_into_dataclasses()

    # WANDB Login
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    
    logger.info(f"🚀 Loading Model: {model_args.model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        dtype=None,
        load_in_4bit=model_args.load_in_4bit,
    )

    # Standardize LoRA config (Single source of truth)
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    logger.info(f"📂 Loading Dataset from {model_args.data_file}")
    dataset = load_dataset("json", data_files=model_args.data_file, split="train")
    
    # Use a robust formatting function handling edge cases
    def formatting_prompts_func(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            # Safety check for None values
            if not instruction or not output: 
                continue
            
            # Use the tokenizer's chat template for consistency
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    logger.info("🔥 Starting Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        args=training_args,
    )
    trainer.train()

    if model_args.final_save_path:
        logger.info(f"💾 Saving Merged Model to {model_args.final_save_path}")
        # Ensure directory exists
        os.makedirs(model_args.final_save_path, exist_ok=True)
        model.save_pretrained_merged(model_args.final_save_path, tokenizer, save_method="merged_16bit")
    else:
        logger.warning("⚠️ No final_save_path provided. Model checkpoints are in output_dir, but merged model is NOT saved.")

if __name__ == "__main__":
    main()