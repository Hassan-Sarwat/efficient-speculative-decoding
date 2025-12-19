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

# Setup Logging (Senior engineers don't use print for status)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "HuggingFace model ID"})
    output_dir: str = field(metadata={"help": "Where to save the merged model"})
    data_file: str = field(metadata={"help": "Path to local .jsonl file"})
    max_seq_length: int = field(default=2048)
    load_in_4bit: bool = field(default=True)
    is_draft_model: bool = field(default=False, metadata={"help": "Smaller batch size/config for draft?"})

def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
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
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
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
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
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

    logger.info(f"💾 Saving Merged Model to {model_args.output_dir}")
    # Ensure directory exists
    os.makedirs(model_args.output_dir, exist_ok=True)
    model.save_pretrained_merged(model_args.output_dir, tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    main()