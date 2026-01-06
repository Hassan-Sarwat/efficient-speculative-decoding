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
from transformers import TrainerCallback
import torch

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

class GPUMemoryCallback(TrainerCallback):
    """
    A callback that logs the peak GPU memory usage at every logging step.
    Useful for tracking VRAM spikes during training.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            # Get peak memory since last reset
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) # Convert to GB
            
            # Add to the logs dict (which goes to WandB/Console)
            logs["gpu_peak_mem_gb"] = round(peak_mem, 2)
            
            # Reset stats so the next step captures only that step's usage
            torch.cuda.reset_peak_memory_stats()

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))

    if len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
        # 1. Load the YAML config
        config_path = sys.argv[1]
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # 2. Parse command line arguments (skipping the yaml filename)
        # allow_extra_keys=True ensures we don't crash on args not in the dataclass yet
        # We parse sys.argv[2:] because sys.argv[1] is the yaml file
        cli_args = parser.parse_args_into_dataclasses(args=sys.argv[2:], return_remaining_strings=True)[0]
        
        # 3. Update the config_dict with any non-default CLI values
        
        # BETTER APPROACH: Parse YAML first, then override with CLI
        # We parse the dict to get base objects
        model_args, training_args = parser.parse_dict(config_dict)
        
        # Now we re-parse CLI args to override the objects
        # Note: HfArgumentParser doesn't support easy "update" of dataclasses.
        # So we simply parse CLI args into a temporary set and manually update.
        
        temp_model_args, temp_training_args = parser.parse_args_into_dataclasses(args=sys.argv[2:])
        
        # Manually override fields that were explicitly set in CLI
        # (Heuristic: if the value in temp_args is different from default, we might want to use it.
        # But since we don't know "default" vs "set", we rely on the fact that you passed specific flags)
        
        # Actually, the cleanest fix for your specific setup is this:
        # Don't use parse_dict. Just load YAML, convert to argv-style list, and prepend to sys.argv.
        
        # Convert YAML dict back to CLI args style list
        yaml_args = []
        for k, v in config_dict.items():
            # Skip keys that might be complex objects if not needed, 
            # but for your config, they are simple strings/ints/bools
            if v is False: continue 
            if v is True: 
                yaml_args.append(f"--{k}")
                continue
            if isinstance(v, list):
                for item in v:
                    yaml_args.append(f"--{k}")
                    yaml_args.append(str(item))
                continue
            
            yaml_args.append(f"--{k}")
            yaml_args.append(str(v))
            
        # Combine: YAML args + CLI overrides (CLI comes last so it wins)
        # We remove the yaml file path from sys.argv
        combined_args = yaml_args + sys.argv[2:]
        model_args, training_args = parser.parse_args_into_dataclasses(args=combined_args)

    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # WANDB Login
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    
    logger.info(f"Loading Model: {model_args.model_name}")
    
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

    logger.info(f"Loading Dataset from {model_args.data_file}")
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

    logger.info("Starting Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        args=training_args,
        callbacks=[GPUMemoryCallback()]
    )
    trainer.train()

    # --- CHANGED: Save ONLY Adapters ---
    if model_args.final_save_path:
        logger.info(f"Saving LoRA Adapters to {model_args.final_save_path}")
        os.makedirs(model_args.final_save_path, exist_ok=True)
        # unsloth's save_pretrained saves the adapter config and weights
        model.save_pretrained(model_args.final_save_path)
        tokenizer.save_pretrained(model_args.final_save_path) 
    else:
        logger.warning("No final_save_path provided.")

if __name__ == "__main__":
    main()