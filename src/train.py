# train.py - Unified Fine-tuning Script with Production Best Practices
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

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class ModelArguments:
    """Arguments for model configuration and data paths."""
    model_name: str = field(metadata={"help": "HuggingFace model ID"})
    data_file: str = field(metadata={"help": "Path to local .jsonl file"})
    final_save_path: str = field(
        default=None, 
        metadata={"help": "Where to save the merged model (separate from checkpoints)"}
    )
    wandb_project: str = field(default="peft_cob", metadata={"help": "WandB Project Name"})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    load_in_4bit: bool = field(default=True, metadata={"help": "Use 4-bit quantization"})
    
    # LoRA Config
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0, metadata={"help": "LoRA dropout"})
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        metadata={"help": "Target modules for LoRA"}
    )
    
    # Data splitting
    val_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio (default: 0.1 = 10%)"}
    )
    random_seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})


class GPUMemoryCallback(TrainerCallback):
    """
    Logs peak GPU memory usage at every logging step.
    Critical for tracking VRAM usage and preventing OOM.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            # Get peak memory since last reset
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            current_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            
            # Add to logs (goes to WandB/Console)
            logs["gpu_peak_mem_gb"] = round(peak_mem, 2)
            logs["gpu_current_mem_gb"] = round(current_mem, 2)
            
            # Reset peak stats for next interval
            torch.cuda.reset_peak_memory_stats()


class DatasetStatsCallback(TrainerCallback):
    """
    Logs dataset statistics at the start of training.
    Helps verify data quality and catch issues early.
    """
    def __init__(self, train_dataset, eval_dataset=None):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
    
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("=" * 60)
        logger.info("📊 DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(self.train_dataset)}")
        
        if self.eval_dataset:
            logger.info(f"Validation samples: {len(self.eval_dataset)}")
            logger.info(f"Val split ratio: {len(self.eval_dataset) / (len(self.train_dataset) + len(self.eval_dataset)):.2%}")
        else:
            logger.warning("⚠️ No validation set - cannot monitor overfitting!")
        
        logger.info("=" * 60)


def formatting_prompts_func(examples, tokenizer):
    """
    Format examples using chat template.
    Handles edge cases and provides clear error messages.
    """
    texts = []
    skipped = 0
    
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # Safety checks
        if not instruction or not output:
            skipped += 1
            continue
        
        if not isinstance(instruction, str) or not isinstance(output, str):
            skipped += 1
            continue
        
        # Use tokenizer's chat template for consistency
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        
        try:
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
        except Exception as e:
            logger.warning(f"⚠️ Failed to format sample: {e}")
            skipped += 1
    
    if skipped > 0:
        logger.warning(f"⚠️ Skipped {skipped} invalid samples during formatting")
    
    return {"text": texts}


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))

    # --- Robust Argument Parsing Logic ---
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
        config_path = sys.argv[1]
        logger.info(f"📄 Loading configuration from {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Parse YAML config
        model_args, training_args = parser.parse_dict(config_dict)
        
        # Manual CLI overrides (to avoid argparse conflicts)
        def override_arg(flag, attr_name, obj):
            if flag in sys.argv:
                idx = sys.argv.index(flag)
                if idx + 1 < len(sys.argv):
                    val = sys.argv[idx + 1]
                    logger.info(f"🔧 Overriding {attr_name}: {getattr(obj, attr_name)} -> {val}")
                    setattr(obj, attr_name, val)
        
        # Apply overrides
        override_arg("--data_file", "data_file", model_args)
        override_arg("--final_save_path", "final_save_path", model_args)
        override_arg("--wandb_project", "wandb_project", model_args)
        override_arg("--output_dir", "output_dir", training_args)
        override_arg("--run_name", "run_name", training_args)
    else:
        # Standard CLI parsing
        model_args, training_args = parser.parse_args_into_dataclasses()

    # WandB Login
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        logger.info("✅ WandB authentication successful")
    else:
        logger.warning("⚠️ WANDB_API_KEY not found - training will not be logged to WandB")
    wandb.init()
    logger.info("=" * 60)
    logger.info(f"🚀 TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_args.model_name}")
    logger.info(f"Data: {model_args.data_file}")
    logger.info(f"Max Seq Length: {model_args.max_seq_length}")
    logger.info(f"4-bit Loading: {model_args.load_in_4bit}")
    logger.info(f"LoRA Rank: {model_args.lora_r}")
    logger.info(f"Val Split: {model_args.val_split_ratio:.1%}")
    logger.info("=" * 60)
    
    # Load Model
    logger.info(f"📥 Loading Model: {model_args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=model_args.load_in_4bit,
    )

    # Apply LoRA
    logger.info(f"🔧 Applying LoRA (rank={model_args.lora_r})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # ✅ Use unsloth's optimized version
        random_state=model_args.random_seed,
    )

    # Load Dataset
    logger.info(f"📥 Loading Dataset from {model_args.data_file}")
    dataset = load_dataset("json", data_files=model_args.data_file, split="train")
    
    logger.info(f"📊 Original dataset size: {len(dataset)} samples")

    # ✅ Train/Validation Split
    if model_args.val_split_ratio > 0:
        logger.info(f"🔀 Splitting dataset (val_ratio={model_args.val_split_ratio:.1%})")
        dataset_split = dataset.train_test_split(
            test_size=model_args.val_split_ratio,
            seed=model_args.random_seed,
            shuffle=True
        )
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        
        logger.info(f"✅ Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        logger.warning("⚠️ No validation split - training without eval monitoring!")
        train_dataset = dataset
        eval_dataset = None

    # Format datasets
    logger.info("🔄 Formatting datasets with chat template...")
    train_dataset = train_dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda examples: formatting_prompts_func(examples, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names
        )

    # Training
    logger.info("🏋️ Initializing Trainer...")
    
    callbacks = [
        GPUMemoryCallback(),
        DatasetStatsCallback(train_dataset, eval_dataset)
    ]
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # ✅ Enable validation
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        args=training_args,
        callbacks=callbacks
    )

    logger.info("🚀 Starting Training...")
    trainer.train()

    # Save LoRA Adapters
    if model_args.final_save_path:
        logger.info(f"💾 Saving LoRA Adapters to {model_args.final_save_path}")
        os.makedirs(model_args.final_save_path, exist_ok=True)
        model.save_pretrained(model_args.final_save_path)
        tokenizer.save_pretrained(model_args.final_save_path)
        logger.info("✅ Model saved successfully")
    else:
        logger.warning("⚠️ No final_save_path provided - model not saved!")

    # ✅ Log final metrics
    if eval_dataset:
        logger.info("📊 Running final evaluation...")
        final_metrics = trainer.evaluate()
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION METRICS")
        logger.info("=" * 60)
        for key, value in final_metrics.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 60)

    logger.info("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()