# train.py - Unified Fine-tuning Script with Production Best Practices
#
# Stack: Unsloth (4-bit + LoRA) + trl 0.24 SFTTrainer + transformers 5.x.
#
# Key trl 0.24 changes from older code:
#   - DataCollatorForCompletionOnlyLM is removed.
#   - `tokenizer=` kwarg is replaced by `processing_class=`.
#   - `max_seq_length`, `dataset_text_field`, `packing` etc. moved from SFTTrainer
#     args to SFTConfig (which subclasses TrainingArguments).
#
# Unsloth 2026.4.8 + trl 0.24 caveat (unsloth-zoo #323): Unsloth's patched
# `sft_prepare_dataset` does NOT recognize a conversational `messages` column —
# it raises "You must specify a `formatting_func`". Workaround: pre-tokenize the
# dataset ourselves into `input_ids` + `labels` with -100 on user/system tokens,
# computing the assistant boundary by tokenizing the prompt (system+user +
# generation prompt) separately. Qwen3's stock chat template lacks `{% generation %}`
# markers so `return_assistant_tokens_mask` won't work — prefix-tokenization is the
# robust path. With `input_ids`+`labels` present, Unsloth picks
# `DataCollatorForSeq2Seq` (respects -100 padding) and skips its broken prep.

# Unsloth must be imported before transformers/trl/peft to enable its patches.
import unsloth  # noqa: F401  -- side-effect import: patches HF stack
from unsloth import FastLanguageModel

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import logging
import gc
from dataclasses import dataclass, field

import torch
import yaml
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import HfArgumentParser, TrainerCallback
from trl import SFTConfig, SFTTrainer
from answer_utils import FORMAT_SYSTEM_MESSAGE

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
    load_in_4bit: bool = field(default=False, metadata={"help": "Use 4-bit quantization"})
    # Note: `max_seq_length` lives on SFTConfig — Unsloth dynamically patches it
    # onto SFTConfig at import time (see unsloth/models/rl.py). Defining it here
    # too would register --max_seq_length twice on HfArgumentParser and crash.

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
    """Logs peak GPU memory at every logging step and tracks the run-wide peak."""
    def __init__(self):
        self.run_peak_gb = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            self.run_peak_gb = max(self.run_peak_gb, peak_mem)

            logs["gpu_peak_mem_gb"] = round(peak_mem, 2)
            logs["gpu_current_mem_gb"] = round(current_mem, 2)
            logs["gpu_run_peak_mem_gb"] = round(self.run_peak_gb, 2)

            torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            logger.info("=" * 60)
            logger.info(f"RUN-WIDE PEAK GPU MEMORY: {self.run_peak_gb:.2f} GB")
            logger.info("=" * 60)


class DatasetStatsCallback(TrainerCallback):
    """Logs dataset statistics at the start of training."""
    def __init__(self, train_dataset, eval_dataset=None):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            total = len(self.train_dataset) + len(self.eval_dataset)
            logger.info(f"Validation samples: {len(self.eval_dataset)}")
            logger.info(f"Val split ratio: {len(self.eval_dataset) / total:.2%}")
        else:
            logger.warning("No validation set - cannot monitor overfitting!")
        logger.info("=" * 60)


def to_messages(example):
    """
    Convert {instruction, output} rows into trl's conversational format.

    SFTTrainer applies the model's chat template automatically when it sees
    a `messages` column. With `assistant_only_loss=True`, loss is computed
    only on assistant tokens (replacing the V0 DataCollatorForCompletionOnlyLM).
    """
    instruction = example.get("instruction") or ""
    output = example.get("output") or ""
    return {
        "messages": [
            {"role": "system", "content": FORMAT_SYSTEM_MESSAGE},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    }


def filter_long_messages(dataset, tokenizer, max_length, label):
    """
    Drop samples whose rendered chat-template length exceeds `max_length`.

    Truncating completions silently destroys the assistant tokens we actually
    train on, so we drop oversize rows instead. Renders the chat template once
    per sample to compute true token length, logs the length distribution, and
    warns whenever any sample is dropped (zero-tolerance — see plan).
    """
    def render_length(messages):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return len(tokenizer.encode(text, add_special_tokens=False))

    lengths = [render_length(msgs) for msgs in dataset["messages"]]
    n = len(lengths)
    if n > 0:
        sorted_lengths = sorted(lengths)
        p50 = sorted_lengths[n // 2]
        p95 = sorted_lengths[min(int(n * 0.95), n - 1)]
        p99 = sorted_lengths[min(int(n * 0.99), n - 1)]
        mx = sorted_lengths[-1]
        logger.info(
            f"{label} length stats: p50={p50} p95={p95} p99={p99} max={mx} "
            f"(limit={max_length})"
        )

    keep_indices = [i for i, length in enumerate(lengths) if length <= max_length]
    original = len(dataset)
    dataset = dataset.select(keep_indices)
    dropped = original - len(dataset)
    drop_rate = dropped / original if original > 0 else 0.0
    logger.info(f"{label}: Dropped {dropped} samples ({drop_rate:.1%})")
    if dropped > 0:
        logger.warning(
            f"{label}: {dropped} sample(s) ({drop_rate:.1%}) exceed max_seq_length "
            f"({max_length}) and will be silently excluded from training. "
            f"Longest dropped row: {max(lengths)} tokens. Consider raising "
            f"max_seq_length if this drift grows."
        )
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, SFTConfig))

    # --- Robust Argument Parsing Logic ---
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
        config_path = sys.argv[1]
        logger.info(f"Loading configuration from {config_path}")

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

                    if isinstance(getattr(obj, attr_name), bool):
                        if val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                        else:
                            logger.warning(f"Warning: Expected boolean for {attr_name}, got {val}")

                    logger.info(f"Overriding {attr_name}: {getattr(obj, attr_name)} -> {val}")
                    setattr(obj, attr_name, val)

        override_arg("--data_file", "data_file", model_args)
        override_arg("--final_save_path", "final_save_path", model_args)
        override_arg("--wandb_project", "wandb_project", model_args)
        override_arg("--load_in_4bit", "load_in_4bit", model_args)
        override_arg("--output_dir", "output_dir", training_args)
        override_arg("--run_name", "run_name", training_args)
    else:
        # Standard CLI parsing
        model_args, training_args = parser.parse_args_into_dataclasses()

    # Resolve max_seq_length from SFTConfig. YAML's `max_seq_length` lands on
    # training_args via Unsloth's patch; fall back to max_length, then a sane
    # default if neither was set.
    max_seq_length = (
        getattr(training_args, "max_seq_length", None)
        or getattr(training_args, "max_length", None)
        or 4096
    )

    # WandB Login
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        logger.info("WandB authentication successful")
    else:
        logger.warning("WANDB_API_KEY not found - training will not be logged to WandB")

    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_args.model_name}")
    logger.info(f"Data: {model_args.data_file}")
    logger.info(f"Max Seq Length: {max_seq_length}")
    logger.info(f"4-bit Loading: {model_args.load_in_4bit}")
    logger.info(f"LoRA Rank: {model_args.lora_r}")
    logger.info(f"Val Split: {model_args.val_split_ratio:.1%}")
    logger.info("=" * 60)

    # Load Model
    logger.info(f"Loading Model: {model_args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=model_args.load_in_4bit,
    )

    # Apply LoRA
    logger.info(f"Applying LoRA (rank={model_args.lora_r})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=model_args.random_seed,
    )

    # Load Dataset
    logger.info(f"Loading Dataset from {model_args.data_file}")
    dataset = load_dataset("json", data_files=model_args.data_file, split="train")
    logger.info(f"Original dataset size: {len(dataset)} samples")

    # Train/Validation Split
    if model_args.val_split_ratio > 0:
        logger.info(f"Splitting dataset (val_ratio={model_args.val_split_ratio:.1%})")
        dataset_split = dataset.train_test_split(
            test_size=model_args.val_split_ratio,
            seed=model_args.random_seed,
            shuffle=True,
        )
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        logger.info(f"Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        logger.warning("No validation split - training without eval monitoring!")
        train_dataset = dataset
        eval_dataset = None

    # Convert to conversational format. SFTTrainer applies the chat template
    # automatically when it sees a `messages` column.
    logger.info("Converting dataset to conversational (messages) format...")
    train_dataset = train_dataset.map(
        to_messages, remove_columns=train_dataset.column_names
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            to_messages, remove_columns=eval_dataset.column_names
        )

    # Drop samples that exceed max_seq_length once rendered through the chat
    # template — see filter_long_messages() docstring for rationale.
    logger.info(f"Filtering samples > {max_seq_length} tokens...")
    train_dataset = filter_long_messages(
        train_dataset, tokenizer, max_seq_length, "Train"
    )
    if eval_dataset is not None:
        eval_dataset = filter_long_messages(
            eval_dataset, tokenizer, max_seq_length, "Val"
        )

    # Pre-tokenize: bypass the Unsloth 2026.4.8 SFTTrainer regression that doesn't
    # recognize a `messages` column (unsloth-zoo #323). We render the chat template,
    # tokenize, and build `labels` ourselves with -100 on user/system tokens so
    # loss is computed only on assistant tokens. With both `input_ids` and `labels`
    # present, Unsloth picks DataCollatorForSeq2Seq (respects -100 padding) instead
    # of its DataCollatorForLanguageModeling fallback (which would ignore masking).
    #
    # Qwen3's stock chat template has no `{% generation %}` markers, so we can't
    # rely on `return_assistant_tokens_mask`. Instead we tokenize the prompt
    # (user/system messages + generation prompt) separately and mask everything in
    # that prefix. This assumes single-turn examples (one user → one assistant),
    # which matches our processed CoT/CoD datasets.
    def tokenize_with_assistant_only_loss(example):
        messages = example["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        if len(prompt_messages) == len(messages):
            raise RuntimeError("Sample has no assistant message — cannot train.")

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "Tokenized prompt is not a prefix of full conversation — "
                "chat template is doing something unexpected at the prompt/"
                "assistant boundary. Check whitespace/special-token handling."
            )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "labels": labels}

    logger.info("Pre-tokenizing with assistant-only loss masking...")
    train_dataset = train_dataset.map(
        tokenize_with_assistant_only_loss,
        remove_columns=train_dataset.column_names,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize_with_assistant_only_loss,
            remove_columns=eval_dataset.column_names,
        )

    # Packing off; max_length kept for safety. We do NOT set assistant_only_loss=True
    # because trl's pre-flight check requires conversational data and would reject
    # our pre-tokenized dataset — we've baked assistant-only masking into `labels`.
    training_args.max_length = max_seq_length
    training_args.max_seq_length = max_seq_length
    training_args.packing = False

    logger.info("Initializing Trainer...")

    callbacks = [
        GPUMemoryCallback(),
        DatasetStatsCallback(train_dataset, eval_dataset),
    ]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=callbacks,
    )

    logger.info("Starting Training...")
    trainer.train()

    # Save LoRA Adapters
    if model_args.final_save_path:
        logger.info(f"Saving LoRA Adapters to {model_args.final_save_path}")
        os.makedirs(model_args.final_save_path, exist_ok=True)
        model.save_pretrained(model_args.final_save_path)
        tokenizer.save_pretrained(model_args.final_save_path)
        logger.info("Model saved successfully")
    else:
        logger.warning("No final_save_path provided - model not saved!")

    # Log final metrics
    if eval_dataset:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info("Validation metrics were logged during training")
        logger.info("Check WandB dashboard for eval_loss curves")
        logger.info("=" * 60)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("GPU memory cleared")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
