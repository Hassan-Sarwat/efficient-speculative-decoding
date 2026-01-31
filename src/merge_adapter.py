#!/usr/bin/env python3
"""
merge_adapter.py - Merge LoRA Adapter into Base Model

This script merges a LoRA adapter back into its base model, creating
a standalone model file that can be used directly for inference without
needing to load adapters at runtime.

Usage:
    python src/merge_adapter.py \
        --base_model "Qwen/Qwen3-14B" \
        --adapter_path "models/target_cot_easy" \
        --output_path "models/target_cot_easy_merged"
"""

import argparse
import logging
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_lora_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """
    Merge a LoRA adapter into its base model and save the result.
    
    Args:
        base_model_path: HuggingFace model ID or path to base model
        adapter_path: Path to LoRA adapter directory
        output_path: Path where merged model will be saved
    """
    logger.info("=" * 60)
    logger.info("MERGING LORA ADAPTER")
    logger.info("=" * 60)
    logger.info(f"Base Model:    {base_model_path}")
    logger.info(f"Adapter Path:  {adapter_path}")
    logger.info(f"Output Path:   {output_path}")
    logger.info("=" * 60)
    
    # Validate adapter exists
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(
            f"Adapter config not found at {adapter_config}. "
            f"Make sure the adapter was trained and saved correctly."
        )
    
    logger.info("Loading base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # Use FP16 to reduce memory
            device_map="auto",          # Automatically distribute across GPUs
            trust_remote_code=True,
        )
        logger.info(f"Base model loaded: {base_model.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise
    
    logger.info("Loading LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.float16,
        )
        logger.info("LoRA adapter loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise
    
    logger.info("Merging adapter into base model...")
    try:
        merged_model = model.merge_and_unload()
        logger.info("Merge completed successfully")
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise
    
    # ensure model is saved as fp16
    logger.info("Converting merged model to fp16...")
    merged_model = merged_model.to(dtype=torch.float16)
    
    logger.info(f"Saving merged model to {output_path}...")
    try:
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,  # Use safetensors format
        )
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Save failed: {e}")
        raise
    
    logger.info("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(output_path)
        logger.info("Tokenizer saved successfully")
    except Exception as e:
        logger.error(f"Tokenizer save failed: {e}")
        raise
    
    # Calculate disk space used
    total_size = 0
    for root, dirs, files in os.walk(output_path):
        total_size += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    
    size_gb = total_size / (1024 ** 3)
    
    logger.info("=" * 60)
    logger.info("MERGE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Output Location: {output_path}")
    logger.info(f"Disk Space Used: {size_gb:.2f} GB")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next Steps:")
    logger.info(f"  1. Use this model directly in vLLM:")
    logger.info(f'     llm = LLM(model="{output_path}")')
    logger.info(f"  2. No need to load LoRA adapters at runtime")
    logger.info(f"  3. Delete adapter-only files to save space if needed")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge target model adapter
  python src/merge_adapter.py \\
      --base_model "Qwen/Qwen3-14B" \\
      --adapter_path "models/target_cot_easy" \\
      --output_path "models/target_cot_easy_merged"
  
  # Merge draft model adapter
  python src/merge_adapter.py \\
      --base_model "Qwen/Qwen3-0.6B" \\
      --adapter_path "models/draft_cot_easy" \\
      --output_path "models/draft_cot_easy_merged"
        """
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HuggingFace model ID or path to base model"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to directory containing LoRA adapter files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where merged model will be saved"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory without asking"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.adapter_path):
        logger.error(f"Adapter path does not exist: {args.adapter_path}")
        return 1
    
    if os.path.exists(args.output_path):
        if not args.force:
            logger.warning(f"Output path already exists: {args.output_path}")
            response = input("Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                logger.info("Aborted by user")
                return 0
        else:
            logger.info(f"Output path exists, overwriting due to --force: {args.output_path}")
    
    try:
        merge_lora_adapter(args.base_model, args.adapter_path, args.output_path)
        return 0
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
