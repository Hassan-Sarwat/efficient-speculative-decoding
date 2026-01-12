import argparse
import json
import os
import logging
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from generated text.
    Handles formats like: "#### 42", "The answer is 42", etc.
    """
    import re
    
    # Try #### format first (GSM8K standard)
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1).strip()
    
    # Try "answer is X" format
    match = re.search(r'(?:answer|result|solution)\s+is\s+(-?\d+(?:\.\d+)?)', text.lower())
    if match:
        return match.group(1).strip()
    
    # Try last number in text as fallback
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].strip()
    
    return ""


def validate_distillation_quality(
    instructions: List[str],
    outputs: List[Any],
    dataset: Any,
    threshold: float = 0.85
) -> float:
    """
    Validate that distilled outputs are correct by comparing to ground truth.
    Returns accuracy percentage.
    """
    logger.info("🔍 Validating distillation quality...")
    
    # Create lookup for ground truth answers
    ground_truth_map = {}
    for item in dataset:
        if "answer" in item:
            ground_truth_map[item["instruction"]] = str(item["answer"]).strip()
    
    if not ground_truth_map:
        logger.warning("⚠️ No ground truth answers found in dataset - skipping validation")
        return 1.0
    
    correct = 0
    total = 0
    
    for instruction, output in zip(instructions, outputs):
        if instruction not in ground_truth_map:
            continue
            
        generated_text = output.outputs[0].text
        generated_answer = extract_answer(generated_text)
        ground_truth = ground_truth_map[instruction]
        
        if generated_answer == ground_truth:
            correct += 1
        else:
            logger.debug(f"Mismatch - GT: {ground_truth}, Generated: {generated_answer}")
        
        total += 1
    
    if total == 0:
        logger.warning("⚠️ No answers could be validated")
        return 1.0
    
    accuracy = correct / total
    logger.info(f"✅ Distillation Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    if accuracy < threshold:
        raise ValueError(
            f"❌ Distillation quality too low ({accuracy:.2%} < {threshold:.0%}). "
            f"Target model may not be generating correct solutions. Check training."
        )
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Distill data using a target model")
    parser.add_argument("--base_model", type=str, required=True, 
                       help="Base HF model (e.g. Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--adapter_path", type=str, default=None, 
                       help="Path to LoRA adapter (optional)")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Path to input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Path to save distilled .jsonl file")
    
    # Sampling parameters (now configurable)
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling top_p (default: 0.9)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Max tokens to generate (default: 1024)")
    
    # Memory optimization
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for generation (reduce if OOM)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                       help="GPU memory utilization (default: 0.90)")
    
    # Validation
    parser.add_argument("--skip_validation", action="store_true",
                       help="Skip distillation quality validation")
    parser.add_argument("--validation_threshold", type=float, default=0.85,
                       help="Minimum accuracy threshold for distillation (default: 0.85)")
    
    args = parser.parse_args()

    logger.info(f"🚀 Initializing vLLM with Base: {args.base_model}")
    
    # Check if adapter exists
    use_adapter = args.adapter_path and os.path.exists(
        os.path.join(args.adapter_path, "adapter_config.json")
    )
    
    if args.adapter_path and not use_adapter:
        logger.warning(
            f"⚠️ Adapter path provided but invalid/empty: {args.adapter_path}. "
            f"Using Base Model only."
        )

    # ✅ FP16 Configuration for 24GB VRAM
    # Key strategies:
    # 1. No quantization (maintains quality)
    # 2. tensor_parallel_size=1 (single GPU)
    # 3. Lower GPU utilization (90% instead of 95%)
    # 4. Smaller max_model_len (reduces KV cache)
    # 5. enforce_eager=True (reduces memory fragmentation)
    llm = LLM(
        model=args.base_model,
        dtype="auto",
        quantization="bitsandbytes",     
        load_format="bitsandbytes",
        enable_lora=True if use_adapter else False,
        max_lora_rank=64,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Load Data
    logger.info(f"📥 Loading Dataset from {args.input_file}...")
    dataset = load_dataset("json", data_files=args.input_file, split="train")

    # Check Existing (resume capability)
    existing_instructions = set()
    if os.path.exists(args.output_file):
        logger.info(f"📂 Found existing output file, loading to resume...")
        with open(args.output_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_instructions.add(json.loads(line)["instruction"])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"⚠️ Skipping malformed line: {e}")

        logger.info(f"✅ Loaded {len(existing_instructions)} existing samples")

    # Prepare Prompts with robust error handling
    prompts = []
    instructions_map = []
    skipped_count = 0
    
    for item in dataset:
        try:
            inst = item.get("instruction", "").strip()
            
            # Validation checks
            if not inst:
                skipped_count += 1
                continue
            
            if inst in existing_instructions:
                continue
            
            # Format prompt using chat template
            messages = [{"role": "user", "content": inst}]
            prompt_text = llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            instructions_map.append(inst)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to process item: {e}")
            skipped_count += 1

    logger.info(
        f"📊 Prepared {len(prompts)} prompts, "
        f"skipped {skipped_count} (already exists or invalid)"
    )

    if not prompts:
        logger.info("✅ All items already processed. Nothing to do.")
        return

    # Setup LoRA request
    lora_request = None
    if use_adapter:
        logger.info(f"🔗 Loading LoRA adapter: {args.adapter_path}")
        lora_request = LoRARequest("custom_adapter", 1, args.adapter_path)

    # ✅ Batched Generation with Progress Bar and Checkpointing
    logger.info(f"🔄 Generating {len(prompts)} responses in batches of {args.batch_size}...")
    
    all_outputs = []
    checkpoint_interval = 50  # Save every 50 samples
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Open file in append mode for incremental saving
    with open(args.output_file, "a") as f:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Distilling"):
            batch_end = min(i + args.batch_size, len(prompts))
            batch_prompts = prompts[i:batch_end]
            batch_instructions = instructions_map[i:batch_end]
            
            # Generate batch
            try:
                batch_outputs = llm.generate(
                    batch_prompts, 
                    sampling_params, 
                    lora_request=lora_request
                )
            except Exception as e:
                logger.error(f"❌ Batch generation failed at index {i}: {e}")
                raise
            
            # Save batch immediately (checkpoint)
            for instruction, output in zip(batch_instructions, batch_outputs):
                generated_text = output.outputs[0].text
                new_entry = {
                    "instruction": instruction,
                    "input": "",
                    "output": generated_text
                }
                f.write(json.dumps(new_entry) + "\n")
            
            # Flush to disk
            f.flush()
            
            # Store for validation
            all_outputs.extend(batch_outputs)
            
            # Log progress
            if (i + args.batch_size) % checkpoint_interval == 0 or batch_end == len(prompts):
                logger.info(f"💾 Checkpoint: {batch_end}/{len(prompts)} samples saved")

    logger.info(f"✅ Saved {len(instructions_map)} new samples to {args.output_file}")

    # Validate distillation quality
    if not args.skip_validation:
        try:
            accuracy = validate_distillation_quality(
                instructions_map,
                all_outputs,
                dataset,
                threshold=args.validation_threshold
            )
            logger.info(f"🎯 Final Validation Accuracy: {accuracy:.2%}")
        except ValueError as e:
            logger.error(str(e))
            raise

    logger.info("🎉 Distillation completed successfully!")


if __name__ == "__main__":
    main()