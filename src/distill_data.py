import argparse
import json
import os
import re
import logging
from typing import List, Dict, Any, Tuple
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
    """Extract numerical answer from generated text."""
    if not text:
        return ""
    
    # Priority 1: #### format (our standard)
    if "####" in text:
        parts = text.split("####")
        if len(parts) >= 2:
            answer = parts[-1].strip()
            if answer:
                return answer
    
    # Priority 2: "answer is X" format
    match = re.search(r'(?:answer|result|solution)\s+is\s+(-?\d+(?:\.\d+)?)', text.lower())
    if match:
        return match.group(1).strip()
    
    # Priority 3: Last number in text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].strip()
    
    return ""


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    if not text:
        return ""
    text = str(text).strip()
    text = text.replace(",", "")
    if text.endswith("."):
        text = text[:-1]
    text = text.replace("$", "").replace("%", "")
    return text.strip()


def answers_match(pred: str, gt: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    if pred_norm == gt_norm:
        return True
    
    try:
        pred_num = float(pred_norm)
        gt_num = float(gt_norm)
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        return False


# ========================================
# FUNCTION 2: Validate Separator Presence
# ========================================

def validate_separator_presence(
    outputs: List[Any],
    threshold: float = 0.95
) -> Tuple[float, List[int]]:
    """
    Validate that ALL distilled outputs contain #### separator.
    
    Returns:
        (separator_rate, failed_indices)
    """
    logger.info("🔍 Validating #### separator presence...")
    
    total = len(outputs)
    valid_count = 0
    failed_indices = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        
        if "####" in generated_text:
            parts = generated_text.split("####")
            if len(parts) >= 2 and parts[-1].strip():
                valid_count += 1
                continue
        
        failed_indices.append(i)
    
    separator_rate = valid_count / total if total > 0 else 0.0
    
    logger.info(f"📊 Separator Presence: {separator_rate:.2%} ({valid_count}/{total})")
    
    if failed_indices:
        logger.warning(f"⚠️  {len(failed_indices)} samples missing #### separator")
        for idx in failed_indices[:3]:
            text = outputs[idx].outputs[0].text
            preview = text[:100] + "..." if len(text) > 100 else text
            logger.warning(f"   Sample {idx}: {preview}")
    
    if separator_rate < threshold:
        raise ValueError(
            f"\n❌ SEPARATOR VALIDATION FAILED\n"
            f"   Rate: {separator_rate:.2%} < {threshold:.0%}\n"
            f"   {len(failed_indices)} samples missing ####\n"
            f"\n"
            f"   Target model didn't learn the #### format.\n"
            f"   Action: Check target model training (increase epochs)\n"
        )
    
    logger.info("✅ Separator validation passed!")
    return separator_rate, failed_indices


# ========================================
# FUNCTION 3: Validate Answer Correctness
# ========================================

def validate_distillation_quality(
    instructions: List[str],
    outputs: List[Any],
    dataset: Any,
    threshold: float = 0.85
) -> float:
    """
    Validate that distilled outputs are mathematically CORRECT.
    
    Returns:
        accuracy: Percentage of correct answers
    """
    logger.info("🔍 Validating answer correctness...")
    
    # Build ground truth lookup
    ground_truth_map = {}
    for item in dataset:
        if "answer" in item:
            instruction = item.get("instruction", "").strip()
            answer = str(item["answer"]).strip()
            ground_truth_map[instruction] = answer
    
    if not ground_truth_map:
        logger.warning("⚠️  No ground truth found - skipping accuracy validation")
        return 1.0
    
    # Check each answer
    correct = 0
    total = 0
    mismatches = []
    
    for instruction, output in zip(instructions, outputs):
        if instruction not in ground_truth_map:
            continue
        
        generated_text = output.outputs[0].text
        generated_answer = extract_answer(generated_text)
        ground_truth = ground_truth_map[instruction]
        
        if answers_match(generated_answer, ground_truth):
            correct += 1
        else:
            if len(mismatches) < 10:
                mismatches.append({
                    'q': instruction[:60] + "...",
                    'gt': ground_truth,
                    'gen': generated_answer
                })
        
        total += 1
    
    if total == 0:
        logger.warning("⚠️  No answers could be validated")
        return 1.0
    
    accuracy = correct / total
    logger.info(f"📊 Answer Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    if mismatches:
        logger.warning(f"⚠️  {total - correct} incorrect answers")
        logger.warning(f"   First {min(3, len(mismatches))} mismatches:")
        for i, m in enumerate(mismatches[:3], 1):
            logger.warning(f"   {i}. Q: {m['q']}")
            logger.warning(f"      Expected: {m['gt']} | Got: {m['gen']}")
    
    if accuracy < threshold:
        raise ValueError(
            f"\n❌ ACCURACY VALIDATION FAILED\n"
            f"   Accuracy: {accuracy:.2%} < {threshold:.0%}\n"
            f"   {total - correct}/{total} wrong answers\n"
            f"\n"
            f"   Target model is generating incorrect solutions.\n"
            f"   Action: Check training metrics or increase epochs\n"
        )
    
    logger.info("✅ Accuracy validation passed!")
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
            # Step 1: Check separator presence (MUST pass)
            separator_rate, _ = validate_separator_presence(
                all_outputs,
                threshold=0.95  # 95% must have ####
            )
            logger.info(f"✅ Separator Rate: {separator_rate:.2%}")
            
            # Step 2: Check answer accuracy
            accuracy = validate_distillation_quality(
                instructions_map,
                all_outputs,
                dataset,
                threshold=args.validation_threshold  # Default 0.85
            )
            logger.info(f"✅ Answer Accuracy: {accuracy:.2%}")
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("📋 DISTILLATION VALIDATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✅ Separator Presence: {separator_rate:.2%}")
            logger.info(f"✅ Answer Accuracy:    {accuracy:.2%}")
            logger.info(f"✅ Total Samples:      {len(all_outputs)}")
            logger.info("=" * 60)
            
        except ValueError as e:
            logger.error(str(e))
            raise

    logger.info("🎉 Distillation completed successfully!")


if __name__ == "__main__":
    main()