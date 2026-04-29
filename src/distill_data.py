import argparse
import json
import os
import logging
from typing import List, Any, Tuple
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from tqdm import tqdm
import gc
import torch
from answer_utils import (
    extract_answer,
    check_equality,
    validate_has_separator,
    classify_extraction_method,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================
# VALIDATION FUNCTIONS
# ========================================

def validate_separator_presence(
    texts: List[str],
    scenario: str = "easy",
    threshold: float = 0.95
) -> Tuple[float, List[int]]:
    """
    Validate that ALL distilled outputs contain the expected separator.
    Uses answer_utils.validate_has_separator() for format checking.

    Returns:
        (separator_rate, failed_indices)
    """
    logger.info("Validating separator presence...")

    total = len(texts)
    valid_count = 0
    failed_indices = []

    for i, text in enumerate(texts):
        if validate_has_separator(text, scenario):
            valid_count += 1
        else:
            failed_indices.append(i)

    separator_rate = valid_count / total if total > 0 else 0.0

    logger.info(f"Separator Presence: {separator_rate:.2%} ({valid_count}/{total})")

    if failed_indices:
        logger.warning(f"{len(failed_indices)} samples missing separator")
        for idx in failed_indices[:3]:
            preview = texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx]
            logger.warning(f"   Sample {idx}: {preview}")
    
    if separator_rate < threshold:
        raise ValueError(
            f"\nSEPARATOR VALIDATION FAILED\n"
            f"   Rate: {separator_rate:.2%} < {threshold:.0%}\n"
            f"   {len(failed_indices)} samples missing separator\n"
            f"\n"
            f"   Target model didn't learn the format.\n"
            f"   Action: Check target model training (increase epochs)\n"
        )
    
    logger.info("Separator validation passed!")
    return separator_rate, failed_indices


def validate_distillation_quality(
    instructions: List[str],
    texts: List[str],
    dataset: Any,
    scenario: str = "easy",
    threshold: float = 0.85
) -> float:
    """
    Validate that distilled outputs are mathematically CORRECT.
    Uses answer_utils functions for robust extraction and comparison.
    
    Returns:
        accuracy: Percentage of correct answers
    """
    logger.info("Validating answer correctness...")
    
    # Build ground truth lookup
    ground_truth_map = {}
    for item in dataset:
        instruction = item.get("instruction", "").strip()
        
        # Try to get answer field first
        if "answer" in item:
            answer = str(item["answer"]).strip()
        elif "output" in item:
            # Extract answer from output field using shared function
            answer = extract_answer(item["output"], scenario)
        else:
            continue
        
        if answer:
            ground_truth_map[instruction] = answer
    
    if not ground_truth_map:
        logger.warning("No ground truth found - skipping accuracy validation")
        return 1.0
    
    # Check each answer
    correct = 0
    total = 0
    mismatches = []
    
    for instruction, text in zip(instructions, texts):
        if instruction not in ground_truth_map:
            continue

        # Use shared extraction function
        generated_answer = extract_answer(text, scenario)
        ground_truth = ground_truth_map[instruction]
        
        # Use shared comparison function
        if check_equality(generated_answer, ground_truth):
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
        logger.warning("No answers could be validated")
        return 1.0
    
    accuracy = correct / total
    logger.info(f"Answer Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    if mismatches:
        logger.warning(f"{total - correct} incorrect answers")
        logger.warning(f"   First {min(3, len(mismatches))} mismatches:")
        for i, m in enumerate(mismatches[:3], 1):
            logger.warning(f"   {i}. Q: {m['q']}")
            logger.warning(f"      Expected: {m['gt']} | Got: {m['gen']}")
    
    if accuracy < threshold:
        raise ValueError(
            f"\nACCURACY VALIDATION FAILED\n"
            f"   Accuracy: {accuracy:.2%} < {threshold:.0%}\n"
            f"   {total - correct}/{total} wrong answers\n"
            f"\n"
            f"   Target model is generating incorrect solutions.\n"
            f"   Action: Check training metrics or increase epochs\n"
        )
    
    logger.info("Accuracy validation passed!")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Distill training data from fine-tuned model")
    
    parser.add_argument("--base_model", type=str, required=True, help="Base model HF ID")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl file")
    parser.add_argument("--scenario", type=str, default="easy", 
                       choices=["easy", "medium", "hard"],
                       help="Dataset scenario for answer extraction")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--skip_validation", action="store_true", help="Skip quality validation")
    parser.add_argument("--validation_threshold", type=float, default=0.85,
                       help="Minimum accuracy threshold (default: 0.85)")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("DISTILLATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Base Model:   {args.base_model}")
    logger.info(f"Adapter:      {args.adapter_path}")
    logger.info(f"Input:        {args.input_file}")
    logger.info(f"Output:       {args.output_file}")
    logger.info(f"Scenario:     {args.scenario}")
    logger.info(f"Batch Size:   {args.batch_size}")
    logger.info("=" * 60)
    
    # Check for existing progress
    existing_instructions = set()
    if os.path.exists(args.output_file):
        logger.info(f"Found existing output file, resuming...")
        with open(args.output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "instruction" in data:
                            existing_instructions.add(data["instruction"])
                    except:
                        pass
        logger.info(f"Already processed {len(existing_instructions)} items")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.input_file}...")
    dataset = load_dataset("json", data_files=args.input_file, split="train")
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    logger.info("Initializing vLLM with FP16 precision...")
    
    llm = LLM(
        model=args.base_model,
        dtype="float16",
        enable_lora=True,
        max_loras=1,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        tensor_parallel_size=1,
    )
    
    tokenizer = llm.get_tokenizer()
    
    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from {args.adapter_path}...")
    lora_request = LoRARequest("target_adapter", 1, args.adapter_path)
    
    # Prepare prompts
    prompts = []
    instructions_map = []
    
    for item in dataset:
        inst = item.get("instruction", "").strip()
        if inst and inst not in existing_instructions:
            messages = [{"role": "user", "content": inst}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            instructions_map.append(inst)
    
    if not prompts:
        logger.info("All items already processed!")
        return
    
    logger.info(f"Generating {len(prompts)} responses...")
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )
    
    # Generate in batches
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    checkpoint_interval = 100
    last_checkpoint = 0

    with open(args.output_file, "a") as f:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Distilling"):
            batch_end = min(i + args.batch_size, len(prompts))
            batch_prompts = prompts[i:batch_end]
            batch_instructions = instructions_map[i:batch_end]

            batch_outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)

            for instruction, output in zip(batch_instructions, batch_outputs):
                generated_text = output.outputs[0].text
                new_entry = {
                    "instruction": instruction,
                    "input": "",
                    "output": generated_text
                }
                f.write(json.dumps(new_entry) + "\n")

            f.flush()

            if batch_end - last_checkpoint >= checkpoint_interval or batch_end == len(prompts):
                logger.info(f"Checkpoint: {batch_end}/{len(prompts)} samples saved")
                last_checkpoint = batch_end
    
    logger.info(f"Saved {len(instructions_map)} new samples to {args.output_file}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("GPU memory cleared after distillation")

    
    # Validate distillation quality on the COMPLETE output file (handles resume mode).
    # We re-load from disk so validation always covers every distilled sample, not just
    # those generated in the current run.
    if not args.skip_validation:
        try:
            logger.info("")
            logger.info("=" * 60)
            logger.info("VALIDATING DISTILLATION QUALITY")
            logger.info("=" * 60)

            logger.info("Loading complete distilled dataset for validation...")
            all_instructions: List[str] = []
            all_texts: List[str] = []
            with open(args.output_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_instructions.append(data["instruction"])
                        all_texts.append(data["output"])
            logger.info(f"Validating {len(all_texts)} total samples")

            # Step 1: Validate separator presence
            separator_rate, _ = validate_separator_presence(
                all_texts,
                scenario=args.scenario,
                threshold=0.95
            )

            # Step 2: Validate answer correctness
            accuracy = validate_distillation_quality(
                all_instructions,
                all_texts,
                dataset,
                scenario=args.scenario,
                threshold=args.validation_threshold
            )

            # Step 3: Track which extraction path was used per sample.
            # High last-number-fallback rate = poor format compliance =
            # extracted answers may be catching number contamination.
            method_counts = {"boxed": 0, "separator": 0, "last_number": 0, "empty": 0}
            for text in all_texts:
                method = classify_extraction_method(text, args.scenario)
                method_counts[method] = method_counts.get(method, 0) + 1
            total = len(all_texts) if all_texts else 1
            fallback_rate = method_counts["last_number"] / total
            FALLBACK_WARN = 0.10

            # Summary
            logger.info("")
            logger.info("=" * 60)
            logger.info("DISTILLATION VALIDATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Separator Presence: {separator_rate:.2%}")
            logger.info(f"Answer Accuracy:    {accuracy:.2%}")
            logger.info(f"Total Samples:      {len(all_texts)}")
            logger.info("Extraction breakdown:")
            for method in ("boxed", "separator", "last_number", "empty"):
                count = method_counts.get(method, 0)
                logger.info(f"   {method:14s} {count:6d} ({count/total:.1%})")
            if fallback_rate > FALLBACK_WARN:
                logger.warning(
                    f"Last-number fallback fired on {fallback_rate:.1%} of samples "
                    f"(threshold {FALLBACK_WARN:.0%}). Extracted accuracy may be "
                    f"inflated by number-contamination matches; investigate format compliance."
                )
            logger.info("=" * 60)

        except ValueError as e:
            logger.error(str(e))
            raise
    
    logger.info("Distillation completed successfully!")


if __name__ == "__main__":
    main()