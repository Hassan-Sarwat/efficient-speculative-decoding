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
    validate_has_separator
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
    outputs: List[Any],
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
    
    total = len(outputs)
    valid_count = 0
    failed_indices = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        
        # Use shared validation function
        if validate_has_separator(generated_text, scenario):
            valid_count += 1
        else:
            failed_indices.append(i)
    
    separator_rate = valid_count / total if total > 0 else 0.0
    
    logger.info(f"Separator Presence: {separator_rate:.2%} ({valid_count}/{total})")
    
    if failed_indices:
        logger.warning(f"{len(failed_indices)} samples missing separator")
        for idx in failed_indices[:3]:
            text = outputs[idx].outputs[0].text
            preview = text[:100] + "..." if len(text) > 100 else text
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
    outputs: List[Any],
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
    
    for instruction, output in zip(instructions, outputs):
        if instruction not in ground_truth_map:
            continue
        
        generated_text = output.outputs[0].text
        
        # Use shared extraction function
        generated_answer = extract_answer(generated_text, scenario)
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
    
    logger.info(f"Initializing vLLM with INT8 quantization...")
    
    llm = LLM(
        model=args.base_model,
        dtype="float16",  
        enable_lora=True,
        max_loras=1,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        enforce_eager=True,
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
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
    )
    
    # Generate in batches
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    all_outputs = []
    checkpoint_interval = 100
    
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
            all_outputs.extend(batch_outputs)
            
            if (i + args.batch_size) % checkpoint_interval == 0 or batch_end == len(prompts):
                logger.info(f"Checkpoint: {batch_end}/{len(prompts)} samples saved")
    
    logger.info(f"Saved {len(instructions_map)} new samples to {args.output_file}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("GPU memory cleared after distillation")

    
    # Validate distillation quality
    if not args.skip_validation:
        try:
            logger.info("")
            logger.info("=" * 60)
            logger.info("VALIDATING DISTILLATION QUALITY")
            logger.info("=" * 60)
            
            # Step 1: Validate separator presence
            separator_rate, _ = validate_separator_presence(
                all_outputs,
                scenario=args.scenario,
                threshold=0.95
            )
            
            # Step 2: Validate answer correctness
            accuracy = validate_distillation_quality(
                instructions_map,
                all_outputs,
                dataset,
                scenario=args.scenario,
                threshold=args.validation_threshold
            )
            
            # Summary
            logger.info("")
            logger.info("=" * 60)
            logger.info("DISTILLATION VALIDATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Separator Presence: {separator_rate:.2%}")
            logger.info(f"Answer Accuracy:    {accuracy:.2%}")
            logger.info(f"Total Samples:      {len(all_outputs)}")
            logger.info("=" * 60)
            
        except ValueError as e:
            logger.error(str(e))
            raise
    
    logger.info("Distillation completed successfully!")


if __name__ == "__main__":
    main()