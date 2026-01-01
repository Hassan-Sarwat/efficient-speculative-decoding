import argparse
import logging
import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_generation.dataset_loader import load_and_filter_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def extract_steps(text: str) -> List[str]:
    """Extracts reasoning steps from text."""
    # Pattern 1: Step 1:, Step 2: etc.
    steps = re.split(r'Step \d+:', text)
    if len(steps) > 1:
        return [s.strip() for s in steps if s.strip()]
    
    # Pattern 2: <thought> tags (treating content as one big step unless internally structured, 
    # but specific request was for CoT/CoD step metrics. CoT usually has steps. CoD might be a summary.)
    # If no explicit steps, treat as single block or split by newlines?
    # Let's try splitting by double newlines as a fallback for structural breaks
    return [s.strip() for s in text.split('\n\n') if s.strip()]

def get_word_count(text: str) -> int:
    return len(text.split())

def main():
    parser = argparse.ArgumentParser(description="Analyze generated datasets")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="Dataset name")
    parser.add_argument("--suffix", type=str, help="File suffix used during generation")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where data is stored")
    args = parser.parse_args()

    safe_name = args.dataset.replace("/", "_")
    output_dir = Path(args.output_dir)

    # 1. Check Dataset File
    if args.suffix:
        dataset_filename = f"{safe_name}_{args.suffix}.jsonl"
    else:
        dataset_filename = f"{safe_name}.jsonl"
    
    dataset_path = output_dir / dataset_filename
    
    if not dataset_path.exists():
        logger.error(f"Dataset file missing: {dataset_path}")
        print(f"\nPlease run launch_generation.py to download and cache the dataset:")
        suffix_arg = f"--file_suffix {args.suffix}" if args.suffix else ""
        print(f"  python data_generation/launch_generation.py --dataset {args.dataset} {suffix_arg} --dry-run")
        sys.exit(1)
    
    logger.info(f"Found dataset file: {dataset_path}")

    # 2. Check CoT File
    if args.suffix:
        cot_filename = f"cot_data_{safe_name}_{args.suffix}.jsonl"
    else:
        cot_filename = f"cot_data_{safe_name}.jsonl"
    
    cot_path = output_dir / cot_filename

    if not cot_path.exists():
        logger.error(f"CoT file missing: {cot_path}")
        print(f"\nPlease generate CoT data:")
        suffix_arg = f"--file_suffix {args.suffix}" if args.suffix else ""
        print(f"  python data_generation/launch_generation.py --dataset {args.dataset} {suffix_arg} --limit 100")
        sys.exit(1)
        
    logger.info(f"Found CoT file: {cot_path}")

    # 3. Check CoD File
    if args.suffix:
        cod_filename = f"cob_data_{safe_name}_{args.suffix}.jsonl"
    else:
        cod_filename = f"cob_data_{safe_name}.jsonl"

    cod_path = output_dir / cod_filename

    if not cod_path.exists():
        logger.error(f"CoD file missing: {cod_path}")
        print(f"\nPlease process results to generate CoD data:")
        suffix_arg = f"--file_suffix {args.suffix}" if args.suffix else ""
        print(f"  python data_generation/process_results.py --dataset {args.dataset} {suffix_arg} --output_dir {args.output_dir}")
        sys.exit(1)

    logger.info(f"Found CoD file: {cod_path}")
    logger.info("All files present. Starting analysis...")

    # Load Data
    
    # Load Source Dataset
    # We use our loader to get it consistently, though we know the path exists now.
    # We set filters=None because the cached file should already be filtered if it was created by us.
    # But if we just load from disk directly it is faster.
    source_data = {}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Normalize to find key
            q = item.get("question") or item.get("problem") or item.get("instruction")
            if q:
                source_data[q.strip()] = item

    # Load CoT
    cot_data = {}
    with open(cot_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            q = item.get("instruction", "").strip()
            if q:
                cot_data[q] = item

    # Load CoD
    cod_data = {}
    with open(cod_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            q = item.get("instruction", "").strip()
            if q:
                cod_data[q] = item

    # Analysis Metrics
    matched_count = 0
    
    total_len_orig = 0
    total_len_cot = 0
    total_len_cod = 0
    
    total_steps_cot = 0
    total_steps_cod = 0
    
    total_steplen_cot = 0
    total_steplen_cod = 0
    
    correct_answers = 0
    
    logger.info(f"Loaded: Source={len(source_data)}, CoT={len(cot_data)}, CoD={len(cod_data)}")

    for q, source_item in source_data.items():
        if q in cot_data and q in cod_data:
            matched_count += 1
            
            cot_item = cot_data[q]
            cod_item = cod_data[q]
            
            # Answer Verification
            # Extract ground truth
            gt_solution = source_item.get("solution") or source_item.get("output") or source_item.get("answer")
            
            # Extract generated answers (assuming #### format)
            cot_generated_full = cot_item.get("output", "")
            cod_generated_full = cod_item.get("output", "")
            
            # Simple check: does the generated string contain the ground truth answer?
            # Or if dataset is math, extract box/final answer. 
            # For this script we will do a basic "ends with" or "contains" check or extract after ####
            
            def extract_hash_answer(text):
                parts = text.split("####")
                if len(parts) > 1:
                    return parts[-1].strip()
                return ""

            cot_ans = extract_hash_answer(cot_generated_full)
            cod_ans = extract_hash_answer(cod_generated_full)
            
            # Ground truth might not have ####, usually standard dataset like GSM8k/Math has it.
            # If standard dataset solution has ####, we extract it too.
            gt_ans = extract_hash_answer(gt_solution) if "####" in str(gt_solution) else str(gt_solution).strip()

            # Relaxed comparison
            if (cot_ans and gt_ans and cot_ans == gt_ans) or (gt_ans in cot_generated_full):
                 # We count logical correctness if CoT matches.
                 # We assume if CoT is correct, we check CoD
                 if (cod_ans and cod_ans == gt_ans) or (cod_ans == cot_ans):
                     correct_answers += 1
            
            # Lengths
            total_len_orig += get_word_count(str(gt_solution))
            
            # CoT Content (Logic)
            # Remove #### part for logic length
            cot_logic = cot_generated_full.split("####")[0]
            cot_steps = extract_steps(cot_logic)
            total_len_cot += get_word_count(cot_logic)
            total_steps_cot += len(cot_steps)
            if cot_steps:
                 total_steplen_cot += sum(get_word_count(s) for s in cot_steps) / len(cot_steps)

            # CoD Content (Summary)
            cod_logic = cod_generated_full.split("####")[0]
            cod_steps = extract_steps(cod_logic)
            total_len_cod += get_word_count(cod_logic)
            total_steps_cod += len(cod_steps)
            if cod_steps:
                 total_steplen_cod += sum(get_word_count(s) for s in cod_steps) / len(cod_steps)

    if matched_count == 0:
        logger.warning("No matching samples found across all three files.")
        return

    print("\n" + "="*40)
    print("DATA ANALYSIS REPORT")
    print("="*40)
    print(f"Matched Samples: {matched_count}")
    print(f"Correct Answer Consistency (Estimate): {correct_answers}/{matched_count} ({correct_answers/matched_count*100:.1f}%)")
    print("-" * 40)
    print(f"Average Word Count (Reference): {total_len_orig / matched_count:.1f}")
    print(f"Average Word Count (CoT):       {total_len_cot / matched_count:.1f}")
    print(f"Average Word Count (CoD):       {total_len_cod / matched_count:.1f}")
    print("-" * 40)
    print(f"Average Steps (CoT):            {total_steps_cot / matched_count:.1f}")
    print(f"Average Step Length (CoT):      {total_steplen_cot / matched_count:.1f} words")
    print("-" * 40)
    print(f"Average Steps (CoD):            {total_steps_cod / matched_count:.1f}")
    print(f"Average Step Length (CoD):      {total_steplen_cod / matched_count:.1f} words")
    print("="*40)

if __name__ == "__main__":
    main()
