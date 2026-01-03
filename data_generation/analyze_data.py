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

# --- Helper Functions ---

def extract_steps(text: str) -> List[str]:
    """Extracts reasoning steps from text."""
    # Pattern 0: Arrow separators (typical for Chain of Draft)
    if "->" in text:
        return [s.strip() for s in text.split("->") if s.strip()]

    # Pattern 1: Step 1:, Step 2: etc.
    steps = re.split(r'Step \d+:', text)
    if len(steps) > 1:
        return [s.strip() for s in steps if s.strip()]
    
    # Pattern 2: List items (1. 2. 3.) could be added here if needed
    
    # Fallback: Split by double newlines
    return [s.strip() for s in text.split('\n\n') if s.strip()]

def get_word_count(text: str) -> int:
    return len(text.split())

def extract_boxed_content(text: str) -> str:
    """Extracts content from the last \\boxed{...} in the text."""
    if not text: return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
        
    # Start scanning after \boxed{
    start_idx = idx + 7  # len("\\boxed{")
    balance = 1
    
    for i in range(start_idx, len(text)):
        char = text[i]
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            if balance == 0:
                return text[start_idx:i]
    return None

def extract_answer(text: str, is_ground_truth: bool = False) -> str:
    """
    Extracts the answer from text.
    
    Args:
        text: The text to extract from
        is_ground_truth: If True, looks for both #### and \boxed formats.
                       If False (generated), only looks for ####.
    """
    if not text:
        return ""

    # 1. Generated Text: Strict #### format
    if not is_ground_truth:
        parts = text.split("####")
        if len(parts) > 1:
            return parts[-1].strip()
        return ""

    # 2. Ground Truth: Flexible format
    # Priority 1: #### (GSM8K style)
    if "####" in text:
        return text.split("####")[-1].strip()
        
    # Priority 2: \boxed{...} (MATH style)
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed.strip()
        
    # Priority 3: Raw text fallback
    return text.strip()

def normalize_answer(text: str) -> str:
    """Normalizes answer for comparison (strip spaces, etc)."""
    if not text:
        return ""
    # Simple normalization
    text = text.strip()
    text = text.replace(",", "")
    return text

# --- Main Analysis Logic ---

def main():
    parser = argparse.ArgumentParser(description="Analyze generated datasets")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="Dataset name")
    parser.add_argument("--suffix", type=str, help="File suffix used during generation")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where data is stored")
    args = parser.parse_args()

    safe_name = args.dataset.replace("/", "_")
    output_dir = Path(args.output_dir)

    # 1. Check Files
    if args.suffix:
        dataset_filename = f"{safe_name}_{args.suffix}.jsonl"
        cot_filename = f"cot_{args.suffix}.jsonl"
        cod_filename = f"cob_{args.suffix}.jsonl"
    else:
        dataset_filename = f"{safe_name}.jsonl"
        cot_filename = f"cot_{safe_name}.jsonl"
        cod_filename = f"cob_{safe_name}.jsonl"
    
    dataset_path = output_dir / dataset_filename
    cot_path = output_dir / cot_filename
    cod_path = output_dir / cod_filename
    
    missing = []
    if not dataset_path.exists(): missing.append(f"Dataset ({dataset_path})")
    if not cot_path.exists(): missing.append(f"CoT ({cot_path})")
    if not cod_path.exists(): missing.append(f"CoD ({cod_path})")
    
    if missing:
        logger.error(f"Missing files: {', '.join(missing)}")
        sys.exit(1)

    logger.info("All files present. Starting analysis...")

    # Load Data
    # We will store data in dual-key dictionaries: id -> item AND question_text -> item
    # This allows O(1) lookup by either ID or Text.
    
    def load_dataset_map(filepath: Path, is_generated: bool = False) -> Dict[str, Any]:
        data_map = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # 1. Extract Question Text (Cleaned)
                    q_text = None
                    if is_generated:
                        q_text = item.get("instruction", "").strip()
                    else:
                        q_text = item.get("question") or item.get("problem") or item.get("instruction")
                        if q_text: q_text = q_text.strip()

                    # 2. Extract ID
                    # generated files might have "id" field now
                    # source files might have "id", "problem_id", etc.
                    item_id = str(item.get("id") or item.get("problem_id") or item.get("question_id") or "")
                    
                    # Store by ID if valid
                    if item_id:
                        data_map[f"ID:{item_id}"] = item
                    
                    # Store by Text if valid
                    if q_text:
                        data_map[f"TXT:{q_text}"] = item
                        
                except json.JSONDecodeError:
                    continue
        return data_map

    source_map = load_dataset_map(dataset_path, is_generated=False)
    cot_map = load_dataset_map(cot_path, is_generated=True)
    cod_map = load_dataset_map(cod_path, is_generated=True)

    # Metrics
    matched_count = 0
    
    total_len_orig = 0
    total_len_cot = 0
    total_len_cod = 0
    
    total_steps_cot = 0
    total_steps_cod = 0
    
    total_steplen_cot = 0
    total_steplen_cod = 0
    
    correct_cot = 0
    correct_cod = 0
    
    # We iterate over source map to find matches in CoT/CoD
    # To avoid duplicates (since we have both ID and TXT keys), we track visited items
    visited_ids = set()
    
    # First pass: Iterate by ID keys in source
    source_items = [v for k, v in source_map.items() if k.startswith("ID:")]
    # Second pass: Iterate by TXT keys in source (only if not visited)
    source_items.extend([v for k, v in source_map.items() if k.startswith("TXT:")])

    logger.info(f"Loaded maps. Starting matching...")

    for source_item in source_items:
        # Determine unique key for this item to prevent double counting
        # (Use ID if available, else Hash of question)
        src_q = source_item.get("question") or source_item.get("problem") or source_item.get("instruction")
        src_id = str(source_item.get("id") or source_item.get("problem_id") or source_item.get("question_id") or "")
        
        unique_key = src_id if src_id else str(hash(src_q))
        
        if unique_key in visited_ids:
            continue
        visited_ids.add(unique_key)

        # Try to find in CoT
        cot_item = None
        if src_id and f"ID:{src_id}" in cot_map:
            cot_item = cot_map[f"ID:{src_id}"]
        elif src_q and f"TXT:{src_q.strip()}" in cot_map:
            cot_item = cot_map[f"TXT:{src_q.strip()}"]
            
        # Try to find in CoD
        cod_item = None
        if src_id and f"ID:{src_id}" in cod_map:
            cod_item = cod_map[f"ID:{src_id}"]
        elif src_q and f"TXT:{src_q.strip()}" in cod_map:
            cod_item = cod_map[f"TXT:{src_q.strip()}"]

        # Only proceed if we have BOTH matched
        if cot_item and cod_item:
            matched_count += 1
            
            # Answer Verification
            gt_solution = source_item.get("solution") or source_item.get("output") or source_item.get("answer") or ""
            cot_generated_full = cot_item.get("output", "")
            cod_generated_full = cod_item.get("output", "")
            
            gt_ans = normalize_answer(extract_answer(str(gt_solution), is_ground_truth=True))
            cot_ans = normalize_answer(extract_answer(cot_generated_full, is_ground_truth=False))
            cod_ans = normalize_answer(extract_answer(cod_generated_full, is_ground_truth=False))
            
            if gt_ans and cot_ans and cot_ans == gt_ans:
                correct_cot += 1
                
            if gt_ans and cod_ans and cod_ans == gt_ans:
                correct_cod += 1
            
            # Text Stats - Original Length
            total_len_orig += get_word_count(str(gt_solution))
            
            # CoT Logic Metrics
            cot_logic = cot_generated_full.split("####")[0]
            cot_steps = extract_steps(cot_logic)
            total_len_cot += get_word_count(cot_logic)
            total_steps_cot += len(cot_steps)
            if cot_steps:
                 total_steplen_cot += sum(get_word_count(s) for s in cot_steps) / len(cot_steps)

            # CoD Logic Metrics
            cod_logic = cod_generated_full.split("####")[0]
            cod_steps = extract_steps(cod_logic)
            total_len_cod += get_word_count(cod_logic)
            total_steps_cod += len(cod_steps)
            if cod_steps:
                 total_steplen_cod += sum(get_word_count(s) for s in cod_steps) / len(cod_steps)

    if matched_count == 0:
        logger.warning("No matching samples found across all three files.")
        return

    # Reporting
    print("\n" + "="*40)
    print("DATA ANALYSIS REPORT")
    print("="*40)
    print(f"Matched Samples: {matched_count}")
    print("-" * 40)
    print(f"CoT Accuracy: {correct_cot}/{matched_count} ({correct_cot/matched_count*100:.1f}%)")
    print(f"CoD Accuracy: {correct_cod}/{matched_count} ({correct_cod/matched_count*100:.1f}%)")
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
