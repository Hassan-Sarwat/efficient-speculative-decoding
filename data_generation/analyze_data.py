import argparse
import logging
import json
import re
import sys
import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

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

def count_steps(text: str) -> int:
    """
    Counts steps in the reasoning chain.
    """
    if not text:
        return 0
        
    # Pattern 0: Arrow separators
    arrow_count = text.count("->")
    if arrow_count > 0:
        return arrow_count + 1

    # Pattern 1: Step \d+:
    step_matches = re.findall(r'Step \d+:', text)
    if step_matches:
        return len(step_matches)
    
    # Fallback: Paragraphs
    parts = [s for s in text.split('\n\n') if s.strip()]
    return len(parts)

def get_char_count(text: str) -> int:
    """Returns the character count of the text."""
    return len(text)

def extract_boxed_content(text: str) -> str:
    """Extracts content from the last \\boxed{...} in the text."""
    if not text: return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
        
    start_idx = idx + 7
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

def clean_competition_math_answer(text: str) -> str:
    """
    Cleans competition math answers. 
    Strips simple latex markers but preserves content like fractions if possible.
    For rigorous numeric checking, the new check_equality function handles numbers well.
    """
    if not text: return ""
    text = text.replace("$", "")
    text = text.replace(",", "").strip()
    return text

def extract_answer(text: str, dataset_name: str = "", is_ground_truth: bool = False) -> str:
    """Extracts answer substring."""
    if not text:
        return ""

    if not is_ground_truth:
        parts = text.split("####")
        if len(parts) > 1:
            return parts[-1].strip()
        return ""

    if "gsm8k" in dataset_name.lower():
        if "####" in text:
            return text.split("####")[-1].strip()
        return text.strip()

    if "competition_math" in dataset_name.lower() or "math" in dataset_name.lower():
        boxed = extract_boxed_content(text)
        if boxed:
            return clean_competition_math_answer(boxed)
        if "####" in text:
            return text.split("####")[-1].strip()
        return text.strip()

    if "####" in text:
        return text.split("####")[-1].strip()
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed
    return text.strip()

def normalize_string(text: str) -> str:
    """Basic string normalization."""
    if not text:
        return ""
    text = str(text).strip()
    text = text.replace(",", "")
    if text.endswith("."):
        text = text[:-1]
    return text

def parse_number(text: str) -> Tuple[Optional[float], bool]:
    """
    Extracts the first valid number from text.
    Returns (value, is_percentage_bool).
    Handles: "16", "16.5", "75%", "$990.00"
    """
    # Remove commas
    clean_text = text.replace(",", "")
    
    # Regex for a number (int or float) possibly followed by %
    # Matches: -123, 123.45, .45, 123%
    pattern = r'(-?\d+\.?\d*|-?\.\d+)(%)?'
    match = re.search(pattern, clean_text)
    
    if match:
        num_str = match.group(1)
        has_percent = bool(match.group(2))
        try:
            return float(num_str), has_percent
        except ValueError:
            return None, False
    return None, False

def check_equality(ans1: str, ans2: str) -> bool:
    """
    Checks if ans1 and ans2 are equivalent.
    1. Exact string match (normalized)
    2. Numeric match (with unit removal, float tolerance, % handling)
    """
    # 1. String Match
    s1 = normalize_string(ans1)
    s2 = normalize_string(ans2)
    if s1 == s2:
        return True
        
    # 2. Numeric Match
    v1, p1 = parse_number(ans1)
    v2, p2 = parse_number(ans2)
    
    if v1 is None or v2 is None:
        return False
        
    # Helper to compare floats with tolerance
    def is_close(a, b):
        return abs(a - b) < 1e-6
        
    # Case A: Both are simple numbers or both are percents -> direct compare
    # "75" vs "75.00" -> True
    # "75%" vs "75.00%" -> True
    if p1 == p2:
        return is_close(v1, v2)
    
    # Case B: One is percent, one is not
    # User rule: "75%" (75) matches "75" (75) OR "0.75" (0.75)
    
    # If p1 is percent (e.g. 75), p2 is not (e.g. 0.75 or 75)
    if p1 and not p2:
        # Check against raw value (75 == 75)
        if is_close(v1, v2): return True
        # Check against decimal value (75/100 == 0.75)
        if is_close(v1/100.0, v2): return True
        
    if p2 and not p1:
        if is_close(v2, v1): return True
        if is_close(v2/100.0, v1): return True
        
    return False

# --- Main Analysis Logic ---

def main():
    parser = argparse.ArgumentParser(description="Analyze generated datasets")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="Dataset name")
    parser.add_argument("--suffix", type=str, help="File suffix used during generation")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where data is stored")
    args = parser.parse_args()

    safe_name = args.dataset.replace("/", "_")
    output_dir = Path(args.output_dir)

    # 1. Determine Filenames
    if args.suffix:
        dataset_filename = f"{safe_name}_{args.suffix}.jsonl"
        cot_filename = f"cot_{args.suffix}.jsonl"
        cod_filename = f"cod_{args.suffix}.jsonl"
        report_filename = f"{safe_name}_{args.suffix}_report.txt"
        incorrect_csv_filename = f"{safe_name}_{args.suffix}_incorrect_samples.csv"
    else:
        dataset_filename = f"{safe_name}.jsonl"
        cot_filename = f"cot_{safe_name}.jsonl"
        cod_filename = f"cod_{safe_name}.jsonl"
        report_filename = f"{safe_name}_report.txt"
        incorrect_csv_filename = f"{safe_name}_incorrect_samples.csv"
    
    dataset_path = output_dir / "raw" / dataset_filename
    cot_path = output_dir / "training" / cot_filename
    cod_path = output_dir / "training" / cod_filename
    
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / report_filename
    incorrect_csv_path = reports_dir / incorrect_csv_filename
    
    # Check existence
    missing = []
    if not dataset_path.exists(): missing.append(f"Dataset ({dataset_path})")
    if not cot_path.exists(): missing.append(f"CoT ({cot_path})")
    if not cod_path.exists(): missing.append(f"CoD ({cod_path})")
    
    if missing:
        logger.error(f"Missing files: {', '.join(missing)}")
        sys.exit(1)

    logger.info("All files present. Starting analysis...")

    # Load Data Helper
    def load_dataset_map(filepath: Path, is_generated: bool = False) -> Dict[str, Any]:
        data_map = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q_text = None
                    if is_generated:
                        q_text = item.get("instruction", "").strip()
                    else:
                        q_text = item.get("question") or item.get("problem") or item.get("instruction")
                        if q_text: q_text = q_text.strip()

                    item_id = str(item.get("id") or item.get("problem_id") or item.get("question_id") or "")
                    
                    if item_id:
                        data_map[f"ID:{item_id}"] = item
                    if q_text:
                        data_map[f"TXT:{q_text}"] = item
                        
                except json.JSONDecodeError:
                    continue
        return data_map

    source_map = load_dataset_map(dataset_path, is_generated=False)
    cot_map = load_dataset_map(cot_path, is_generated=True)
    cod_map = load_dataset_map(cod_path, is_generated=True)

    matched_count = 0
    correct_cot = 0
    correct_cod = 0
    
    steps_cot_list = []
    chars_cot_list = []
    steps_cod_list = []
    chars_cod_list = []
    
    incorrect_samples = []

    visited_ids = set()
    source_items = [v for k, v in source_map.items() if k.startswith("ID:")]
    source_items.extend([v for k, v in source_map.items() if k.startswith("TXT:")])

    for source_item in source_items:
        src_q = source_item.get("question") or source_item.get("problem") or source_item.get("instruction")
        src_id = str(source_item.get("id") or source_item.get("problem_id") or source_item.get("question_id") or "")
        
        unique_key = src_id if src_id else str(hash(src_q))
        if unique_key in visited_ids:
            continue
        visited_ids.add(unique_key)

        cot_item = None
        if src_id and f"ID:{src_id}" in cot_map: cot_item = cot_map[f"ID:{src_id}"]
        elif src_q and f"TXT:{src_q.strip()}" in cot_map: cot_item = cot_map[f"TXT:{src_q.strip()}"]
            
        cod_item = None
        if src_id and f"ID:{src_id}" in cod_map: cod_item = cod_map[f"ID:{src_id}"]
        elif src_q and f"TXT:{src_q.strip()}" in cod_map: cod_item = cod_map[f"TXT:{src_q.strip()}"]

        if cot_item and cod_item:
            matched_count += 1
            
            # --- Answer Verification ---
            gt_solution = source_item.get("solution") or source_item.get("output") or source_item.get("answer") or ""
            cot_generated_full = cot_item.get("output", "")
            cod_generated_full = cod_item.get("output", "")
            
            # Extract raw answers first
            gt_ans_raw = extract_answer(str(gt_solution), args.dataset, is_ground_truth=True)
            cot_ans_raw = extract_answer(cot_generated_full, args.dataset, is_ground_truth=False)
            cod_ans_raw = extract_answer(cod_generated_full, args.dataset, is_ground_truth=False)
            
            # Check correctness using robust logic
            cot_is_correct = False
            if gt_ans_raw and cot_ans_raw and check_equality(cot_ans_raw, gt_ans_raw):
                correct_cot += 1
                cot_is_correct = True
            
            cod_is_correct = False
            if gt_ans_raw and cod_ans_raw and check_equality(cod_ans_raw, gt_ans_raw):
                correct_cod += 1
                cod_is_correct = True
                
            # Log incorrect samples
            if not gt_ans_raw or (not cot_is_correct or not cod_is_correct):
                incorrect_samples.append({
                    "index": src_id if src_id else "N/A",
                    "full answer": gt_solution,
                    "extracted answer": gt_ans_raw,
                    "cot_answer": cot_ans_raw,
                    "cod_answer": cod_ans_raw
                })
            
            # --- Metrics ---
            cot_logic = cot_generated_full.split("####")[0] if "####" in cot_generated_full else cot_generated_full
            cod_logic = cod_generated_full.split("####")[0] if "####" in cod_generated_full else cod_generated_full
            
            steps_cot_list.append(count_steps(cot_logic))
            steps_cod_list.append(count_steps(cod_logic))
            
            chars_cot_list.append(get_char_count(cot_logic))
            chars_cod_list.append(get_char_count(cod_logic))

    if matched_count == 0:
        logger.warning("No matching samples found.")
        return

    # --- Statistics Calculation ---
    avg_steps_cot = sum(steps_cot_list) / matched_count
    avg_steps_cod = sum(steps_cod_list) / matched_count
    
    avg_chars_cot = sum(chars_cot_list) / matched_count
    avg_chars_cod = sum(chars_cod_list) / matched_count
    
    min_steps_cot = min(steps_cot_list) if steps_cot_list else 0
    max_steps_cot = max(steps_cot_list) if steps_cot_list else 0
    min_steps_cod = min(steps_cod_list) if steps_cod_list else 0
    max_steps_cod = max(steps_cod_list) if steps_cod_list else 0
    
    min_chars_cot = min(chars_cot_list) if chars_cot_list else 0
    max_chars_cot = max(chars_cot_list) if chars_cot_list else 0
    min_chars_cod = min(chars_cod_list) if chars_cod_list else 0
    max_chars_cod = max(chars_cod_list) if chars_cod_list else 0
    
    imp_steps_pct = ((avg_steps_cot - avg_steps_cod) / avg_steps_cot * 100) if avg_steps_cot > 0 else 0
    imp_chars_pct = ((avg_chars_cot - avg_chars_cod) / avg_chars_cot * 100) if avg_chars_cot > 0 else 0

    # --- Generate Report ---
    report_lines = []
    report_lines.append(f"DATA ANALYSIS REPORT: {safe_name}")
    report_lines.append("="*50)
    report_lines.append(f"Files Analyzed:")
    report_lines.append(f"  Dataset: {dataset_filename}")
    report_lines.append(f"  CoT:     {cot_filename}")
    report_lines.append(f"  CoD:     {cod_filename}")
    report_lines.append("-" * 50)
    report_lines.append(f"Matched Samples: {matched_count}")
    report_lines.append(f"CoT Accuracy:    {correct_cot}/{matched_count} ({correct_cot/matched_count*100:.2f}%)")
    report_lines.append(f"CoD Accuracy:    {correct_cod}/{matched_count} ({correct_cod/matched_count*100:.2f}%)")
    report_lines.append("-" * 50)
    report_lines.append("METRICS (Average / Min / Max)")
    report_lines.append(f"CoT Steps:       {avg_steps_cot:.2f} / {min_steps_cot} / {max_steps_cot}")
    report_lines.append(f"CoD Steps:       {avg_steps_cod:.2f} / {min_steps_cod} / {max_steps_cod}")
    report_lines.append(f"CoT Characters:  {avg_chars_cot:.2f} / {min_chars_cot} / {max_chars_cot}")
    report_lines.append(f"CoD Characters:  {avg_chars_cod:.2f} / {min_chars_cod} / {max_chars_cod}")
    report_lines.append("-" * 50)
    report_lines.append("IMPROVEMENT (CoD vs CoT)")
    report_lines.append(f"Step Reduction:      {avg_steps_cot - avg_steps_cod:.2f} steps ({imp_steps_pct:.2f}%)")
    report_lines.append(f"Character Reduction: {avg_chars_cot - avg_chars_cod:.2f} chars ({imp_chars_pct:.2f}%)")
    report_lines.append("="*50)
    
    report_content = "\n".join(report_lines)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    if incorrect_samples:
        fieldnames = ["index", "full answer", "extracted answer", "cot_answer", "cod_answer"]
        with open(incorrect_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(incorrect_samples)
        logger.info(f"Incorrect samples saved to {incorrect_csv_path}")
        report_lines.append(f"Incorrect Samples saved to: {incorrect_csv_filename}")
        print(report_content + f"\nIncorrect Samples saved to: {incorrect_csv_filename}")
    else:
        print(report_content)
        
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
