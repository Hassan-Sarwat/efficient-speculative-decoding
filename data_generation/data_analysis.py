"""
Post-generation dataset validation and analysis.
Run after data generation completes.
"""

import json
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Any # <--- FIXED: Added imports

def validate_structure(output: str) -> Dict[str, bool]:
    """
    Granular check for specific tokens.
    Returns a dict of passing checks.
    """
    return {
        "has_draft_open": "<draft>" in output,
        "has_draft_close": "</draft>" in output,
        "has_separator": "####" in output,
        # Check order: draft tags must appear before the answer separator
        "draft_before_answer": output.find("</draft>") < output.find("####") if "####" in output else False,
        # Check content: Ensure draft is not empty
        "has_content_in_draft": len(re.findall(r'<draft>(.*?)</draft>', output, re.DOTALL)) > 0
    }

def analyze_dataset(filepath: Path):
    """Analyzes generated dataset quality and statistics"""
    
    samples = []
    if not filepath.exists():
        print(f"Error: File not found {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"\n{'='*50}")
    print(f"DATASET ANALYSIS: {filepath.name}")
    print(f"{'='*50}\n")

    # --- 1. Token Verification (New Granular Checks) ---
    print(f"Token Verification Analysis:")
    structural_failures = {
        "missing_draft_tags": 0,
        "missing_separator": 0,
        "draft_after_answer": 0,
        "empty_draft": 0
    }
    
    # We will also track 'total_invalid' for the return value
    invalid_samples = []

    for i, sample in enumerate(samples):
        output = sample.get('output', '')
        checks = validate_structure(output)
        
        is_valid = True

        if not (checks["has_draft_open"] and checks["has_draft_close"]):
            structural_failures["missing_draft_tags"] += 1
            is_valid = False
        elif not checks["has_content_in_draft"]:
            structural_failures["empty_draft"] += 1
            is_valid = False
        
        if not checks["has_separator"]:
            structural_failures["missing_separator"] += 1
            is_valid = False
        elif not checks["draft_before_answer"] and checks["has_draft_close"]:
            structural_failures["draft_after_answer"] += 1
            is_valid = False
            
        if not is_valid:
            invalid_samples.append(i)

    # Report Granular Stats
    print(f"  Missing Draft Tags: {structural_failures['missing_draft_tags']}")
    print(f"  Missing '####':     {structural_failures['missing_separator']}")
    print(f"  Order Invalid:      {structural_failures['draft_after_answer']}")
    print(f"  Empty Drafts:       {structural_failures['empty_draft']}")
    print(f"  ------------------")
    print(f"  Total Valid: {len(samples) - len(invalid_samples)}/{len(samples)}")
    
    # --- 2. Length Statistics ---
    def extract_draft_length(output):
        match = re.search(r'<draft>(.*?)</draft>', output, re.DOTALL)
        return len(match.group(1).split()) if match else 0
    
    lengths = [extract_draft_length(s['output']) for s in samples]
    
    avg_len = sum(lengths)/len(lengths) if lengths else 0
    
    print(f"\nReasoning Length Statistics:")
    print(f"  Mean: {avg_len:.1f} words")
    if lengths:
        print(f"  Median: {sorted(lengths)[len(lengths)//2]} words")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    
    # --- 3. Plot Distribution ---
    # Create analysis directory if it doesn't exist
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(100, color='red', linestyle='--', label='100-word threshold')
    plt.xlabel('Chain Length (words)')
    plt.ylabel('Frequency')
    plt.title(f'Reasoning Length Distribution - {filepath.stem}')
    plt.legend()
    
    plot_path = output_dir / f'{filepath.stem}_distribution.png'
    plt.savefig(plot_path)
    print(f"\n✓ Plot saved: {plot_path}")
    
    return {
        'total': len(samples),
        'format_errors': len(invalid_samples), # <--- FIXED: Mapped to new logic
        'avg_length': avg_len,
        'length_distribution': dict(Counter([l//10*10 for l in lengths]))
    }

if __name__ == "__main__":
    # Ensure analysis directory exists
    Path('analysis').mkdir(exist_ok=True)
    
    # Example usage - check if files exist before running
    datasets = ['gsm8k', 'MATH_L1-2_Algebra', 'MATH_L3-4_IntermediateAlgebra']
    for ds in datasets:
        filepath = Path(f'data/cob_data_{ds}.jsonl')
        if filepath.exists():
            analyze_dataset(filepath)