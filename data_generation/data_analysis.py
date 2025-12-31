"""
Post-generation dataset validation and analysis.
Run after data generation completes.
"""

import json
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(filepath: Path):
    """Analyzes generated dataset quality and statistics"""
    
    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"\n{'='*50}")
    print(f"DATASET ANALYSIS: {filepath.name}")
    print(f"{'='*50}\n")
    
    # 1. Format Validation
    format_errors = []
    for i, sample in enumerate(samples):
        output = sample.get('output', '')
        if not re.match(r'<draft>.*?</draft>\s*####\s*.+', output, re.DOTALL):
            format_errors.append((i, output[:100]))
    
    print(f"Format Validation:")
    print(f"  Valid: {len(samples) - len(format_errors)}/{len(samples)}")
    print(f"  Invalid: {len(format_errors)}")
    
    # 2. Length Statistics
    def extract_draft_length(output):
        match = re.search(r'<draft>(.*?)</draft>', output, re.DOTALL)
        return len(match.group(1).split()) if match else 0
    
    lengths = [extract_draft_length(s['output']) for s in samples]
    
    print(f"\nReasoning Length Statistics:")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f} words")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]} words")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    
    # 3. Plot Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(100, color='red', linestyle='--', label='100-word threshold')
    plt.xlabel('Chain Length (words)')
    plt.ylabel('Frequency')
    plt.title(f'Reasoning Length Distribution - {filepath.stem}')
    plt.legend()
    plt.savefig(f'analysis/{filepath.stem}_distribution.png')
    print(f"\n✓ Plot saved: analysis/{filepath.stem}_distribution.png")
    
    return {
        'total': len(samples),
        'format_errors': len(format_errors),
        'avg_length': sum(lengths)/len(lengths),
        'length_distribution': Counter([l//10*10 for l in lengths])  # Bin by 10s
    }

if __name__ == "__main__":
    Path('analysis').mkdir(exist_ok=True)
    
    datasets = ['gsm8k', 'MATH_L1-2_Algebra', 'MATH_L3-4_IntermediateAlgebra']
    for ds in datasets:
        filepath = Path(f'data/cob_data_{ds}.jsonl')
        if filepath.exists():
            analyze_dataset(filepath)