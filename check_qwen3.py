#!/usr/bin/env python3
"""
Check Qwen3 model family vocabulary compatibility for speculative decoding.
Compare with Qwen2.5 to see if it's a better choice.

This version fetches config.json directly from HuggingFace Hub without downloading models.
"""

import requests
import json

def check_model_vocab(model_path: str):
    """Get vocab size for a model from HuggingFace Hub."""
    try:
        # Fetch config.json from HuggingFace Hub
        url = f"https://huggingface.co/{model_path}/raw/main/config.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            return "NOT FOUND"
        
        response.raise_for_status()
        config = response.json()
        
        # Get vocab_size from config
        vocab_size = config.get('vocab_size')
        if vocab_size is None:
            return "NO VOCAB_SIZE IN CONFIG"
        
        return vocab_size
        
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except requests.exceptions.RequestException as e:
        return f"ERROR: {str(e)}"
    except json.JSONDecodeError:
        return "INVALID JSON"
    except Exception as e:
        return f"ERROR: {str(e)}"

print("\n" + "="*80)
print("QWEN MODEL FAMILY VOCABULARY COMPARISON")
print("Fetching config.json directly from HuggingFace Hub (no downloads)")
print("="*80)

# Qwen2.5 Family (your current choice)
print("\n📦 QWEN 2.5 FAMILY:")
print("-" * 80)
qwen25_models = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct", "3B"),
    ("Qwen/Qwen2.5-7B-Instruct", "7B"),
    ("Qwen/Qwen2.5-14B-Instruct", "14B"),
    ("Qwen/Qwen2.5-32B-Instruct", "32B"),
]

qwen25_vocabs = {}
for model_path, size in qwen25_models:
    vocab_size = check_model_vocab(model_path)
    qwen25_vocabs[size] = vocab_size
    status = "✓" if isinstance(vocab_size, int) else "✗"
    print(f"  {status} {size:6s}: {vocab_size:>10,}" if isinstance(vocab_size, int) else f"  {status} {size:6s}: {vocab_size}")

# Qwen3 Family (potential alternative)
print("\n📦 QWEN 3 FAMILY:")
print("-" * 80)
qwen3_models = [
    ("Qwen/Qwen3-0.6B", "0.6B"),
    ("Qwen/Qwen3-1.7B", "1.7B"),
    ("Qwen/Qwen3-4B", "4B"),
    ("Qwen/Qwen3-8B", "8B"),
    ("Qwen/Qwen3-14B", "14B"),
    ("Qwen/Qwen3-32B", "32B"),
]

qwen3_vocabs = {}
for model_path, size in qwen3_models:
    vocab_size = check_model_vocab(model_path)
    qwen3_vocabs[size] = vocab_size
    status = "✓" if isinstance(vocab_size, int) else "✗"
    print(f"  {status} {size:6s}: {vocab_size:>10,}" if isinstance(vocab_size, int) else f"  {status} {size:6s}: {vocab_size}")

# Analysis
print("\n" + "="*80)
print("COMPATIBILITY ANALYSIS")
print("="*80)

def analyze_family(vocabs, family_name):
    """Analyze vocabulary consistency in a model family."""
    print(f"\n{family_name}:")
    
    valid_vocabs = {k: v for k, v in vocabs.items() if isinstance(v, int)}
    
    if not valid_vocabs:
        print("  ❌ No models available")
        return None
    
    unique_vocabs = set(valid_vocabs.values())
    
    if len(unique_vocabs) == 1:
        vocab_size = list(unique_vocabs)[0]
        print(f"  ✅ ALL models share same vocabulary: {vocab_size:,} tokens")
        print(f"  ✅ Perfect for speculative decoding!")
        return vocab_size
    else:
        print(f"  ⚠️  Multiple vocabulary sizes detected: {sorted(unique_vocabs)}")
        print(f"  ❌ Requires careful model pair selection")
        
        # Group by vocab size
        groups = {}
        for size, vocab in valid_vocabs.items():
            if vocab not in groups:
                groups[vocab] = []
            groups[vocab].append(size)
        
        print(f"\n  Compatible groups:")
        for vocab, sizes in sorted(groups.items()):
            print(f"    • Vocab {vocab:,}: {', '.join(sizes)}")
        
        return None

qwen25_unified = analyze_family(qwen25_vocabs, "Qwen 2.5")
qwen3_unified = analyze_family(qwen3_vocabs, "Qwen 3")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR SPECULATIVE DECODING")
print("="*80)

if qwen3_unified:
    print(f"\n✅ RECOMMENDED: Use Qwen3 Family")
    print(f"   All models share {qwen3_unified:,} token vocabulary")
    print(f"\n   Suggested pairs for your use case:")
    print(f"   • Target: Qwen/Qwen3-14B-Instruct (14B params)")
    print(f"   • Draft:  Qwen/Qwen3-0.6B-Instruct (600M params) - Best speedup")
    print(f"   • Draft:  Qwen/Qwen3-1.7B-Instruct (1.7B params) - Better quality")
    print(f"   • Draft:  Qwen/Qwen3-4B-Instruct (4B params) - Best balance")
    
elif qwen25_unified:
    print(f"\n✅ Use Qwen 2.5 Family")
    print(f"   All models share {qwen25_unified:,} token vocabulary")
    
else:
    print(f"\n⚠️  For Qwen 2.5: Use these compatible pairs:")
    if qwen25_vocabs.get("14B") == qwen25_vocabs.get("3B"):
        print(f"   • Target: Qwen2.5-14B + Draft: Qwen2.5-3B")
    if qwen25_vocabs.get("14B") == qwen25_vocabs.get("7B"):
        print(f"   • Target: Qwen2.5-14B + Draft: Qwen2.5-7B")

print("\n" + "="*80 + "\n")