#!/usr/bin/env python3
"""
Check vocabulary sizes of target and draft models for speculative decoding compatibility.
"""

import sys
from transformers import AutoTokenizer, AutoConfig

def check_vocab_size(model_path: str, model_name: str):
    """Check vocabulary size of a model."""
    try:
        print(f"\n{'='*60}")
        print(f"Checking: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")
        
        # Load config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Config vocab_size: {config.vocab_size}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Tokenizer vocab_size: {len(tokenizer)}")
        
        # Check if they match
        if config.vocab_size == len(tokenizer):
            print(f"✅ MATCH - Model and tokenizer aligned")
        else:
            print(f"⚠️  MISMATCH - Config: {config.vocab_size}, Tokenizer: {len(tokenizer)}")
            print(f"   This can happen after fine-tuning with token additions")
        
        return config.vocab_size, len(tokenizer)
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VOCABULARY SIZE CHECKER FOR SPECULATIVE DECODING")
    print("="*60)
    
    # Models to check (update these paths to match your setup)
    models = [
        ("Qwen/Qwen2.5-14B-Instruct", "Base Target (14B)"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "Base Draft (0.5B)"),
        # Add your fine-tuned model paths here:
        # ("models/target_cot_hard", "Fine-tuned Target CoT"),
        # ("models/draft_cod_hard", "Fine-tuned Draft CoD"),
    ]
    
    results = {}
    for model_path, model_name in models:
        config_size, tokenizer_size = check_vocab_size(model_path, model_name)
        if config_size is not None:
            results[model_name] = (config_size, tokenizer_size)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not results:
        print("❌ No models could be loaded")
        sys.exit(1)
    
    vocab_sizes = [size for size, _ in results.values()]
    all_match = len(set(vocab_sizes)) == 1
    
    if all_match:
        print(f"✅ All models have matching vocabulary size: {vocab_sizes[0]}")
        print("   Speculative decoding should work!")
    else:
        print("❌ VOCABULARY SIZE MISMATCH DETECTED!")
        print("\nDetails:")
        for name, (config_size, tokenizer_size) in results.items():
            print(f"  {name}: {config_size}")
        print("\n⚠️  vLLM requires all models to have identical vocabularies")
        print("   for speculative decoding to work.")
        print("\nPossible causes:")
        print("  1. Fine-tuned models added special tokens")
        print("  2. Different base model families (e.g., Llama vs Qwen)")
        print("  3. Tokenizer was modified during training")
        sys.exit(1)
    
    print("\n" + "="*60)