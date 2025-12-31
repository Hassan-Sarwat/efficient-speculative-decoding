
from datasets import load_dataset

def inspect():
    try:
        ds = load_dataset("qwedsacf/competition_math", split="train", streaming=True)
        print("Successfully loaded dataset streaming.")
        
        types = set()
        
        count = 0
        for sample in ds:
            if count > 2000: break # Check more samples to find rarer types
            t = str(sample.get("type", "N/A"))
            types.add(t)
            count += 1
            
        print(f"Types found: {sorted(list(types))}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
