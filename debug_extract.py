import re

def extract_answer(generation, expected_answer):
    """
    Scans the entire generated text for the *last* number.
    """
    gen_text = generation.strip()
    
    if "####" in gen_text:
        pred = gen_text.split("####")[-1].strip()
    elif "The answer is" in gen_text:
        pred = gen_text.split("The answer is")[-1].strip()
    else:
        numbers = re.findall(r'-?\d+\.?\d*', gen_text)
        if numbers:
            pred = numbers[-1]
        else:
            return 0.0 

    pred = re.sub(r'[^\d\.]', '', pred)
    expected = re.sub(r'[^\d\.]', '', expected_answer.split("####")[-1])
    
    print(f"Gen: '{generation}'")
    print(f"Expected Full: '{expected_answer}'")
    print(f"Pred (raw): {pred}, Expected (clean): {expected}")
    
    try:
        result = 1.0 if float(pred) == float(expected) else 0.0
        print(f"Result: {result}")
        return result
    except ValueError:
        print("ValueError during float conversion")
        return 0.0

# Sample 0
gen_0 = "...ily earnings:**     9 eggs ×  $2 per egg = $18  Janet makes **$18** every day at the farmers' market."
gt_0 = "#### 18"
print("--- Sample 0 ---")
extract_answer(gen_0, gt_0)

# Sample 1
gen_1 = "...he breakdown: *   Blue fiber: 2 bolts *   White fiber: 1 bolt (half of 2) *   Total: 2 + 1 = 3 bolts"
gt_1 = "#### 3"
print("\n--- Sample 1 ---")
extract_answer(gen_1, gt_1)
