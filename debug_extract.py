import re 

def extract_answer(generation, expected_answer):
    text_to_search = generation
    if "####" in generation:
        text_to_search = generation.split("####")[-1]
    elif "The answer is" in generation:
        text_to_search = generation.split("The answer is")[-1]

    clean_text = text_to_search.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', clean_text)
    
    if numbers:
        pred_str = numbers[-1]
        if pred_str.endswith('.'):
             pred_str = pred_str[:-1]
    else:
        print("No numbers found in generation")
        return 0.0

    expected_str = expected_answer.split("####")[-1].strip().replace(',', '')
    exp_matches = re.findall(r'-?\d+\.?\d*', expected_str)
    if exp_matches:
        expected_val = exp_matches[-1]
        if expected_val.endswith('.'):
            expected_val = expected_val[:-1]
    else:
        print("No numbers found in expected")
        return 0.0
    
    print(f"Gen: '{generation[-50:]}'") # shortened
    print(f"Pred (raw): {pred_str}, Expected (clean): {expected_val}")
    
    try:
        result = 1.0 if float(pred_str) == float(expected_val) else 0.0
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

# Sample Fail Case Theory: "The answer is 1 then 2"
gen_fail = "The answer is the first number is 100, then 200... 540"
gt_fail = "#### 540"
print("\n--- Sample Fail Theory ---")
extract_answer(gen_fail, gt_fail)

