import re
from typing import Optional

def resolve_fractions(text: str) -> str:
    """
    Resolves both LaTeX fractions (\\frac{1}{2}) AND plain text fractions (1/2) to decimals.
    
    Examples: 
      "\\frac{1}{2}" -> "0.5"
      "1/2"          -> "0.5"
      "1,000/4"      -> "250.0"
      "3\\frac{1}{2}" -> "3.5"
    
    Args:
        text: Input text containing fractions
    
    Returns:
        Text with fractions converted to decimals
    """
    if not text: 
        return text
    text = str(text)
    
    def calculate_div(n_str, d_str):
        try:
            n = float(n_str.replace(',', '').strip())
            d = float(d_str.replace(',', '').strip())
            if d == 0: 
                return None
            return str(n / d)
        except ValueError:
            return None
    
    # Pass 1: LaTeX Fractions (\frac{a}{b} or \dfrac{a}{b})
    def repl_latex(m):
        val = calculate_div(m.group(2), m.group(3))
        return val if val is not None else m.group(0)
    
    text = re.sub(r'\\(d?)frac\{([^{}]+)\}\{([^{}]+)\}', repl_latex, text)
    
    # Pass 2: Plain Text Fractions (a/b)
    def repl_plain(m):
        val = calculate_div(m.group(1), m.group(2))
        return val if val is not None else m.group(0)
    
    plain_pattern = r'(-?\d+(?:,\d+)*)\s*/\s*(-?\d+(?:,\d+)*)'
    text = re.sub(plain_pattern, repl_plain, text)
    
    return text


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from LaTeX \\boxed{...} with balanced brace matching.
    
    Examples:
      "Therefore \\boxed{42}" -> "42"
      "Answer: \\boxed{\\frac{1}{2}}" -> "\\frac{1}{2}"
      "\\boxed{(1,2)}" -> "(1,2)"
    
    Args:
        text: Input text containing \\boxed{} expression
    
    Returns:
        Content inside \\boxed{}, or None if not found
    """
    if not text: 
        return None
    
    idx = text.rfind("\\boxed{")
    if idx == -1: 
        return None
    
    start_idx = idx + 7  # Length of "\\boxed{"
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
    Remove LaTeX formatting from MATH dataset answers.
    
    Examples:
      "$42$" -> "42"
      "1,000" -> "1000"
      "$\\frac{1}{2}$" -> "\\frac{1}{2}"
    
    Args:
        text: LaTeX-formatted answer
    
    Returns:
        Cleaned answer string
    """
    if not text: 
        return ""
    text = text.replace("$", "")
    text = text.replace(",", "").strip()
    return text


def extract_answer(text: str, scenario: str = "easy") -> str:
    """
    Extract final answer from model output based on dataset format.
    
    Scenarios:
      - "easy": GSM8K format (prioritizes ####)
      - "medium"/"hard": MATH format (prioritizes \\boxed{})
    
    Examples:
      GSM8K: "Let me solve: 5+3=8\\n#### 8" -> "8"
      MATH: "Therefore \\boxed{\\frac{1}{2}}" -> "0.5" (after fraction resolution)
      Fallback: "The answer is 42" -> "42"
    
    Args:
        text: Model output text
        scenario: Dataset difficulty level
    
    Returns:
        Extracted answer string (empty if extraction fails)
    """
    if not text: 
        return ""
    
    text = str(text).strip()
    is_math = scenario in ['medium', 'hard']
    is_gsm8k = scenario == 'easy'
    
    if is_math:
        # Priority 1: LaTeX \boxed{} format
        boxed = extract_boxed_content(text)
        if boxed:
            cleaned = clean_competition_math_answer(boxed)
            # Resolve fractions in the extracted answer
            return resolve_fractions(cleaned)
        
        # Priority 2: GSM8K-style #### separator (cross-contamination)
        if "####" in text:
            parts = [p.strip() for p in text.split("####") if p.strip()]
            if len(parts) >= 2:
                answer = parts[-1]
                if answer:
                    return resolve_fractions(answer)
        
        # Priority 3: Last number fallback
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    if is_gsm8k:
        # Priority 1: #### separator (required format)
        if "####" in text:
            parts = [p.strip() for p in text.split("####") if p.strip()]
            if len(parts) >= 2:
                answer = parts[-1]
                if answer and re.search(r'\d', answer):
                    return resolve_fractions(answer)
        
        # Priority 2: Last number fallback
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    if "####" in text:
        parts = [p.strip() for p in text.split("####") if p.strip()]
        if len(parts) >= 2:
            return resolve_fractions(parts[-1])
    
    boxed = extract_boxed_content(text)
    if boxed:
        cleaned = clean_competition_math_answer(boxed)
        return resolve_fractions(cleaned)
    
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return ""


def normalize_string(text: str) -> str:
    """
    Normalize text for comparison.
    
    Examples:
      "1,000" -> "1000"
      "42." -> "42"
      "  $50  " -> "50"
    
    Args:
        text: Input text
    
    Returns:
        Normalized string
    """
    if not text: 
        return ""
    text = str(text).strip()
    text = text.replace(",", "")
    text = text.replace("$", "").replace("%", "")
    if text.endswith("."): 
        text = text[:-1]
    return text.strip()


def parse_number(text: str):
    """
    Parse number from text, handling percentages.
    
    Examples:
      "42" -> (42.0, False)
      "50%" -> (50.0, True)
      "3.14" -> (3.14, False)
    
    Args:
        text: Text containing a number
    
    Returns:
        Tuple of (number, is_percentage) or (None, False) if parsing fails
    """
    clean_text = text.replace(",", "")
    pattern = r'(-?\d+\.?\d*|-?\.\d+)(%)?'
    match = re.search(pattern, clean_text)
    if match:
        try:
            return float(match.group(1)), bool(match.group(2))
        except ValueError:
            pass
    return None, False


def check_equality(pred: str, gt: str) -> bool:
    """
    Check if predicted answer equals ground truth.
    
    Handles:
      - String match: "42" == "42"
      - Numeric comparison: "42.0" == "42"
      - Fractions: "0.5" == "1/2" == "\\frac{1}{2}"
      - Percentages: "50%" == "0.5"
      - Tuples: "(1,2)" == "(1, 2)"
      - Lists: "[1,2,3]" == "[1, 2, 3]"
    
    Examples:
      check_equality("42", "42.0") -> True
      check_equality("1/2", "0.5") -> True
      check_equality("50%", "0.5") -> True
      check_equality("(1,2)", "(1, 2)") -> True
    
    Args:
        pred: Predicted answer
        gt: Ground truth answer
    
    Returns:
        True if answers match, False otherwise
    """
    # Resolve fractions first
    pred = resolve_fractions(pred)
    gt = resolve_fractions(gt)
    
    # String match
    pred_norm = normalize_string(pred)
    gt_norm = normalize_string(gt)
    
    if pred_norm == gt_norm: 
        return True
    
    # Numeric comparison
    pred_num, pred_is_pct = parse_number(pred)
    gt_num, gt_is_pct = parse_number(gt)
    
    if pred_num is None or gt_num is None: 
        return False
    
    def is_close(a, b): 
        return abs(a - b) < 1e-6
    
    # Both same format (both % or both not %)
    if pred_is_pct == gt_is_pct: 
        return is_close(pred_num, gt_num)
    
    # Handle percentage conversions
    if pred_is_pct and not gt_is_pct: 
        return is_close(pred_num, gt_num) or is_close(pred_num/100.0, gt_num)
    if gt_is_pct and not pred_is_pct: 
        return is_close(gt_num, pred_num) or is_close(gt_num/100.0, pred_num)
    
    return False


def validate_has_separator(text: str, scenario: str = "easy") -> bool:
    """
    Check if text has the expected answer separator for the dataset.
    
    Args:
        text: Model output text
        scenario: Dataset difficulty level
    
    Returns:
        True if separator found, False otherwise
    """
    if not text:
        return False
    
    text = str(text)
    
    if scenario == 'easy':  # GSM8K requires ####
        return "####" in text
    
    if scenario in ['medium', 'hard']:  # MATH prefers \boxed{}
        has_boxed = "\\boxed{" in text
        has_separator = "####" in text
        return has_boxed or has_separator
    
    return False


if __name__ == "__main__":
    """Test suite for answer extraction and comparison"""
    
    print("🧪 Testing Answer Extraction & Comparison\n")
    
    test_cases = [
        # (predicted, ground_truth, should_match, description)
        ("42", "42", True, "Exact match"),
        ("42.0", "42", True, "Float vs int"),
        ("1,000", "1000", True, "Comma handling"),
        ("1/2", "0.5", True, "Fraction to decimal"),
        ("\\frac{1}{2}", "0.5", True, "LaTeX fraction"),
        ("50%", "0.5", True, "Percentage"),
        ("(1,2)", "(1, 2)", True, "Tuple with spaces"),
        ("[1,2,3]", "[1, 2, 3]", True, "List with spaces"),
        ("42", "43", False, "Different numbers"),
        ("1/2", "1/3", False, "Different fractions"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gt, expected, desc in test_cases:
        result = check_equality(pred, gt)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} {desc}")
        print(f"   Pred: '{pred}' | GT: '{gt}' | Match: {result}")
        if result != expected:
            print(f"   ⚠️  Expected: {expected}, Got: {result}")
        print()
    
    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 60)