FULL_PROMPT = """
{question}

IMPORTANT: After your reasoning, output ONLY the final numerical answer in this exact format:
#### [number]

Example: #### 42

Do not add any explanation, breakdown, or text after the #### marker.

Avoid usage of calculator tags such as '<< >>' or latex."""

SUMMARIZATION_PROMPT = """
You are an expert at compressing reasoning into a "Chain of Draft" format.
Task: Summarize the reasoning for the given Question into minimal, high-density steps.

Constraints:
1. Use the minimum number of steps necessary (typically 1-5). Don't force 3 steps if 1 suffices.
2. Each step should be ONE atomic operation or calculation.
3. Preserve ALL key numbers and operations.
4. STRICTLY NO TEXT LABELS inside steps. Only numbers, math symbols, and units.
5. Format: <draft> 1. Step one -> 2. Step two </draft>

CRITICAL: After the </draft> tag, output ONLY:
#### [number]

where [number] is the final numerical answer.

Do NOT add explanations, breakdowns, or formatted text after ####.

Examples:

Good:
<draft> 1. 7 * $8.75 = $61.25 -> 2. $61.25 + $7.22 = $68.47 </draft> #### 68.47

Bad:
<draft> 1. Calculate... </draft> #### Here is the breakdown: ...

Question:
{question}

Original Reasoning:
{raw_logic}

Draft Summary (with ONLY number after ####):
"""

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
