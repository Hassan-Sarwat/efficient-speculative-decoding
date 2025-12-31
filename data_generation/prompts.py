
SUMMARIZATION_PROMPT = """
You are an expert at compressing reasoning into a "Chain of Draft" format.
Task: Summarize the reasoning for the given Question into minimal, high-density steps.

Constraints:
1. Use the minimum number of steps necessary (typically 1-5). Don't force 3 steps if 1 suffices.
2. Each step should be ONE atomic operation or calculation.
3. Preserve ALL key numbers and operations.
4. STRICTLY NO TEXT LABELS inside steps. Only numbers, math symbols, and units (e.g., "7 * $8.75 = $61.25", not "Cost: $61.25").
5. Format: <draft> 1. Step one -> 2. Step two </draft>

Examples:

Good (Pure Math):
<draft> 1. 7 * $8.75 = $61.25 -> 2. $61.25 + $7.22 = $68.47 -> 3. $68.47 + $11.53 = $80 </draft>

Bad (Verbose/Abstract):
<draft> 1. Calculate cost of robots -> 2. Add the tax amount -> 3. Subtract from total cash </draft>

Bad (Contains Text Labels):
<draft> 1. Cost: 61.25 -> 2. Total: 68.47 -> 3. Change: 11.53 </draft>

Question:
{question}

Original Reasoning:
{raw_logic}

Draft Summary:
"""

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
