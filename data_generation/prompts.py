
# Constants for prompts and safety settings

SUMMARIZATION_PROMPT = """
Summarize the following reasoning process into 3-5 concise steps (max 5 words each).
Format the output as a single line with numbered steps separated by arrows, wrapped in <draft> tags.
Example: <draft> 1. Step one -> 2. Step two -> 3. Step three </draft>

Reasoning:
{raw_logic}
"""

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
