import unittest
import sys
import os

# Ensure we can import modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_generation.analyze_data import extract_steps
from data_generation.result_processor import parse_candidate_content, FormatConstants

class TestParsers(unittest.TestCase):

    def test_extract_steps_numbered(self):
        """Test extraction of explicitly numbered steps."""
        text = "Step 1: Calculate mass. Step 2: Multiply by gravity."
        steps = extract_steps(text)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0], "Calculate mass.")
        self.assertEqual(steps[1], "Multiply by gravity.")

    def test_extract_steps_newlines(self):
        """Test fallback to newline splitting."""
        text = "First calculation.\n\nSecond calculation."
        steps = extract_steps(text)
        self.assertEqual(len(steps), 2)

    def test_parse_candidate_content_mixed(self):
        """Test parsing a candidate with both thought and text parts."""
        mock_candidate = {
            "content": {
                "parts": [
                    {"text": "I need to integrate.", "thought": True},
                    {"text": "The answer is 5."}
                ]
            }
        }
        logic, answer = parse_candidate_content(mock_candidate)
        self.assertEqual(logic, "I need to integrate.")
        self.assertEqual(answer, "The answer is 5.")

    def test_format_constants(self):
        """Ensure formatting tags don't accidentally drift."""
        raw = "My Draft"
        wrapped = FormatConstants.wrap_draft(raw)
        self.assertIn("<draft>", wrapped)
        self.assertIn("</draft>", wrapped)
        # Test idempotency (shouldn't double wrap)
        wrapped_again = FormatConstants.wrap_draft(wrapped)
        self.assertEqual(wrapped_again.count("<draft>"), 1)

if __name__ == '__main__':
    unittest.main()