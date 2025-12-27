import unittest
import sys
import os
import json
from unittest.mock import patch, mock_open, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies to avoid import errors
sys.modules["whisper"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()

# Now it is safe to import
from main_video_editor import extract_json_from_text, convert_timestamp_format, convert_gemini_response_to_cut_format

class TestMainVideoEditor(unittest.TestCase):

    def test_extract_json_from_text_simple(self):
        text = '{"ranges_to_delete": [{"start": "00:00:01", "end": "00:00:02"}]}'
        result = extract_json_from_text(text)
        self.assertEqual(result['ranges_to_delete'][0]['start'], "00:00:01")

    def test_extract_json_from_text_markdown(self):
        text = '```json\n{"ranges_to_delete": [{"start": "00:00:01", "end": "00:00:02"}]}\n```'
        result = extract_json_from_text(text)
        self.assertEqual(result['ranges_to_delete'][0]['start'], "00:00:01")

    def test_extract_json_from_text_with_noise(self):
        text = 'Here is the json:\n```json\n{"ranges_to_delete": []}\n```\nHope it helps.'
        result = extract_json_from_text(text)
        self.assertIsNotNone(result)
        self.assertIn("ranges_to_delete", result)

    def test_extract_json_from_text_invalid(self):
        text = 'No json here'
        result = extract_json_from_text(text)
        self.assertIsNone(result)

    def test_convert_timestamp_format(self):
        # Already correct
        self.assertEqual(convert_timestamp_format("00:01:02"), "00:01:02")
        # SRT format
        self.assertEqual(convert_timestamp_format("00:01:02,123"), "00:01:02")
        # Missing leading zeros (if supported by function logic)
        self.assertEqual(convert_timestamp_format("1:2:3"), "01:02:03")
        # Edge cases - should raise ValueError
        with self.assertRaises(ValueError):
            convert_timestamp_format("invalid")

    def test_convert_gemini_response_to_cut_format(self):
        gemini_content = '{"ranges_to_delete": [{"start": "00:00:01,000", "end": "00:00:02,000"}]}'
        
        with patch("builtins.open", mock_open(read_data=gemini_content)) as m_open:
            with patch("os.path.exists", return_value=True):
                # We need to handle the opening of the OUTPUT file too
                # The function opens gemini_path (read) and output_path (write)
                
                output_path = convert_gemini_response_to_cut_format("gemini_response.txt")
                
                self.assertIsNotNone(output_path)
                self.assertTrue(output_path.endswith("_ranges.json"))
                
                # Capture what was written
                handle = m_open()
                written_content = ""
                for call in handle.write.call_args_list:
                    written_content += call[0][0]
                
                # The function converts SRT timestamps to Simple timestamps
                self.assertIn('"start": "00:00:01"', written_content)
                self.assertIn('"end": "00:00:02"', written_content)

if __name__ == '__main__':
    unittest.main()
