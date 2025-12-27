import unittest
import sys
import os
from unittest.mock import patch, mock_open

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_utils import parse_time_to_ms, format_ms_to_srt, clean_srt_response, get_api_key

class TestCommonUtils(unittest.TestCase):
    
    def test_parse_time_to_ms(self):
        # Happy paths
        self.assertEqual(parse_time_to_ms("00:00:01,000"), 1000)
        self.assertEqual(parse_time_to_ms("00:01:00,000"), 60000)
        self.assertEqual(parse_time_to_ms("01:00:00,000"), 3600000)
        self.assertEqual(parse_time_to_ms("00:00:00,500"), 500)
        
        # Dot separator
        self.assertEqual(parse_time_to_ms("00:00:01.000"), 1000)
        
        # No ms
        self.assertEqual(parse_time_to_ms("00:00:01"), 1000)
        
        # Flexible parts (MM:SS)
        self.assertEqual(parse_time_to_ms("01:00"), 60000)
        
        # Whitespace
        self.assertEqual(parse_time_to_ms(" 00:00:01,000 "), 1000)
        
        # Invalid / Fallback to 0 if format mismatch logic in original code
        # Original code: if len(parts) not 2 or 3 -> returns 0? No, returns None or crashes?
        # looking at code:
        # if len(parts) == 3: ... elif len(parts) == 2: ... else: return 0
        self.assertEqual(parse_time_to_ms("invalid"), 0)

    def test_format_ms_to_srt(self):
        self.assertEqual(format_ms_to_srt(1000), "00:00:01,000")
        self.assertEqual(format_ms_to_srt(500), "00:00:00,500")
        self.assertEqual(format_ms_to_srt(60000), "00:01:00,000")
        self.assertEqual(format_ms_to_srt(3661000), "01:01:01,000")
        
    def test_clean_srt_response(self):
        text = "```srt\n1\n00:00:01,000 --> 00:00:02,000\nHello\n```"
        expected = "1\n00:00:01,000 --> 00:00:02,000\nHello"
        self.assertEqual(clean_srt_response(text), expected)
        
        text2 = "1\n00:00:01,000 --> 00:00:02,000\nHello"
        self.assertEqual(clean_srt_response(text2), "1\n00:00:01,000 --> 00:00:02,000\nHello")
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_env_key"})
    def test_get_api_key_env(self):
        self.assertEqual(get_api_key(), "test_env_key")
        
    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_file(self):
        # We need to ensure os.environ is empty of GEMINI_API_KEY
        with patch("builtins.open", mock_open(read_data="test_file_key")):
            with patch("os.path.exists", return_value=True):
                 self.assertEqual(get_api_key(), "test_file_key")
                 
    @patch.dict(os.environ, {}, clear=True)
    @patch("os.path.exists", return_value=False)
    @patch("builtins.input", side_effect=["user_key", "n"])
    @patch("builtins.print")  # Mock print to keep stdout clean
    def test_get_api_key_user_input_no_save(self, mock_print, mock_input, mock_exists):
        # Mocks input returning "user_key" then "n" (for save to env)
        self.assertEqual(get_api_key(), "user_key")

if __name__ == '__main__':
    unittest.main()
