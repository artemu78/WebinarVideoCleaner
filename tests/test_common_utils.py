import unittest
import sys
import os
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common_utils
from common_utils import parse_time_to_ms, format_ms_to_srt, clean_srt_response, get_api_key, calculate_gemini_cost, get_total_gemini_cost

class TestCommonUtils(unittest.TestCase):
    
    def setUp(self):
        # Reset total cost before each test
        common_utils._TOTAL_GEMINI_COST = 0.0
    
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

    def test_calculate_gemini_cost(self):
        # case 1: Normal usage
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1_000_000
        mock_response.usage_metadata.candidates_token_count = 1_000_000
        
        # Expected:
        # Input: 1M * $0.50 = $0.50
        # Output: 1M * $3.00 = $3.00
        # Total: $3.50
        cost, input_tokens, output_tokens = calculate_gemini_cost(mock_response)
        self.assertAlmostEqual(cost, 3.50)
        self.assertEqual(input_tokens, 1_000_000)
        self.assertEqual(output_tokens, 1_000_000)
        self.assertAlmostEqual(get_total_gemini_cost(), 3.50)
        
        # case 2: Accumulation
        mock_response_2 = MagicMock()
        mock_response_2.usage_metadata.prompt_token_count = 2_000_000 # $1.00
        mock_response_2.usage_metadata.candidates_token_count = 0 # $0.00
        
        cost2, in2, out2 = calculate_gemini_cost(mock_response_2)
        self.assertAlmostEqual(cost2, 1.00)
        self.assertEqual(in2, 2_000_000)
        self.assertEqual(out2, 0)
        self.assertAlmostEqual(get_total_gemini_cost(), 4.50)

    def test_calculate_gemini_cost_edge_cases(self):
        # case 1: No usage metadata
        mock_response_no_meta = MagicMock()
        del mock_response_no_meta.usage_metadata # Make sure attribute doesn't exist raises AttributeError if not set, 
                                                 # but MagicMock creates it by default on access. 
                                                 # We need to explicitly make sure hasattr returns false or it is None
        # Resetting the mock to not have the attribute is tricky with MagicMock as accessing it creates it.
        # Easier way: create a plain object or configure mock spec
        
        class EmptyResponse:
            pass
            
        cost, _, _ = calculate_gemini_cost(EmptyResponse())
        self.assertEqual(cost, 0.0)
        
        # case 2: Usage is None
        mock_response_none_usage = MagicMock()
        mock_response_none_usage.usage_metadata = None
        cost, _, _ = calculate_gemini_cost(mock_response_none_usage)
        self.assertEqual(cost, 0.0)
        
        # case 3: Token counts are None (should be treated as 0)
        mock_response_none_tokens = MagicMock()
        mock_response_none_tokens.usage_metadata.prompt_token_count = None
        mock_response_none_tokens.usage_metadata.candidates_token_count = None
        
        cost, _, _ = calculate_gemini_cost(mock_response_none_tokens)
        self.assertEqual(cost, 0.0)

if __name__ == '__main__':
    unittest.main()
