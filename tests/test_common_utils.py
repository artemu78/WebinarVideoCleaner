import unittest
import sys
import os
import json
import http.client
import urllib.error
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
        
        # Float input
        self.assertEqual(format_ms_to_srt(1234.56), "00:00:01,234")
        
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

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gemini_test_key"}, clear=True)
    @patch("common_utils.genai")
    def test_generate_content_gemini(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = MagicMock(text="hello from gemini")

        response_text = common_utils.generate_content(
            provider="gemini",
            model="gemini-3.1-flash-lite-preview",
            prompt="translate me",
        )

        self.assertEqual(response_text, "hello from gemini")
        mock_genai.Client.assert_called_once_with(api_key="gemini_test_key")
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-3.1-flash-lite-preview",
            contents="translate me",
            config=None
        )

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or_test_key"}, clear=True)
    def test_get_openrouter_api_key_env(self):
        self.assertEqual(common_utils.get_openrouter_api_key(), "or_test_key")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or_test_key"}, clear=True)
    @patch("common_utils.urllib.request.urlopen")
    def test_generate_content_openrouter(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"choices":[{"message":{"content":"hello from openrouter"}}],'
            b'"usage":{"prompt_tokens":1000,"completion_tokens":500,'
            b'"total_tokens":1500,"cost":0.012345}}'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_response

        response_text = common_utils.generate_content(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            prompt="translate me",
        )

        self.assertEqual(response_text, "hello from openrouter")
        self.assertAlmostEqual(get_total_gemini_cost(), 0.012345)
        self.assertTrue(mock_urlopen.called)
        # Verify that it was called with the right headers
        args, kwargs = mock_urlopen.call_args
        request = args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer or_test_key")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or_test_key"}, clear=True)
    @patch("common_utils.time.sleep")
    @patch("common_utils.urllib.request.urlopen")
    def test_generate_content_openrouter_reports_null_message_content_without_retrying(
        self, mock_urlopen, mock_sleep
    ):
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"model":"example/model","choices":[{"message":{"content":null},'
            b'"finish_reason":"stop"}]}'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with self.assertRaisesRegex(
            RuntimeError,
            "OpenRouter returned no final message content.*finish_reason='stop'",
        ):
            common_utils.generate_content(
                provider="openrouter",
                model="example/model",
                prompt="translate me",
            )

        mock_urlopen.assert_called_once()
        mock_sleep.assert_not_called()

    @patch.dict(
        os.environ,
        {
            "OPENROUTER_API_KEY": "or_test_key",
            "OPENROUTER_INFERENCE_PROVIDER": "  deepinfra  ",
        },
        clear=True,
    )
    @patch("common_utils.urllib.request.urlopen")
    def test_generate_content_openrouter_ignores_global_inference_provider(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices":[{"message":{"content":"hello"}}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        common_utils.generate_content(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            prompt="translate me",
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertNotIn("provider", payload)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or_test_key"}, clear=True)
    @patch("common_utils.urllib.request.urlopen")
    def test_generate_content_openrouter_uses_explicit_inference_provider(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices":[{"message":{"content":"hello"}}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        common_utils.generate_content(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            prompt="translate me",
            inference_provider="  deepinfra  ",
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["provider"], {"only": ["deepinfra"]})

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or_test_key"}, clear=True)
    @patch("common_utils.time.sleep")
    @patch("common_utils.timed_input")
    @patch("common_utils.urllib.request.urlopen")
    def test_generate_content_openrouter_retries_truncated_response_without_prompting(
        self, mock_urlopen, mock_timed_input, mock_sleep
    ):
        mock_urlopen.side_effect = http.client.IncompleteRead(b'{"partial": true}')
        mock_timed_input.side_effect = AssertionError("OpenRouter retries must not prompt for input")

        with self.assertRaisesRegex(RuntimeError, "OpenRouter request failed after 3 attempts"):
            common_utils.generate_content(
                provider="openrouter",
                model="openai/gpt-4o-mini",
                prompt="translate me",
            )

        self.assertEqual(mock_urlopen.call_count, 3)
        mock_timed_input.assert_not_called()
        self.assertEqual(mock_sleep.call_count, 2)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "or_test_key"}, clear=True)
    @patch("common_utils.time.sleep")
    @patch("common_utils.urllib.request.urlopen")
    def test_generate_content_openrouter_does_not_retry_forbidden_response(
        self, mock_urlopen, mock_sleep
    ):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions", 403, "Forbidden", None, None
        )

        with self.assertRaises(urllib.error.HTTPError) as raised:
            common_utils.generate_content(
                provider="openrouter",
                model="example/model",
                prompt="translate me",
            )

        self.assertEqual(raised.exception.code, 403)
        mock_urlopen.assert_called_once()
        mock_sleep.assert_not_called()

if __name__ == '__main__':
    unittest.main()
