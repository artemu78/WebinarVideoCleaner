import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global mock for google.genai
mock_genai = MagicMock()
# We need to ensure 'google' contains 'genai'
mock_google = MagicMock()
mock_google.genai = mock_genai
sys.modules["google"] = mock_google
sys.modules["google.genai"] = mock_genai
sys.modules["google.genai.types"] = MagicMock() 

# Define a real exception class for ClientError
mock_errors = MagicMock()
class MockClientError(Exception):
    pass
mock_errors.ClientError = MockClientError
sys.modules["google.genai.errors"] = mock_errors
sys.modules["dotenv"] = MagicMock()

# Import scripts after mocking
from audio_cleaner import process_srt_file
from correct_srt_errors import process_srt_correction, parse_srt, write_srt
from generate_chapters import generate_chapters

class TestGeminiScripts(unittest.TestCase):
    
    def setUp(self):
        # Reset mocks between tests
        mock_genai.reset_mock()
        
    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("os.path.exists", return_value=True)
    def test_audio_cleaner_process(self, mock_exists, mock_get_key):
        # Mock client instance
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        # Mock upload
        mock_file = MagicMock()
        mock_file.uri = "http://fake/uri"
        mock_file.name = "files/123"
        mock_file.state.name = "ACTIVE" # Or processing loop logic (ACTIVE != PROCESSING)
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        
        # Mock generate_content
        mock_response = MagicMock()
        mock_response.text = '{"ranges_to_delete": []}'
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_response
        
        # Run
        with patch("builtins.open", mock_open()) as m_open:
            output = process_srt_file("test.srt")
            
            # Verify
            mock_client.files.upload.assert_called()
            mock_client.models.generate_content.assert_called()
            m_open.assert_called() # Should save to file
            self.assertIn("gemini_response.txt", output)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("os.path.exists", return_value=True)
    def test_audio_cleaner_process_with_audio(self, mock_exists, mock_get_key):
        # Mock client instance
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        # Mock upload
        mock_file = MagicMock()
        mock_file.uri = "http://fake/uri"
        mock_file.name = "files/123"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        
        # Mock generate_content
        mock_response = MagicMock()
        mock_response.text = '{"ranges_to_delete": []}'
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_response
        
        # Run
        with patch("builtins.open", mock_open()) as m_open:
            output = process_srt_file("test.srt", audio_path="test.mp3")
            
            # Verify upload called twice (srt + audio)
            self.assertEqual(mock_client.files.upload.call_count, 2)
            mock_client.models.generate_content.assert_called()
            m_open.assert_called() 
            self.assertIn("gemini_response.txt", output)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_correct_transcription_process(self, mock_remove, mock_exists, mock_get_key):
         # Mock client instance
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        # Mock upload
        mock_file = MagicMock()
        mock_file.uri = "http://fake/uri"
        mock_file.name = "files/123"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        
        # Mock generate_content
        mock_response = MagicMock()
        mock_response.text = '[{"id": "1", "text": "Corrected text"}]'
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_response
        
        # Mock file content for parse_srt
        mock_srt_content = "1\n00:01:00,000 --> 00:01:02,000\nOriginal text\n\n"
        
        with patch("builtins.open", mock_open(read_data=mock_srt_content)) as m_open:
            output = process_srt_correction("test.srt", language="en")
            
            mock_client.files.upload.assert_called()
            mock_client.models.generate_content.assert_called()
            self.assertIn("corrected_by_gemini.srt", output)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_correct_transcription_process_with_topic(self, mock_remove, mock_exists, mock_get_key):
         # Mock client instance
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        # Mock upload
        mock_file = MagicMock()
        mock_file.uri = "http://fake/uri"
        mock_file.name = "files/123"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        
        # Mock generate_content
        mock_response = MagicMock()
        mock_response.text = '[{"id": "1", "text": "Corrected text"}]'
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_response
        
        # Mock file content for parse_srt
        mock_srt_content = "1\n00:01:00,000 --> 00:01:02,000\nOriginal text\n\n"
        
        with patch("builtins.open", mock_open(read_data=mock_srt_content)) as m_open:
            output = process_srt_correction("test.srt", language="en", webinar_topic="Rocket Science")
            
            mock_client.files.upload.assert_called()
            # Verify that generate_content was called (we can't easily check prompt content without deeper inspection of call args)
            # But at least we verify it runs with the 3rd argument
            mock_client.models.generate_content.assert_called()
            self.assertIn("corrected_by_gemini.srt", output)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("os.path.exists", return_value=True)
    def test_generate_chapters_process(self, mock_exists, mock_get_key):
         # Mock client instance
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        # Mock upload
        mock_file = MagicMock()
        mock_file.uri = "http://fake/uri"
        mock_file.name = "files/123"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        
        # Mock generate_content
        mock_response = MagicMock()
        mock_response.text = "00:00:00 - Intro"
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_response
        
        with patch("builtins.open", mock_open()) as m_open:
            output = generate_chapters("test.srt")
            
            mock_client.files.upload.assert_called()
            mock_client.models.generate_content.assert_called()
            self.assertIn("_chapters.txt", output)
            
    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("os.path.exists", return_value=True)
    def test_generate_chapters_process_with_topic(self, mock_exists, mock_get_key):
         # Mock client instance
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        # Mock upload
        mock_file = MagicMock()
        mock_file.uri = "http://fake/uri"
        mock_file.name = "files/123"
        mock_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        
        # Mock generate_content
        mock_response = MagicMock()
        mock_response.text = "00:00:00 - Intro"
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_client.models.generate_content.return_value = mock_response
        
        with patch("builtins.open", mock_open()) as m_open:
            output = generate_chapters("test.srt", webinar_topic="Rocket Science")
            
            mock_client.files.upload.assert_called()
            mock_client.models.generate_content.assert_called()
            self.assertIn("_chapters.txt", output)

    def test_parse_srt(self):
        srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nWorld\n"
        with patch("builtins.open", mock_open(read_data=srt_content)):
            blocks = parse_srt("dummy.srt")
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0]['text'], "Hello")
            self.assertEqual(blocks[1]['text'], "World")

    def test_write_srt(self):
        blocks = [
            {'index': "1", 'start': "00:00:01,000", 'end': "00:00:02,000", 'text': "Hello"}
        ]
        with patch("builtins.open", mock_open()) as m:
            write_srt(blocks, "output.srt")
            m.assert_called_with("output.srt", 'w', encoding='utf-8')
            handle = m()
            args_list = handle.write.call_args_list
            full_content = "".join([call[0][0] for call in args_list])
            self.assertIn("00:00:01,000 --> 00:00:02,000", full_content)
            self.assertIn("Hello", full_content)

if __name__ == '__main__':
    unittest.main()
