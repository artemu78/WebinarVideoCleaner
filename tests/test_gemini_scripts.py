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
sys.modules["google.genai.errors"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

# Import scripts after mocking
from audio_cleaner import process_srt_file
from correct_transcription import process_srt_correction
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
    def test_correct_transcription_process(self, mock_exists, mock_get_key):
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
        mock_response.text = "1\n00:01:00,000 --> 00:01:02,000\nCorrected text"
        mock_client.models.generate_content.return_value = mock_response
        
        with patch("builtins.open", mock_open()) as m_open:
            output = process_srt_correction("test.srt", language="en")
            
            mock_client.files.upload.assert_called()
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
        mock_client.models.generate_content.return_value = mock_response
        
        with patch("builtins.open", mock_open()) as m_open:
            output = generate_chapters("test.srt")
            
            mock_client.files.upload.assert_called()
            mock_client.models.generate_content.assert_called()
            self.assertIn("_chapters.txt", output)

if __name__ == '__main__':
    unittest.main()
