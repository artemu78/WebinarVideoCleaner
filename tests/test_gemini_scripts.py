import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define real classes for things that need to be caught
class MockClientError(Exception): pass

class TestGeminiScripts(unittest.TestCase):
    
    def setUp(self):
        # Safety mock for all scripts
        self.module_patcher = patch.dict(sys.modules, {
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": MagicMock(),
            "google.genai.errors": MagicMock(),
            "dotenv": MagicMock()
        })
        self.module_patcher.start()
        sys.modules["google.genai.errors"].ClientError = MockClientError
        
        self.patcher_copy = patch("shutil.copy2")
        self.mock_copy = self.patcher_copy.start()
        
    def tearDown(self):
        self.patcher_copy.stop()
        self.module_patcher.stop()
        
    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    def test_audio_cleaner_process(self, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import audio_cleaner
        mock_exists.side_effect = lambda p: "gemini_response" not in p
        
        mock_client = MagicMock()
        with patch("google.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            mock_client.files.get.return_value = mock_file
            
            mock_response = MagicMock()
            mock_response.text = '{"ranges_to_delete": []}'
            mock_client.models.generate_content.return_value = mock_response
            
            # Manually fix audio_cleaner references if it was already imported
            with patch("audio_cleaner.calculate_gemini_cost", return_value=(0.0001, 100, 50)):
                with patch("builtins.open", mock_open()):
                    output = audio_cleaner.process_srt_file("test.srt")
                    self.assertIn("gemini_response.txt", output)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    def test_audio_cleaner_process_with_audio(self, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import audio_cleaner
        mock_exists.side_effect = lambda p: "gemini_response" not in p
        
        mock_client = MagicMock()
        with patch("google.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            
            mock_response = MagicMock()
            mock_response.text = '{"ranges_to_delete": []}'
            mock_client.models.generate_content.return_value = mock_response
            
            with patch("audio_cleaner.calculate_gemini_cost", return_value=(0.0001, 100, 50)):
                with patch("builtins.open", mock_open()):
                    output = audio_cleaner.process_srt_file("test.srt", audio_path="test.mp3")
                    self.assertEqual(mock_safe_upload.call_count, 2)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_correct_transcription_process(self, mock_remove, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import correct_srt_errors
        mock_exists.side_effect = lambda p: "corrected_by_gemini" not in p
        
        mock_client = MagicMock()
        with patch("google.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            
            mock_response = MagicMock()
            mock_response.text = '[{"id": "1", "text": "Corrected text"}]'
            mock_client.models.generate_content.return_value = mock_response
            
            mock_srt_content = "1\n00:01:00,000 --> 00:01:02,000\nOriginal text\n\n"
            with patch("correct_srt_errors.calculate_gemini_cost", return_value=(0.0001, 100, 50)):
                with patch("builtins.open", mock_open(read_data=mock_srt_content)):
                    output = correct_srt_errors.process_srt_correction("test.srt", language="en")
                    self.assertIn("corrected_by_gemini.srt", output)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    def test_generate_chapters_process(self, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import generate_chapters
        mock_exists.side_effect = lambda p: "_chapters.txt" not in p
        
        mock_client = MagicMock()
        with patch("google.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            
            mock_response = MagicMock()
            mock_response.text = "00:00:00 - Intro"
            mock_client.models.generate_content.return_value = mock_response
            
            with patch("generate_chapters.calculate_gemini_cost", return_value=(0.0001, 100, 50)):
                with patch("builtins.open", mock_open()):
                    output = generate_chapters.generate_chapters("test.srt")
                    self.assertIn("_chapters.txt", output)

    @patch("common_utils.get_api_key", return_value="fake_key")
    @patch("delivery_metrics.get_api_key")
    @patch("delivery_metrics.calculate_gemini_cost")
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    def test_generate_delivery_metrics_process(self, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key, mock_common_get_key):
        import delivery_metrics
        mock_exists.side_effect = lambda p: "_delivery_metrics.html" not in p

        mock_client = MagicMock()
        with patch("google.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file

            mock_response = MagicMock()
            mock_response.text = "<h1>Delivery Metrics Report</h1>"
            mock_client.models.generate_content.return_value = mock_response

            mock_srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            with patch("delivery_metrics.calculate_gemini_cost", return_value=(0.0001, 100, 50)):
                with patch("builtins.open", mock_open(read_data=mock_srt_content)):
                    output = delivery_metrics.generate_delivery_metrics("test.srt", "test_chapters.txt")
                    self.assertIsNotNone(output)
                    self.assertIn("_delivery_metrics.html", output)

if __name__ == '__main__':
    unittest.main()
