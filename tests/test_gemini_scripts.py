import unittest
import sys
import os
import importlib
import urllib.error
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define real classes for things that need to be caught
class MockClientError(Exception): pass

class TestAIScripts(unittest.TestCase):
    
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
        
    @patch("common_utils.get_gemini_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    def test_audio_cleaner_process_gemini(self, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import audio_cleaner
        mock_exists.side_effect = lambda p: "gemini_response" not in p
        
        mock_client = MagicMock()
        with patch("common_utils.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            mock_client.files.get.return_value = mock_file
            
            mock_response = MagicMock()
            mock_response.text = '{"ranges_to_delete": []}'
            mock_client.models.generate_content.return_value = mock_response
            
            with patch("builtins.open", mock_open()):
                output = audio_cleaner.process_srt_file("test.srt", provider="gemini")
                self.assertIn("gemini_response.txt", output)

    @patch("common_utils.generate_content")
    @patch("os.path.exists")
    def test_audio_cleaner_process_openrouter(self, mock_exists, mock_gen):
        import audio_cleaner
        with patch.dict(
            os.environ,
            {"AUDIO_CLEANER_OPENROUTER_INFERENCE_PROVIDER": "deepinfra"},
            clear=False,
        ):
            importlib.reload(audio_cleaner)
            mock_exists.side_effect = lambda p: "gemini_response" not in p
            mock_gen.return_value = '{"ranges_to_delete": []}'

            with patch("builtins.open", mock_open(read_data="SRT content")):
                output = audio_cleaner.process_srt_file("test.srt", provider="openrouter")
        self.assertIn("gemini_response.txt", output)
        mock_gen.assert_called_once()
        self.assertEqual(mock_gen.call_args.kwargs["inference_provider"], "deepinfra")

    @patch("common_utils.get_gemini_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_correct_transcription_process_gemini(self, mock_remove, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import correct_srt_errors
        importlib.reload(correct_srt_errors)
        mock_exists.side_effect = lambda p: "_corrected_by_gemini" not in p
        
        mock_client = MagicMock()
        with patch("common_utils.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            
            mock_response = MagicMock()
            mock_response.text = '[{"id": "1", "text": "Corrected text"}]'
            mock_client.models.generate_content.return_value = mock_response
            
            mock_srt_content = "1\n00:01:00,000 --> 00:01:02,000\nOriginal text\n\n"
            with patch("builtins.open", mock_open(read_data=mock_srt_content)):
                output = correct_srt_errors.process_srt_correction("test.srt", language="en", provider="gemini")
                self.assertIn("_corrected_by_gemini.srt", output)

    @patch("common_utils.generate_content", return_value='[{"id": "1", "text": "Fixed"}]')
    @patch("os.path.exists")
    def test_correct_transcription_openrouter(self, mock_exists, mock_gen):
        import correct_srt_errors
        with patch.dict(
            os.environ,
            {"CORRECT_SRT_ERRORS_OPENROUTER_INFERENCE_PROVIDER": "baidu/fp8"},
            clear=False,
        ):
            importlib.reload(correct_srt_errors)

            mock_exists.side_effect = lambda p: "_corrected_by_openrouter" not in p
            mock_srt_content = "1\n00:01:00,000 --> 00:01:02,000\nOriginal text\n\n"
            with patch("builtins.open", mock_open(read_data=mock_srt_content)):
                output = correct_srt_errors.process_srt_correction(
                    "test.srt", language="ru", provider="openrouter"
                )
        self.assertIn("_corrected_by_openrouter.srt", output)
        mock_gen.assert_called()
        self.assertEqual(mock_gen.call_args.kwargs["inference_provider"], "baidu/fp8")

    @patch("common_utils.generate_content", return_value='[{"id": "1", "text": "Fixed"}]')
    @patch("os.path.exists")
    def test_correct_transcription_does_not_pin_provider_for_another_model(self, mock_exists, mock_gen):
        import correct_srt_errors
        with patch.dict(
            os.environ,
            {"CORRECT_SRT_ERRORS_OPENROUTER_INFERENCE_PROVIDER": "baidu/fp8"},
            clear=False,
        ):
            importlib.reload(correct_srt_errors)
            mock_exists.side_effect = lambda p: "_corrected_by_openrouter" not in p
            mock_srt_content = "1\n00:01:00,000 --> 00:01:02,000\nOriginal text\n\n"
            with patch("builtins.open", mock_open(read_data=mock_srt_content)):
                correct_srt_errors.process_srt_correction(
                    "test.srt",
                    language="ru",
                    provider="openrouter",
                    correction_model="google/gemini-2.5-flash",
                )

        self.assertIsNone(mock_gen.call_args.kwargs["inference_provider"])

    @patch("os.path.exists")
    def test_correct_transcription_stops_on_openrouter_forbidden(self, mock_exists):
        import correct_srt_errors
        importlib.reload(correct_srt_errors)
        mock_exists.side_effect = lambda p: "_corrected_by_openrouter" not in p
        mock_srt_content = "1\n00:01:00,000 --> 00:01:02,000\nOriginal text\n\n"

        with (
            patch("builtins.open", mock_open(read_data=mock_srt_content)),
            patch.object(
                correct_srt_errors,
                "generate_content",
                side_effect=urllib.error.HTTPError("https://openrouter.ai", 403, "Forbidden", None, None),
            ) as mock_gen,
            patch.object(correct_srt_errors, "write_srt") as mock_write,
        ):
            output = correct_srt_errors.process_srt_correction(
                "test.srt", language="ru", provider="openrouter"
            )

        self.assertIsNone(output)
        mock_gen.assert_called_once()
        mock_write.assert_not_called()

    @patch("os.path.exists")
    def test_correct_transcription_does_not_write_when_all_batches_fail(self, mock_exists):
        import correct_srt_errors
        importlib.reload(correct_srt_errors)
        mock_exists.side_effect = lambda p: "_corrected_by_openrouter" not in p
        mock_srt_content = "\n\n".join(
            f"{index}\n00:01:00,000 --> 00:01:02,000\nOriginal text {index}"
            for index in range(1, 102)
        )

        with (
            patch("builtins.open", mock_open(read_data=mock_srt_content)),
            patch.object(
                correct_srt_errors,
                "generate_content",
                side_effect=RuntimeError("OpenRouter request failed after 3 attempts"),
            ) as mock_gen,
            patch.object(correct_srt_errors, "write_srt") as mock_write,
        ):
            output = correct_srt_errors.process_srt_correction(
                "test.srt", language="ru", provider="openrouter"
            )

        self.assertIsNone(output)
        self.assertEqual(mock_gen.call_count, 2)
        mock_write.assert_not_called()

    @patch("common_utils.get_gemini_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    def test_generate_chapters_process_gemini(self, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import generate_chapters
        importlib.reload(generate_chapters)
        mock_exists.side_effect = lambda p: "_chapters.txt" not in p
        
        mock_client = MagicMock()
        with patch("generate_chapters.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            mock_client.files.get.return_value = mock_file
            
            mock_response = MagicMock()
            mock_response.text = "00:00:00 - Intro"
            mock_client.models.generate_content.return_value = mock_response
            
            with patch("builtins.open", mock_open(read_data="SRT content")):
                output = generate_chapters.generate_chapters("test.srt", provider="gemini")
                self.assertIn("_chapters.txt", output)

    @patch("common_utils.generate_content")
    @patch("os.path.exists")
    def test_generate_chapters_process_openrouter(self, mock_exists, mock_gen):
        import generate_chapters
        with patch.dict(
            os.environ,
            {"GENERATE_CHAPTERS_OPENROUTER_INFERENCE_PROVIDER": "deepinfra"},
            clear=False,
        ):
            importlib.reload(generate_chapters)
            mock_exists.side_effect = lambda p: "_chapters.txt" not in p
            mock_gen.return_value = "00:00:00 - Intro"

            with patch("builtins.open", mock_open(read_data="SRT content")):
                output = generate_chapters.generate_chapters("test.srt", provider="openrouter")
        self.assertIn("_chapters.txt", output)
        mock_gen.assert_called_once()
        self.assertEqual(mock_gen.call_args.kwargs["inference_provider"], "deepinfra")

    @patch("common_utils.get_gemini_api_key", return_value="fake_key")
    @patch("common_utils.calculate_gemini_cost", return_value=(0.0001, 100, 50))
    @patch("common_utils.safe_upload")
    @patch("os.path.exists")
    def test_generate_delivery_metrics_process_gemini(self, mock_exists, mock_safe_upload, mock_calc_cost, mock_get_key):
        import delivery_metrics
        importlib.reload(delivery_metrics)
        mock_exists.side_effect = lambda p: "_delivery_metrics.html" not in p

        mock_client = MagicMock()
        # Patch delivery_metrics.genai.Client because it imports genai at top level
        with patch("delivery_metrics.genai.Client", return_value=mock_client):
            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_safe_upload.return_value = mock_file
            mock_client.files.get.return_value = mock_file

            mock_response = MagicMock()
            mock_response.text = "<h1>Report</h1>"
            mock_client.models.generate_content.return_value = mock_response

            mock_srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            with patch("builtins.open", mock_open(read_data=mock_srt_content)):
                output = delivery_metrics.generate_delivery_metrics("test.srt", "test_chapters.txt", provider="gemini")
                self.assertIsNotNone(output)
                self.assertIn("_delivery_metrics.html", output)


    @patch("common_utils.generate_content")
    @patch("os.path.exists")
    def test_generate_delivery_metrics_process_openrouter(self, mock_exists, mock_gen):
        import delivery_metrics
        with patch.dict(
            os.environ,
            {"DELIVERY_METRICS_OPENROUTER_INFERENCE_PROVIDER": "deepinfra"},
            clear=False,
        ):
            importlib.reload(delivery_metrics)
            mock_exists.side_effect = lambda p: "_delivery_metrics.html" not in p
            mock_gen.return_value = "<h1>Report</h1>"

            mock_srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            with patch("builtins.open", mock_open(read_data=mock_srt_content)):
                output = delivery_metrics.generate_delivery_metrics("test.srt", "test_chapters.txt", provider="openrouter")
        self.assertIsNotNone(output)
        self.assertIn("_delivery_metrics.html", output)
        mock_gen.assert_called_once()
        self.assertEqual(mock_gen.call_args.kwargs["inference_provider"], "deepinfra")
if __name__ == '__main__':
    unittest.main()
