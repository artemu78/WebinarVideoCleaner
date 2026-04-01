import unittest
import sys
import os
import json
from unittest.mock import patch, mock_open, MagicMock, ANY

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMainVideoEditor(unittest.TestCase):
    
    def setUp(self):
        self.module_patcher = patch.dict(sys.modules, {
            "whisper": MagicMock(),
            "google": MagicMock(),
            "google.genai": MagicMock(),
            "google.genai.types": MagicMock(),
            "google.genai.errors": MagicMock(),
            "dotenv": MagicMock(),
            "moviepy": MagicMock(),
            "delivery_metrics": MagicMock()
        })
        self.module_patcher.start()

    def tearDown(self):
        self.module_patcher.stop()

    def test_extract_json_from_text_simple(self):
        from main_video_editor import extract_json_from_text
        text = '{"ranges_to_delete": [{"start": "00:00:01", "end": "00:00:02"}]}'
        result = extract_json_from_text(text)
        self.assertEqual(result['ranges_to_delete'][0]['start'], "00:00:01")

    def test_convert_timestamp_format(self):
        from main_video_editor import convert_timestamp_format
        self.assertEqual(convert_timestamp_format("00:01:02"), "00:01:02")
        self.assertEqual(convert_timestamp_format("00:01:02,123"), "00:01:02")

    def test_convert_gemini_response_to_cut_format(self):
        from main_video_editor import convert_gemini_response_to_cut_format
        gemini_content = '{"ranges_to_delete": [{"start": "00:00:01,000", "end": "00:00:02,000"}]}'
        with patch("builtins.open", mock_open(read_data=gemini_content)) as m_open:
            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: "_ranges.json" not in p
                output_path = convert_gemini_response_to_cut_format("gemini_response.txt")
                self.assertIsNotNone(output_path)

    @patch('main_video_editor.delivery_metrics')
    @patch('main_video_editor.check_srt_alignment')
    @patch('main_video_editor.transcribe_to_srt')
    @patch('main_video_editor.correct_srt_errors')
    @patch('main_video_editor.audio_cleaner')
    @patch('main_video_editor.cut_mp4')
    @patch('main_video_editor.apply_cuts_to_srt')
    @patch('main_video_editor.generate_chapters')
    @patch('main_video_editor.common_utils')
    @patch('builtins.input')
    @patch('os.path.exists')
    def test_main_interactive_defaults(self, mock_exists, mock_input, 
                                     mock_utils, mock_chapters, mock_apply, mock_cut, 
                                     mock_cleaner, mock_correct, mock_transcribe, mock_check, mock_metrics):
        from main_video_editor import main
        mock_exists.return_value = True
        
        mock_input.side_effect = ["video.mp4", "1", "", "", "", ""]
        
        mock_transcribe.main.return_value = (os.path.abspath("video.srt"), "en")
        mock_correct.process_srt_correction.return_value = os.path.abspath("video_corrected.srt")
        mock_cleaner.process_srt_file.return_value = os.path.abspath("gemini_response.txt")
        mock_utils.get_total_gemini_cost.return_value = 0.0
        mock_check.check_alignment.return_value = True
        
        with patch('main_video_editor.convert_gemini_response_to_cut_format') as mock_convert:
            mock_convert.return_value = os.path.abspath("ranges.json")
            mock_cut.process_video.return_value = os.path.abspath("video_cleaned.mp4")
            mock_apply.main.return_value = os.path.abspath("video_cleaned.srt")
            mock_chapters.generate_chapters.return_value = os.path.abspath("chapters.txt")
            mock_metrics.generate_delivery_metrics.return_value = os.path.abspath("metrics.html")
            
            def exists_side_effect(path):
                if path.startswith("/") and any(m in os.path.basename(path) for m in ["video.srt", "video_corrected.srt", "gemini_response.txt", "ranges.json", "video_cleaned.mp4", "video_cleaned.srt", "chapters.txt", "metrics.html"]):
                    return True
                if any(m in path for m in ["cleaned", "corrected", "ranges.json", "chapters.txt", "metrics.html", "gemini_response.txt"]):
                    return False
                return True
            mock_exists.side_effect = exists_side_effect

            main()
            mock_transcribe.main.assert_called_once()
            mock_chapters.generate_chapters.assert_called_once()

    @patch('main_video_editor.delivery_metrics')
    @patch('main_video_editor.check_srt_alignment')
    @patch('main_video_editor.transcribe_to_srt')
    @patch('main_video_editor.correct_srt_errors')
    @patch('main_video_editor.audio_cleaner')
    @patch('main_video_editor.cut_mp4')
    @patch('main_video_editor.apply_cuts_to_srt')
    @patch('main_video_editor.generate_chapters')
    @patch('main_video_editor.common_utils')
    @patch('builtins.input')
    @patch('os.path.exists')
    def test_main_interactive_specifics(self, mock_exists, mock_input, 
                                     mock_utils, mock_chapters, mock_apply, mock_cut, 
                                     mock_cleaner, mock_correct, mock_transcribe, mock_check, mock_metrics):
        from main_video_editor import main
        def exists_side_effect(path):
            if path.startswith("/") and any(m in os.path.basename(path) for m in ["video.srt", "video_corrected.srt", "gemini_response.txt", "ranges.json", "video_cleaned.mp4", "video_cleaned.srt", "chapters.txt", "metrics.html"]):
                return True
            if any(m in path for m in ["cleaned", "corrected", "ranges.json", "chapters.txt", "metrics.html", "gemini_response.txt"]):
                return False
            return True
        mock_exists.side_effect = exists_side_effect
        
        mock_input.side_effect = ["video.mp4", "1", "Topic", "y", "small", "ru"]
        
        mock_check.check_alignment.return_value = True
        mock_transcribe.main.return_value = (os.path.abspath("video.srt"), "en")
        mock_correct.process_srt_correction.return_value = os.path.abspath("video_corrected.srt")
        mock_cleaner.process_srt_file.return_value = os.path.abspath("gemini_response.txt")
        mock_utils.get_total_gemini_cost.return_value = 0.5
        
        with patch('main_video_editor.convert_gemini_response_to_cut_format') as mock_convert:
            mock_convert.return_value = os.path.abspath("ranges.json")
            mock_cut.process_video.return_value = os.path.abspath("video_cleaned.mp4")
            mock_apply.main.return_value = os.path.abspath("video_cleaned.srt")
            mock_chapters.generate_chapters.return_value = os.path.abspath("chapters.txt")
            mock_metrics.generate_delivery_metrics.return_value = os.path.abspath("metrics.html")
            
            main()
            mock_transcribe.main.assert_called_once()
            mock_chapters.generate_chapters.assert_called_once()

if __name__ == '__main__':
    unittest.main()
