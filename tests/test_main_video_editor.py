import unittest
import sys
import os
import json
from unittest.mock import patch, mock_open, MagicMock, ANY

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies to avoid import errors
sys.modules["whisper"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()
sys.modules["google.genai.errors"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["moviepy"] = MagicMock()

# Now it is safe to import
from main_video_editor import extract_json_from_text, convert_timestamp_format, convert_gemini_response_to_cut_format, main

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
                                     mock_cleaner, mock_correct, mock_transcribe, mock_check):
        # Setup mocks
        mock_exists.return_value = True
        mock_input.side_effect = [
            "video.mp4", # File path
            "1",         # Mode 1 (Full)
            "",          # Topic (skip)
            "",          # Model (default turbo)
            "",          # Language (default auto)
        ]
        
        # Mock return values for internal steps to ensure flow continues
        mock_transcribe.main.return_value = ("video.srt", "en")
        mock_correct.process_srt_correction.return_value = "video_corrected.srt"
        mock_cleaner.process_srt_file.return_value = "gemini_response.txt"
        mock_utils.get_total_gemini_cost.return_value = 0.0
        mock_check.check_alignment.return_value = True
        
        # Determine how to mock convert_gemini_response_to_cut_format since it's imported directly
        # We can mock it in the test using patch.object or patch the imported name in main_video_editor
        with patch('main_video_editor.convert_gemini_response_to_cut_format') as mock_convert:
            mock_convert.return_value = "ranges.json"
            mock_cut.process_video.return_value = "video_cleaned.mp4"
            mock_apply.main.return_value = "video_cleaned.srt"
            mock_chapters.generate_chapters.return_value = "chapters.txt"
            
            # Run main
            main()
            
            # VERIFY changes
            # Check transcribe called with turbo and None (auto-detect)
            mock_transcribe.main.assert_called_once_with(
                file_input="video.mp4",
                model="turbo",          # DEFAULT verification
                max_segment_duration=8.0,
                use_srt=True,
                language=None           # DEFAULT verification
            )
            
            # Verify correct_srt_errors called with webinar topic None
            mock_correct.process_srt_correction.assert_called_once_with(
                os.path.abspath("video.srt"), "en", None
            )

            # Verify generate_chapters called with webinar topic None
            mock_chapters.generate_chapters.assert_called_once_with(
                os.path.abspath("video_cleaned.srt"), language="en", webinar_topic=None
            )
            
            # Verify alignment check called
            mock_check.check_alignment.assert_called_once()

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
                                     mock_cleaner, mock_correct, mock_transcribe, mock_check):
        # Setup mocks
        mock_exists.return_value = True
        mock_input.side_effect = [
            "video.mp4", # File path
            "1",         # Mode 1 (Full)
            "My Topic",  # Topic
            "small",     # Model (small)
            "ru",        # Language (Russian)
        ]
        
        mock_check.check_alignment.return_value = True
        mock_transcribe.main.return_value = ("video.srt", "en")
        mock_correct.process_srt_correction.return_value = "video_corrected.srt"
        mock_cleaner.process_srt_file.return_value = "gemini_response.txt"
        mock_utils.get_total_gemini_cost.return_value = 0.5
        
        with patch('main_video_editor.convert_gemini_response_to_cut_format') as mock_convert:
            mock_convert.return_value = "ranges.json"
            mock_cut.process_video.return_value = "video_cleaned.mp4"
            mock_apply.main.return_value = "video_cleaned.srt"
            mock_chapters.generate_chapters.return_value = "chapters.txt"
            
            # Run main
            main()
            
            # VERIFY changes
            # Check transcribe called with small and ru
            mock_transcribe.main.assert_called_once_with(
                file_input="video.mp4",
                model="small",          # SPECIFIC verification
                max_segment_duration=8.0,
                use_srt=True,
                language="ru"           # SPECIFIC verification
            )
            
            # Verify correct_srt_errors called with correct topic
            mock_correct.process_srt_correction.assert_called_once_with(
                os.path.abspath("video.srt"), "en", "My Topic"
            )

            # Verify generate_chapters called with correct topic
            mock_chapters.generate_chapters.assert_called_once_with(
                os.path.abspath("video_cleaned.srt"), language="en", webinar_topic="My Topic"
            )
            
            # Verify alignment check called
            mock_check.check_alignment.assert_called_once()

if __name__ == '__main__':
    unittest.main()
