import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock whisper module
mock_whisper = MagicMock()
sys.modules["whisper"] = mock_whisper

from transcribe_to_srt import format_timestamp, segments_to_srt, detect_language, natural_sort_key, segments_to_plain_text, get_language_codes_help

class TestTranscribeUtils(unittest.TestCase):
    
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(1.5), "00:00:01,500")
        self.assertEqual(format_timestamp(3661), "01:01:01,000")
        
    def test_segments_to_srt(self):
        segments = [
            {'start': 0, 'end': 1, 'text': 'Hello'},
            {'start': 1, 'end': 2, 'text': 'World'}
        ]
        # Check expected content
        result = segments_to_srt(segments)
        self.assertIn("1\n00:00:00,000 --> 00:00:01,000\nHello", result)
        self.assertIn("2\n00:00:01,000 --> 00:00:02,000\nWorld", result)

    @patch("transcribe_to_srt.whisper")
    def test_detect_language_success(self, mock_whisper_lib):
        # Setup mocks on the passed mock object
        mock_whisper_lib.load_audio.return_value = "audio_data"
        mock_whisper_lib.pad_or_trim.return_value = "padded_audio"
        mock_mel = MagicMock()
        mock_whisper_lib.log_mel_spectrogram.return_value = mock_mel
        mock_mel.to.return_value = mock_mel
        
        mock_model = MagicMock()
        mock_model.device = "cpu"
        probs = {"fr": 0.9, "en": 0.1}
        mock_model.detect_language.return_value = (None, probs)
        
        lang, conf = detect_language(mock_model, "dummy.mp3")
        
        self.assertEqual(lang, "fr")
        self.assertAlmostEqual(conf, 0.9)
        
        # Verify calls
        mock_whisper_lib.load_audio.assert_called_with("dummy.mp3")
        mock_model.detect_language.assert_called()

    def test_natural_sort_key(self):
        files = ["file1.txt", "file10.txt", "file2.txt"]
        sorted_files = sorted(files, key=natural_sort_key)
        self.assertEqual(sorted_files, ["file1.txt", "file2.txt", "file10.txt"])

    def test_segments_to_plain_text(self):
        segments = [
            {'start': 0, 'end': 1, 'text': 'Hello'},
            {'start': 1, 'end': 2, 'text': 'World'}
        ]
        result = segments_to_plain_text(segments)
        self.assertEqual(result, "Hello\nWorld")
        
    def test_get_language_codes_help(self):
        help_text = get_language_codes_help()
        self.assertIn("en: English", help_text)
        self.assertIn("ru: Russian", help_text)

if __name__ == '__main__':
    unittest.main()
