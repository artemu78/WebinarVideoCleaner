import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock whisper module
mock_whisper = MagicMock()
sys.modules["whisper"] = mock_whisper

from transcribe_to_srt import format_timestamp, segments_to_srt, detect_language

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

    def test_detect_language_success(self):
        # Setup mocks on the global mock_whisper
        mock_whisper.load_audio.return_value = "audio_data"
        mock_whisper.pad_or_trim.return_value = "padded_audio"
        mock_mel = MagicMock()
        mock_whisper.log_mel_spectrogram.return_value = mock_mel
        mock_mel.to.return_value = mock_mel
        
        mock_model = MagicMock()
        mock_model.device = "cpu"
        probs = {"fr": 0.9, "en": 0.1}
        mock_model.detect_language.return_value = (None, probs)
        
        lang, conf = detect_language(mock_model, "dummy.mp3")
        
        self.assertEqual(lang, "fr")
        self.assertAlmostEqual(conf, 0.9)
        
        # Verify calls
        mock_whisper.load_audio.assert_called_with("dummy.mp3")
        mock_model.detect_language.assert_called()

if __name__ == '__main__':
    unittest.main()
