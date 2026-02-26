
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock whisper before import
sys.modules["whisper"] = MagicMock()

import transcribe_to_srt

class TestTranscribeToSrtInteractive(unittest.TestCase):
    
    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.whisper.load_model')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('builtins.input')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    @patch('sys.argv', ["transcribe_to_srt.py"]) # Patch argv to avoid argparse error
    def test_existing_srt_use_existing_no_language(self, mock_walk, mock_isdir, mock_exists, mock_input, mock_extract, mock_load, mock_detect, mock_has_audio):
        # Scenario: File exists, User says 'n' (don't regenerate), No language provided in args.
        # User should be prompted for language.
        
        # Mocks
        mock_isdir.return_value = False # Not a folder
        mock_has_audio.return_value = True
        
        # We need careful handling of exists() since it's used for input checks AND output check
        # 1. check file_input exists -> True
        # 2. check output path exists -> True (Trigger prompt)
        mock_exists.return_value = True 
        
        # Inputs:
        # 1. folder_input (Enter to skip)
        # 2. file_input ("video.mp4")
        # 3. srt_input ("y")
        # 4. regenerate path ("n") -> Use existing
        # 5. language prompt ("ru")
        mock_input.side_effect = ["", "video.mp4", "y", "n", "ru"]
        
        # Run
        outpath, language = transcribe_to_srt.main()
        
        # Verify
        self.assertEqual(language, "ru")
        # Output path depends on logic: video.mp4 -> video.srt
        self.assertTrue(outpath.endswith(".srt"))
        
        # Verify model NOT loaded
        mock_load.assert_not_called()
        
    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.whisper.load_model')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('builtins.input')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    def test_existing_srt_use_existing_with_language(self, mock_walk, mock_isdir, mock_exists, mock_input, mock_extract, mock_load, mock_detect, mock_has_audio):
        # Scenario: File exists, User says 'n', Language provided in args.
        # No extra prompt for language.
        
        mock_isdir.return_value = False
        mock_exists.return_value = True
        mock_has_audio.return_value = True
        
        # Inputs:
        # "Regenerate?" prompt uses input().
        mock_input.side_effect = ["n"]
        
        outpath, language = transcribe_to_srt.main(file_input="video.mp4", language="es")
        
        self.assertEqual(language, "es")
        mock_load.assert_not_called()
        
    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.whisper.load_model')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('transcribe_to_srt.segments_to_srt')
    @patch('builtins.open')
    @patch('builtins.input')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_existing_srt_regenerate(self, mock_isdir, mock_exists, mock_input, mock_open, mock_segs, mock_extract, mock_load, mock_detect, mock_has_audio):
        # Scenario: File exists, User says 'y' (regenerate).
        # Should load model and proceed.
        
        mock_isdir.return_value = False
        mock_exists.return_value = True
        mock_has_audio.return_value = True
        
        # Inputs for regenerate prompt
        mock_input.side_effect = ["y"]
        
        # Mock detection/transcription returns
        mock_detect.return_value = ("en", 0.99)
        mock_ws_model = MagicMock()
        mock_load.return_value = mock_ws_model
        
        # Configure whisper version
        transcribe_to_srt.whisper.__version__ = "20231117"
        
        # Mock transcribe result
        mock_ws_model.transcribe.return_value = {"segments": [], "language": "en"}
        
        transcribe_to_srt.main(file_input="video.mp4")
        
        # Verify model LOADED
        mock_load.assert_called_once()
        # Verify transcribe called
        mock_ws_model.transcribe.assert_called()

if __name__ == '__main__':
    unittest.main()
