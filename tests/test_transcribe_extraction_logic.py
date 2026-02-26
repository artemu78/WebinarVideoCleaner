
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock whisper before import
sys.modules["whisper"] = MagicMock()

import transcribe_to_srt

class TestTranscribeExtractionLogic(unittest.TestCase):
    
    def test_get_extracted_mp3_path(self):
        # Should return path in same directory with _extracted.mp3 suffix
        path = transcribe_to_srt.get_extracted_mp3_path("/path/to/video.mp4")
        self.assertEqual(path, "/path/to/video_extracted.mp3")

    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.whisper.load_model')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('transcribe_to_srt.get_segments_from_file')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.path.getmtime')
    def test_skips_extraction_if_file_exists(self, mock_getmtime, mock_isdir, mock_exists, mock_open, mock_get_segments, mock_extract, mock_load, mock_detect, mock_has_audio):
        """
        Verify that if the _extracted.mp3 file already exists, extract_mp3_from_mp4 is NOT called.
        """
        # Setup mocks
        mock_isdir.return_value = False
        # Mock exists to return True for:
        # 1. The input video.mp4
        # 2. The extracted video_extracted.mp3
        # 3. BUT return False for the output .srt (so we don't get the "Regenerate?" prompt)
        def exists_side_effect(path):
            if "video.mp4" in path: return True
            if "video_extracted.mp3" in path: return True
            if path.endswith(".srt"): return False
            return False
            
        mock_exists.side_effect = exists_side_effect
        mock_has_audio.return_value = True
        mock_detect.return_value = ("en", 0.99)
        mock_get_segments.return_value = ([], "en")
        
        # Configure whisper version
        transcribe_to_srt.whisper.__version__ = "20231117"
        
        # Run main in non-interactive mode
        transcribe_to_srt.main(file_input="video.mp4", model="base")
        
        # Verify extract_mp3_from_mp4 was NEVER called
        mock_extract.assert_not_called()
        
        # Verify get_segments_from_file was called with the mp3 path
        # Note: In the code, it uses audio_path = extracted_mp3
        mock_get_segments.assert_called()
        call_args = mock_get_segments.call_args[0]
        # args[1] is audio_path
        self.assertTrue(call_args[1].endswith("video_extracted.mp3"))

    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.whisper.load_model')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('transcribe_to_srt.get_segments_from_file')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists')
    @patch('os.path.isdir')
    def test_performs_extraction_if_file_missing(self, mock_isdir, mock_exists, mock_open, mock_get_segments, mock_extract, mock_load, mock_detect, mock_has_audio):
        """
        Verify that if the _extracted.mp3 file DOES NOT exist, extract_mp3_from_mp4 IS called.
        """
        # Setup mocks
        mock_isdir.return_value = False
        
        def exists_side_effect(path):
            if "video.mp4" in path: return True
            if "video_extracted.mp3" in path: return False # Missing!
            if path.endswith(".srt"): return False
            return False
            
        mock_exists.side_effect = exists_side_effect
        mock_has_audio.return_value = True
        mock_detect.return_value = ("en", 0.99)
        mock_get_segments.return_value = ([], "en")
        mock_extract.return_value = "video_extracted.mp3"
        
        # Configure whisper version
        transcribe_to_srt.whisper.__version__ = "20231117"
        
        # Run main
        transcribe_to_srt.main(file_input="video.mp4", model="base")
        
        # Verify extract_mp3_from_mp4 WAS called
        mock_extract.assert_called()

if __name__ == '__main__':
    unittest.main()
