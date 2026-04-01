import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestTranscribeToSrtInteractive(unittest.TestCase):
    
    def setUp(self):
        # Use patch.dict to safely mock sys.modules for this test
        self.module_patcher = patch.dict(sys.modules, {"whisper": MagicMock()})
        self.module_patcher.start()
        
    def tearDown(self):
        self.module_patcher.stop()

    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('builtins.input')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    @patch('transcribe_to_srt.argparse.ArgumentParser')
    def test_existing_srt_use_existing_no_language(self, mock_parser_class, mock_walk, mock_isdir, mock_exists, mock_input, mock_extract, mock_detect, mock_has_audio):
        import transcribe_to_srt
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(folder_input=None, file_input=None, model="turbo", language=None, webinar_topic=None, skip_if_exists=False)
        
        mock_isdir.return_value = False
        mock_has_audio.return_value = True
        mock_exists.return_value = True 
        
        mock_input.side_effect = ["", "video.mp4", "y", "n", "ru"]
        
        outpath, language = transcribe_to_srt.main()
        self.assertEqual(language, "ru")

    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('builtins.input')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.walk')
    @patch('transcribe_to_srt.argparse.ArgumentParser')
    def test_existing_srt_use_existing_with_language(self, mock_parser_class, mock_walk, mock_isdir, mock_exists, mock_input, mock_extract, mock_detect, mock_has_audio):
        import transcribe_to_srt
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(folder_input=None, file_input=None, model="turbo", language="es", webinar_topic=None, skip_if_exists=False)
        
        mock_isdir.return_value = False
        mock_exists.return_value = True
        mock_has_audio.return_value = True
        
        mock_input.side_effect = ["", "video.mp4", "y", "n"]
        
        outpath, language = transcribe_to_srt.main(language="es")
        self.assertEqual(language, "es")

    @patch('transcribe_to_srt.has_audio_stream')
    @patch('transcribe_to_srt.detect_language')
    @patch('transcribe_to_srt.whisper.load_model')
    @patch('transcribe_to_srt.extract_mp3_from_mp4')
    @patch('transcribe_to_srt.segments_to_srt')
    @patch('builtins.input')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('transcribe_to_srt.argparse.ArgumentParser')
    def test_existing_srt_regenerate(self, mock_parser_class, mock_isdir, mock_exists, mock_input, mock_segs, mock_extract, mock_load, mock_detect, mock_has_audio):
        import transcribe_to_srt
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(folder_input=None, file_input=None, model="turbo", language=None, webinar_topic=None, skip_if_exists=False)
        
        mock_isdir.return_value = False
        mock_exists.return_value = True
        mock_has_audio.return_value = True
        
        mock_input.side_effect = ["", "video.mp4", "y", "y"]
        mock_detect.return_value = ("en", 0.99)
        mock_ws_model = MagicMock()
        mock_load.return_value = mock_ws_model
        transcribe_to_srt.whisper.__version__ = "20231117"
        mock_ws_model.transcribe.return_value = {"segments": [], "language": "en"}
        
        with patch("builtins.open", mock_open()):
            transcribe_to_srt.main()
        
        mock_load.assert_called_once()

if __name__ == '__main__':
    unittest.main()
