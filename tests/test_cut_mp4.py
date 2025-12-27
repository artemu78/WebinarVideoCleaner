import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock moviepy system-wide before importing the module
# This ensures that when cut_mp4 imports from moviepy, it gets our mocks
mock_moviepy = MagicMock()
sys.modules["moviepy"] = mock_moviepy

from cut_mp4 import time_to_seconds, format_seconds, process_video

class TestCutMp4Utils(unittest.TestCase):
    
    def test_time_to_seconds(self):
        self.assertEqual(time_to_seconds("00:00:10"), 10.0)
        self.assertEqual(time_to_seconds("01:00:00"), 3600.0)
        self.assertEqual(time_to_seconds("10"), 10.0)
        self.assertEqual(time_to_seconds(10), 10.0)
        self.assertEqual(time_to_seconds("01:30"), 90.0)
        self.assertEqual(time_to_seconds(None), 0)
        
    def test_format_seconds(self):
        # timedelta string representation depends on total duration
        self.assertEqual(format_seconds(10), "0:00:10")
        self.assertEqual(format_seconds(3600), "1:00:00")
        
    @patch("cut_mp4.VideoFileClip")
    @patch("cut_mp4.concatenate_videoclips")
    @patch("os.path.exists")
    def test_process_video_remove(self, mock_exists, mock_concat, mock_videoclip):
        # Setup mocks
        mock_exists.return_value = True
        
        mock_video_instance = MagicMock()
        mock_video_instance.duration = 100
        mock_videoclip.return_value = mock_video_instance
        
        mock_subclip = MagicMock()
        mock_video_instance.subclipped.return_value = mock_subclip
        
        mock_final_video = MagicMock()
        mock_final_video.duration = 80
        mock_concat.return_value = mock_final_video
        
        # User asks to remove 10-20
        # Expected behavior: Keep 0-10, Keep 20-100
        start = 10
        end = 20
        
        output = process_video("test.mp4", start=start, end=end, mode='remove')
        
        # Verify VideoFileClip called with path
        mock_videoclip.assert_called_with("test.mp4")
        
        # Verify logic: we expect two subclipped calls: (0, 10.0) and (20.0, 100)
        calls = mock_video_instance.subclipped.call_args_list
        self.assertEqual(len(calls), 2)
        
        # args[0] is start, args[1] is end
        self.assertEqual(calls[0][0][0], 0)
        self.assertEqual(calls[0][0][1], 10.0)
        
        self.assertEqual(calls[1][0][0], 20.0)
        self.assertEqual(calls[1][0][1], 100)
        
        # Verify write
        self.assertTrue(mock_final_video.write_videofile.called)

    @patch("cut_mp4.VideoFileClip")
    @patch("cut_mp4.concatenate_videoclips")
    @patch("os.path.exists")
    def test_process_video_keep(self, mock_exists, mock_concat, mock_videoclip):
        # Setup mocks
        mock_exists.return_value = True
        
        mock_video_instance = MagicMock()
        mock_video_instance.duration = 100
        mock_videoclip.return_value = mock_video_instance
        
        mock_subclip = MagicMock()
        mock_video_instance.subclipped.return_value = mock_subclip
        
        mock_final_video = MagicMock()
        mock_concat.return_value = mock_final_video
        
        # Keep: 10-20
        start = 10
        end = 20
        
        process_video("test.mp4", start=start, end=end, mode='keep')
        
        # Should call subclipped only once for 10-20
        mock_video_instance.subclipped.assert_called_with(10.0, 20.0)

if __name__ == '__main__':
    unittest.main()
