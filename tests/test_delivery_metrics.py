import unittest
import sys
import os
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies to avoid import errors for delivery_metrics
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()
sys.modules["google.genai.errors"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

from delivery_metrics import parse_srt_for_metrics, calculate_manual_metrics, read_chapters

class TestDeliveryMetrics(unittest.TestCase):
    
    def test_parse_srt_for_metrics(self):
        srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:05,000\nWorld"
        with patch("builtins.open", mock_open(read_data=srt_content)):
            blocks = parse_srt_for_metrics("dummy.srt")
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0]['start'], 1000)
            self.assertEqual(blocks[0]['end'], 2000)
            self.assertEqual(blocks[1]['start'], 3000)
            self.assertEqual(blocks[1]['end'], 5000)

    def test_calculate_manual_metrics(self):
        # Setup: 2 blocks with a 4s gap between them
        blocks = [
            {'index': '1', 'start': 0, 'end': 1000, 'text': 'word1 word2'}, # 1s duration, 2 words
            {'index': '2', 'start': 5000, 'end': 6000, 'text': 'word3 word4'} # 1s duration, 2 words
        ]
        # Gap: 1000 to 5000 = 4000ms (Dead Air threshold is 3000ms)
        # Total duration: 6000ms
        # Total dead air: 4000ms
        # Speaking duration: 2000ms
        # Total words: 4
        # WPM: 4 / (2000/60000) = 4 / (1/30) = 120.0
        # Dead Air %: (4000 / 6000) * 100 = 66.666...%
        
        metrics = calculate_manual_metrics(blocks)
        self.assertEqual(metrics['dead_air_count'], 1)
        self.assertEqual(metrics['total_dead_air_ms'], 4000)
        self.assertEqual(metrics['average_wpm'], '120.0')
        self.assertEqual(metrics['dead_air_percentage'], '66.7%')
        self.assertEqual(len(metrics['dead_air_intervals']), 1)
        self.assertEqual(metrics['dead_air_intervals'][0]['duration'], 4000)

    def test_calculate_manual_metrics_empty(self):
        expected = {
            'total_duration_ms': 0,
            'dead_air_count': 0,
            'total_dead_air_ms': 0,
            'dead_air_percentage': "0.0%",
            'average_wpm': "0.0",
            'dead_air_intervals': []
        }
        self.assertEqual(calculate_manual_metrics([]), expected)

    def test_read_chapters(self):
        chapters_content = "00:00:00 - Intro\n00:05:00 - Deep Dive"
        with patch("builtins.open", mock_open(read_data=chapters_content)):
            with patch("os.path.exists", return_value=True):
                content = read_chapters("chapters.txt")
                self.assertEqual(content, chapters_content)
                
        with patch("os.path.exists", return_value=False):
            content = read_chapters("non_existent.txt")
            self.assertEqual(content, "No chapters provided.")

if __name__ == '__main__':
    unittest.main()
