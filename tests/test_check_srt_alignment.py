import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_srt_alignment import check_alignment

class TestCheckSrtAlignment(unittest.TestCase):

    @patch('check_srt_alignment.os.path.exists')
    def test_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        result = check_alignment("non_existent.srt")
        self.assertFalse(result)

    @patch('check_srt_alignment.os.path.exists')
    @patch('check_srt_alignment.parse_srt')
    def test_no_subtitles(self, mock_parse, mock_exists):
        mock_exists.return_value = True
        mock_parse.return_value = []
        result = check_alignment("empty.srt")
        self.assertFalse(result)

    @patch('check_srt_alignment.os.path.exists')
    @patch('check_srt_alignment.parse_srt')
    def test_valid_alignment(self, mock_parse, mock_exists):
        mock_exists.return_value = True
        mock_parse.return_value = [
            {'index': 1, 'start': 1000, 'end': 2000, 'text': 'Subtitle 1'},
            {'index': 2, 'start': 2000, 'end': 3000, 'text': 'Subtitle 2'},
            {'index': 3, 'start': 3500, 'end': 4000, 'text': 'Subtitle 3'}
        ]
        result = check_alignment("valid.srt")
        self.assertTrue(result)

    @patch('check_srt_alignment.os.path.exists')
    @patch('check_srt_alignment.parse_srt')
    def test_invalid_start_ge_end(self, mock_parse, mock_exists):
        mock_exists.return_value = True
        # Subtitle 2 has start == end, Subtitle 3 has start > end
        mock_parse.return_value = [
            {'index': 1, 'start': 1000, 'end': 2000, 'text': 'Valid'},
            {'index': 2, 'start': 3000, 'end': 3000, 'text': 'Zero Duration'},
            {'index': 3, 'start': 4000, 'end': 3500, 'text': 'Negative Duration'}
        ]
        result = check_alignment("invalid_durations.srt")
        self.assertFalse(result)

    @patch('check_srt_alignment.os.path.exists')
    @patch('check_srt_alignment.parse_srt')
    def test_overlap(self, mock_parse, mock_exists):
        mock_exists.return_value = True
        # Subtitle 2 starts before Subtitle 1 ends
        mock_parse.return_value = [
            {'index': 1, 'start': 1000, 'end': 3000, 'text': 'Subtitle 1'},
            {'index': 2, 'start': 2500, 'end': 4000, 'text': 'Overlapping Subtitle'}
        ]
        result = check_alignment("overlap.srt")
        self.assertFalse(result)

    @patch('check_srt_alignment.os.path.exists')
    @patch('check_srt_alignment.parse_srt')
    def test_unsorted(self, mock_parse, mock_exists):
        mock_exists.return_value = True
        # Subtitle 2 starts before Subtitle 1 starts (not just overlap, but out of order)
        mock_parse.return_value = [
            {'index': 1, 'start': 3000, 'end': 4000, 'text': 'Subtitle 1'},
            {'index': 2, 'start': 1000, 'end': 2000, 'text': 'Earlier Subtitle'}
        ]
        result = check_alignment("unsorted.srt")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
