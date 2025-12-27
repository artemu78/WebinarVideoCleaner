import unittest
import sys
import os
from unittest.mock import patch, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apply_cuts_to_srt import map_time, apply_cuts_to_subs, parse_srt, save_srt, load_cuts

class TestCorrectSrt(unittest.TestCase):
    
    def test_load_cuts(self):
        json_content = '[{"start": "00:00:01", "end": "00:00:02"}, {"start": "00:00:05", "end": "00:00:06"}]'
        with patch("builtins.open", mock_open(read_data=json_content)):
             with patch("os.path.exists", return_value=True):
                cuts = load_cuts("dummy.json")
                self.assertEqual(len(cuts), 2)
                # 00:00:01 = 1000ms
                self.assertEqual(cuts[0], (1000, 2000))
                self.assertEqual(cuts[1], (5000, 6000))

    
    def test_map_time(self):
        # Cuts: remove 1000-2000
        cuts = [(1000, 2000)]
        
        # Before cut
        t, deleted = map_time(500, cuts)
        self.assertEqual(t, 500)
        self.assertFalse(deleted)
        
        # Inside cut
        t, deleted = map_time(1500, cuts)
        self.assertEqual(t, 1000) # Clamped to start of cut (start - offset) -> 1000 - 0 = 1000
        self.assertTrue(deleted)
        
        # After cut (should shift by 1000ms)
        t, deleted = map_time(3000, cuts)
        self.assertEqual(t, 2000) # 3000 - 1000
        self.assertFalse(deleted)

    def test_multiple_cuts(self):
        # Cuts: 1000-2000, 4000-5000
        cuts = [(1000, 2000), (4000, 5000)]
        
        # Before all
        self.assertEqual(map_time(500, cuts)[0], 500)
        
        # Between first and second
        # 3000 -> minus 1000 offset -> 2000
        self.assertEqual(map_time(3000, cuts)[0], 2000)
        
        # After all
        # 6000 -> minus 2000 offset -> 4000
        self.assertEqual(map_time(6000, cuts)[0], 4000)
        
    def test_apply_cuts_to_subs(self):
        subs = [
            {'index': 1, 'start': 0, 'end': 1000, 'text': 'keep1'},
            {'index': 2, 'start': 1200, 'end': 1800, 'text': 'delete'},
            {'index': 3, 'start': 2500, 'end': 3500, 'text': 'keep2'}
        ]
        cuts = [(1000, 2000)]
        
        new_subs = apply_cuts_to_subs(subs, cuts)
        
        # Expect sub 1 to be kept normal
        self.assertEqual(new_subs[0]['text'], 'keep1')
        self.assertEqual(new_subs[0]['end'], 1000)
        
        # Expect sub 2 to be dropped (fully inside or mostly inside)
        # mapped start: 1200 -> 1000 (del), mapped end: 1800 -> 1000 (del). 
        # duration 0. Dropped.
        
        # Expect sub 3 to be shifted
        # 2500 -> 1500, 3500 -> 2500
        self.assertEqual(len(new_subs), 2)
        self.assertEqual(new_subs[1]['text'], 'keep2')
        self.assertEqual(new_subs[1]['start'], 1500)
        self.assertEqual(new_subs[1]['end'], 2500)
        
        # Indices should be renumbered
        self.assertEqual(new_subs[0]['index'], 1)
        self.assertEqual(new_subs[1]['index'], 2)

    def test_parse_srt(self):
        srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,000\nWorld"
        with patch("builtins.open", mock_open(read_data=srt_content)):
            with patch("os.path.exists", return_value=True):
                subs = parse_srt("dummy.srt")
                self.assertEqual(len(subs), 2)
                self.assertEqual(subs[0]['text'], "Hello")
                self.assertEqual(subs[0]['start'], 1000)
                self.assertEqual(subs[1]['text'], "World")
                
    def test_save_srt(self):
        subs = [
             {'index': 1, 'start': 1000, 'end': 2000, 'text': 'Hello'}
        ]
        with patch("builtins.open", mock_open()) as m:
            save_srt(subs, "output.srt")
            m.assert_called_with("output.srt", 'w', encoding='utf-8')
            handle = m()
            # Check write calls contain the content
            # Aggregate writes
            args_list = handle.write.call_args_list
            full_content = "".join([call[0][0] for call in args_list])
            self.assertIn("00:00:01,000 --> 00:00:02,000", full_content)
            self.assertIn("Hello", full_content)
            
if __name__ == '__main__':
    unittest.main()
