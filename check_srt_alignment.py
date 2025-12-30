#!/usr/bin/env python3
import sys
import os
from common_utils import format_ms_to_srt
from apply_cuts_to_srt import parse_srt

def check_alignment(srt_path):
    """
    Checks if SRT subtitles are correctly aligned:
    1. start < end
    2. No overlaps (next start >= prev end)
    3. Monotonic (next start >= prev start)
    """
    print(f"Checking alignment for: {srt_path}")
    
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return False
        
    subs = parse_srt(srt_path)
    if not subs:
        print("No subtitles found.")
        return False
    
    issues = []
    
    for i in range(len(subs)):
        current = subs[i]
        
        # Check 1: start < end
        if current['start'] >= current['end']:
            issues.append(f"Subtitle {current['index']}: Start time ({format_ms_to_srt(current['start'])}) >= End time ({format_ms_to_srt(current['end'])})")
            
        if i > 0:
            prev = subs[i-1]
            
            # Check 2: Overlap (next start < prev end)
            if current['start'] < prev['end']:
                overlap_ms = prev['end'] - current['start']
                issues.append(f"Overlap detected between {prev['index']} and {current['index']}: Prev End ({format_ms_to_srt(prev['end'])}) > Curr Start ({format_ms_to_srt(current['start'])}). Overlap: {overlap_ms}ms")
                
            # Check 3: Monotonic (next start < prev start) - stronger than overlap, implies unsorted
            if current['start'] < prev['start']:
                 issues.append(f"Unsorted subtitles detected at {current['index']}: Start ({format_ms_to_srt(current['start'])}) < Prev Start ({format_ms_to_srt(prev['start'])})")

    if issues:
        print(f"Found {len(issues)} alignment issues:")
        for issue in issues[:10]: # Print first 10
            print(f"- {issue}")
        if len(issues) > 10:
            print(f"... and {len(issues) - 10} more.")
        return False
    else:
        print("✓ Alignment check passed. All subtitles are sequential and valid.")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./check_srt_alignment.py <srt_file>")
        sys.exit(1)
        
    srt_file = sys.argv[1]
    success = check_alignment(srt_file)
    if not success:
        sys.exit(1)
