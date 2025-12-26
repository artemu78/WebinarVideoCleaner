#!/usr/bin/env python3
import json
import os
import sys
import argparse
from datetime import timedelta
# Modern import for MoviePy 3.0+
from moviepy import VideoFileClip, concatenate_videoclips

def time_to_seconds(t_str):
    """
    Converts HH:MM:SS, MM:SS, or seconds string to float seconds.
    """
    if t_str is None:
        return 0
    if isinstance(t_str, (int, float)):
        return float(t_str)
    
    t_str = str(t_str).strip()
    
    # Try direct float conversion
    try:
        return float(t_str)
    except ValueError:
        pass
        
    parts = list(map(float, t_str.split(':')))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0

def format_seconds(seconds):
    """Formats seconds to HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def process_video(video_path, json_path=None, start=None, end=None, mode='remove'):
    """
    Edits video by removing or keeping segments.
    
    Args:
        video_path: Path to input video
        json_path: Path to JSON file with 'start'/'end' dicts (usually for remove mode)
        start: Start time (HH:MM:SS or seconds) for manual single operation
        end: End time (HH:MM:SS or seconds) for manual single operation
        mode: 'remove' (default) - deletes specified segments
              'keep' - keeps specified segments (trimming)
        
    Returns:
        str: Path to output file or None
    """
    # Normalize inputs
    intervals = []
    
    if json_path:
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found: {json_path}")
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract list if it's wrapped in a dict
            if isinstance(data, dict):
                if 'ranges_to_delete' in data:
                    intervals = data['ranges_to_delete']
                # If we ever add 'ranges_to_keep', handle it here
                elif 'ranges_to_keep' in data:
                    intervals = data['ranges_to_keep']
                    mode = 'keep' # Auto-switch mode if JSON is explicit
                else:
                    # Fallback or error? Let's check if values look like list
                    pass
            
            if isinstance(data, list):
                intervals = data
                
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return None
    elif start is not None and end is not None:
        intervals = [{'start': start, 'end': end}]
    else:
        print("Error: Must provide either json_path or start/end times.")
        return None

    if not isinstance(intervals, list):
        print(f"Error: Invalid cuts data format. Expected list, got {type(intervals)}")
        return None

    if len(intervals) == 0:
        print("Warning: No intervals provided. Video will not be changed.")
        return None

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    # Prepare output path
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_basename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_basename)
    
    if mode == 'keep':
        action_tag = "trimmed"
    else:
        action_tag = "cleaned"
        
    output_path = os.path.join(video_dir, f"{action_tag}_{video_name}{video_ext}")

    print(f"Processing: {video_path}")
    print(f"Mode: {mode.upper()}")
    
    try:
        video = VideoFileClip(video_path)
        initial_length = video.duration
        
        # Parse intervals into (start, end) tuples in seconds
        parsed_intervals = []
        for item in intervals:
            try:
                s = time_to_seconds(item.get('start', 0))
                e = time_to_seconds(item.get('end', 0))
                
                # Validation
                if s >= e:
                    print(f"Skipping invalid interval (start >= end): {s}-{e}")
                    continue
                
                # Clamp to video duration
                if s >= initial_length: 
                    continue
                e = min(e, initial_length)
                
                parsed_intervals.append((s, e))
            except Exception as e:
                print(f"Error parsing interval {item}: {e}")
                
        parsed_intervals.sort()
        
        final_clips = []
        
        if mode == 'remove':
            # Calculate segments to KEEP (inverse of remove)
            last_end = 0
            for start, end in parsed_intervals:
                if start > last_end:
                    final_clips.append(video.subclipped(last_end, start))
                last_end = max(last_end, end)
            
            if last_end < initial_length:
                final_clips.append(video.subclipped(last_end, initial_length))
                
        elif mode == 'keep':
            # Just keep the requested segments
            for start, end in parsed_intervals:
                final_clips.append(video.subclipped(start, end))
        
        if not final_clips:
            print("Error: No video content remains after processing.")
            video.close()
            return None
            
        final_video = concatenate_videoclips(final_clips)
        final_length = final_video.duration
        
        print("\n" + "="*40)
        print("DURATION REPORT:")
        print(f"Original: {format_seconds(initial_length)}")
        print(f"Final:    {format_seconds(final_length)}")
        print("="*40 + "\n")
        
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        final_video.close()
        video.close()
        
        print(f"\nDone! Saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Critical Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Cut or Trim MP4 videos")
    parser.add_argument("video_path", nargs="?", help="Path to input video file")
    parser.add_argument("--json", help="Path to JSON file with cuts/ranges")
    parser.add_argument("--start", help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", help="End time (HH:MM:SS or seconds)")
    parser.add_argument("--mode", choices=['remove', 'keep'], default='remove', 
                        help="Action to perform on the specified ranges: 'remove' (default) deletes them, 'keep' extracts them.")

    args = parser.parse_args()

    # Interactive Fallback if no args provided (except script name)
    if not args.video_path:
        print("--- Manual Video Cutter ---")
        args.video_path = input("Enter video path: ").strip()
        
        choice = input("Do you have a JSON file? (y/n): ").strip().lower()
        if choice == 'y':
            args.json = input("Enter JSON path: ").strip()
        else:
            args.start = input("Start time (HH:MM:SS): ").strip()
            args.end = input("End time (HH:MM:SS): ").strip()
            mode_in = input("Mode (remove/keep) [remove]: ").strip().lower()
            if mode_in in ['keep', 'remove']:
                args.mode = mode_in

    if args.video_path:
        # Remove quotes
        if args.video_path.startswith('"') and args.video_path.endswith('"'):
            args.video_path = args.video_path[1:-1]
        elif args.video_path.startswith("'") and args.video_path.endswith("'"):
            args.video_path = args.video_path[1:-1]
    
    process_video(args.video_path, args.json, args.start, args.end, args.mode)

if __name__ == "__main__":
    main()