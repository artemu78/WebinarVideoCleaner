#!/usr/bin/env python3
import json
import re
import os
import sys

def parse_time_to_ms(time_str):
    """
    Parses timestamp to milliseconds.
    Supports 'HH:MM:SS,mmm' (SRT) and 'HH:MM:SS' (Simple).
    """
    time_str = time_str.strip()
    # SRT format
    if ',' in time_str:
        hms, ms = time_str.split(',')
    elif '.' in time_str:
         hms, ms = time_str.split('.')
    else:
        hms = time_str
        ms = '000'

    parts = hms.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h = 0
        m, s = map(int, parts)
    else:
        return 0
    
    return (h * 3600 + m * 60 + s) * 1000 + int(ms)

def format_ms_to_srt(ms):
    """
    Formats milliseconds to 'HH:MM:SS,mmm'.
    """
    seconds = ms // 1000
    milliseconds = ms % 1000
    minutes = seconds // 60
    hours = minutes // 60
    
    seconds %= 60
    minutes %= 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def load_cuts(json_path):
    """
    Loads cuts from JSON file.
    Expected format: list of dicts with 'start' and 'end' (HH:MM:SS).
    Returns list of tuples (start_ms, end_ms).
    """
    if not os.path.exists(json_path):
        print(f"Error: Cuts file not found: {json_path}")
        return []
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        cuts = []
        for item in data:
            if 'start' in item and 'end' in item:
                start_ms = parse_time_to_ms(item['start'])
                end_ms = parse_time_to_ms(item['end'])
                cuts.append((start_ms, end_ms))
        
        # Sort cuts by start time
        cuts.sort(key=lambda x: x[0])
        return cuts
    except Exception as e:
        print(f"Error loading cuts: {e}")
        return []

def parse_srt(srt_path):
    """
    Parses SRT file.
    Returns list of dicts: {'index': int, 'start': ms, 'end': ms, 'text': str}
    """
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return []

    subs = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            try:
                index = int(lines[0].strip())
                time_line = lines[1].strip()
                if '-->' in time_line:
                    start_str, end_str = time_line.split('-->')
                    start_ms = parse_time_to_ms(start_str)
                    end_ms = parse_time_to_ms(end_str)
                    
                    text = '\n'.join(lines[2:])
                    
                    subs.append({
                        'index': index,
                        'start': start_ms,
                        'end': end_ms,
                        'text': text
                    })
            except ValueError:
                continue
                
    return subs

def map_time(t, cuts):
    """
    Maps original time t (ms) to new time (ms) after cuts.
    Returns (new_time, is_deleted)
    """
    offset = 0
    for start, end in cuts:
        if t < start:
            # Before this cut, so just apply accumulated offset
            return t - offset, False
        elif t < end:
            # Inside this cut
            # Map to the point where cut happened (start - offset)
            return start - offset, True
        else:
            # After this cut, add to offset
            offset += (end - start)
            
    # After all cuts
    return t - offset, False

def apply_cuts_to_subs(subs, cuts):
    """
    Generates new subtitle list based on cuts.
    """
    new_subs = []
    new_index = 1
    
    for sub in subs:
        start_ms = sub['start']
        end_ms = sub['end']
        
        new_start, start_deleted = map_time(start_ms, cuts)
        new_end, end_deleted = map_time(end_ms, cuts)
        
        # If the subtitle duration becomes too short or negative, drop it
        # (Meaning it was fully or mostly inside a deleted region)
        if new_end - new_start < 100:  # Minimum 100ms duration
            continue
            
        # If it was significantly cut but enough remains, keep it
        new_subs.append({
            'index': new_index,
            'start': new_start,
            'end': new_end,
            'text': sub['text']
        })
        new_index += 1
        
    return new_subs

def save_srt(subs, output_path):
    """
    Saves subtitles to SRT file.
    """
    lines = []
    for sub in subs:
        lines.append(str(sub['index']))
        time_str = f"{format_ms_to_srt(sub['start'])} --> {format_ms_to_srt(sub['end'])}"
        lines.append(time_str)
        lines.append(sub['text'])
        lines.append("")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def main(srt_path, cuts_json_path):
    print(f"Correcting SRT: {srt_path}")
    print(f"Using cuts from: {cuts_json_path}")
    
    cuts = load_cuts(cuts_json_path)
    if not cuts:
        print("No cuts loaded. Creating copy of original SRT.")
        subs = parse_srt(srt_path)
        if subs:
            output_path = os.path.splitext(srt_path)[0] + "_corrected.srt"
            save_srt(subs, output_path)
            return output_path
        return None

    subs = parse_srt(srt_path)
    if not subs:
        print("No subtitles parsed.")
        return None
        
    print(f"Original subtitle count: {len(subs)}")
    
    new_subs = apply_cuts_to_subs(subs, cuts)
    print(f"New subtitle count: {len(new_subs)}")
    
    output_path = os.path.splitext(srt_path)[0] + "_corrected.srt"
    save_srt(new_subs, output_path)
    print(f"Saved corrected SRT to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./correct_srt.py <srt_file> <cuts_json>")
        sys.exit(1)
        
    srt_file = sys.argv[1]
    cuts_file = sys.argv[2]
    
    main(srt_file, cuts_file)
