#!/usr/bin/env python3
"""
Main script to orchestrate MP4 video editing workflow:
1. Transcribe MP4 to SRT
2. Analyze SRT with Gemini to find ranges to delete
3. Cut those ranges from the original MP4
"""

import os
import json
import re
import sys
import time
from datetime import datetime

# Import the three modules
try:
    import transcribe_to_srt
    import audio_cleaner
    import cut_mp4
    import apply_cuts_to_srt
    import generate_chapters
    import correct_srt_errors
    import common_utils
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure transcribe_to_srt.py, audio_cleaner.py, cut_mp4.py, apply_cuts_to_srt.py, generate_chapters.py, and correct_srt_errors.py are in the same directory.")
    sys.exit(1)


def extract_json_from_text(text):
    """
    Extract JSON object from text that might contain extra content.
    Looks for JSON objects with 'ranges_to_delete' key.
    Handles markdown code blocks and other formatting.
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON in the text
    # First, try to find a JSON object with ranges_to_delete using a more robust pattern
    # Match from { to } with proper nesting
    brace_count = 0
    start_idx = -1
    found_objects = []
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and 'ranges_to_delete' in parsed:
                        return parsed
                    found_objects.append(parsed)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # If we found JSON objects but none had ranges_to_delete, try the first one
    # that looks like it might be our format
    for obj in found_objects:
        if isinstance(obj, dict):
            # Check if it has a list that might be ranges
            for key, value in obj.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict) and ('start' in value[0] or 'end' in value[0]):
                        # This looks like our format, reconstruct it
                        return {'ranges_to_delete': value}
    
    # Last resort: try to parse the entire text as JSON
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and 'ranges_to_delete' in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    return None


def convert_timestamp_format(timestamp_str):
    """
    Convert SRT timestamp format (HH:MM:SS,mmm) to simple format (HH:MM:SS).
    Also handles HH:MM:SS format if already in that format.
    Handles edge cases like missing leading zeros.
    """
    if not timestamp_str or not isinstance(timestamp_str, str):
        raise ValueError(f"Invalid timestamp: must be a non-empty string, got {type(timestamp_str)}")
    
    # Remove milliseconds if present
    if ',' in timestamp_str:
        timestamp_str = timestamp_str.split(',')[0]
    
    # Strip whitespace
    timestamp_str = timestamp_str.strip()
    
    # Ensure it's in HH:MM:SS format
    parts = timestamp_str.split(':')
    
    if len(parts) == 2:
        # MM:SS -> 00:MM:SS
        # Handle missing leading zeros
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return f"00:{minutes:02d}:{seconds:02d}"
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    elif len(parts) == 3:
        # Already HH:MM:SS, but ensure proper formatting with leading zeros
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str} (expected HH:MM:SS or MM:SS)")


def convert_gemini_response_to_cut_format(gemini_response_path):
    """
    Convert Gemini response text file to JSON format expected by cut_mp4.py.
    
    Args:
        gemini_response_path: Path to the text file with Gemini response
        
    Returns:
        str: Path to the JSON file with ranges in cut_mp4.py format, or None on error
    """
    if not os.path.exists(gemini_response_path):
        print(f"Error: Gemini response file not found: {gemini_response_path}")
        return None
    
    # Read the response text
    with open(gemini_response_path, 'r', encoding='utf-8') as f:
        response_text = f.read()
    
    # Extract JSON from the response
    json_data = extract_json_from_text(response_text)
    
    if not json_data or 'ranges_to_delete' not in json_data:
        print("Error: Could not extract 'ranges_to_delete' from Gemini response")
        print("Response text preview:")
        print(response_text[:500])
        return None
    
    # Validate that ranges_to_delete is a list
    ranges_to_delete = json_data['ranges_to_delete']
    if not isinstance(ranges_to_delete, list):
        print(f"Error: 'ranges_to_delete' must be a list, got {type(ranges_to_delete)}")
        print(f"Value: {ranges_to_delete}")
        return None
    
    # Check if list is empty before processing
    if len(ranges_to_delete) == 0:
        print("Warning: 'ranges_to_delete' is an empty list. No ranges to process.")
        return None
    
    # Convert to format expected by cut_mp4.py
    # cut_mp4.py expects a list of dicts with 'start' and 'end' keys
    ranges = []
    for range_item in ranges_to_delete:
        try:
            start = convert_timestamp_format(range_item['start'])
            end = convert_timestamp_format(range_item['end'])
            ranges.append({
                'start': start,
                'end': end
            })
        except (KeyError, ValueError) as e:
            print(f"Warning: Skipping invalid range item: {range_item}, error: {e}")
            continue
    
    if not ranges:
        print("Error: No valid ranges found in Gemini response")
        return None
    
    # Save to JSON file (use absolute path)
    json_output_path = os.path.splitext(os.path.abspath(gemini_response_path))[0] + "_ranges.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(ranges, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(ranges)} ranges to JSON format: {json_output_path}")
    return json_output_path


def main():
    """
    Main workflow:
    1. Ask user for MP4 file
    2. Transcribe to SRT
    3. Correct Transcription Errors
    4. Analyze SRT with Gemini
    5. Convert Gemini response to cut format
    6. Cut video
    7. Correct SRT timestamps
    8. Generate Chapters
    """
    start_time = time.time()
    script_start_dt = datetime.now()
    
    print("=" * 60)
    print("MP4 Video Editor - Automated Workflow")
    print(f"Started at: {script_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Step 1: Get MP4 file from user
    mp4_path = input("Enter the path to the MP4 file: ").strip()
    
    # Remove quotes if user pasted a path with quotes
    if mp4_path.startswith('"') and mp4_path.endswith('"'):
        mp4_path = mp4_path[1:-1]
    elif mp4_path.startswith("'") and mp4_path.endswith("'"):
        mp4_path = mp4_path[1:-1]
    
    if not mp4_path:
        print("Error: No file path provided.")
        return
    
    if not os.path.exists(mp4_path):
        print(f"Error: MP4 file not found: {mp4_path}")
        return
    
    if not mp4_path.lower().endswith('.mp4'):
        print(f"Warning: File does not have .mp4 extension: {mp4_path}")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    print(f"\nProcessing MP4 file: {mp4_path}\n")

    # Ask for mode
    print("Select Mode:")
    print("1. Full Video Cleaner (Transcribe + Cut + Chapters)")
    print("2. Transcription & Chapters Only (No Cut)")
    mode_input = input("Enter choice (1/2, default 1): ").strip()
    no_cut_mode = (mode_input == '2')

    if no_cut_mode:
        print("\nMode: Transcription & Chapters Only (No Cut)")
    else:
        print("\nMode: Full Video Cleaner")

    # Ask for Webinar Topic (Optional)
    webinar_topic = input("Enter webinar topic (optional, press Enter to skip): ").strip()
    if not webinar_topic:
        webinar_topic = None
    else:
        print(f"Topic set to: {webinar_topic}")

    # Initialize variables that might be skipped
    gemini_response_path = "Skipped"
    json_ranges_path = "Skipped"
    output_video_path = "Skipped"
    corrected_srt_path = "Skipped"
    
    # Step 2: Transcribe MP4 to SRT
    step_start_time = time.time()
    print("=" * 60)
    print("STEP 1: Transcribing MP4 to SRT")
    print("=" * 60)
    try:
        srt_path, detected_language = transcribe_to_srt.main(
            file_input=mp4_path,
            model="small",
            max_segment_duration=8.0,
            use_srt=True,
            language=None  # Auto-detect
        )
        
        if not srt_path:
            print("Error: Transcription failed or SRT file was not created")
            return
        
        # Convert to absolute path to handle relative paths correctly
        srt_path = os.path.abspath(srt_path)
        
        if not os.path.exists(srt_path):
            print(f"Error: SRT file was not found at: {srt_path}")
            return
        
        print(f"\n✓ Transcription complete: {srt_path}\n")
        print(f"Step 1 duration: {time.time() - step_start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Correct Transcription Errors
    step_start_time = time.time()
    print("=" * 60)
    print(f"STEP 2: Correcting Transcription Errors (Language: {detected_language})")
    print("=" * 60)
    try:
        if not detected_language:
            detected_language = "en"
            print("Warning: Language was not detected, defaulting to 'en'")
            
        corrected_transcription_path = correct_srt_errors.process_srt_correction(srt_path, detected_language, webinar_topic)
        
        if corrected_transcription_path and os.path.exists(corrected_transcription_path):
            corrected_transcription_path = os.path.abspath(corrected_transcription_path)
            print(f"\n✓ Transcription correction complete: {corrected_transcription_path}\n")
            # Update srt_path to use the corrected version for subsequent steps
            srt_path = corrected_transcription_path
        else:
            print("Warning: Transcription correction failed or returned no file. Using original SRT.")
            
    except Exception as e:
        print(f"Error during transcription correction: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with original SRT...")
    
    print(f"Step 2 duration: {time.time() - step_start_time:.2f} seconds")
    
    # Step 3, 4, 5, 6: Cut video workflow (Skipped if No Cut mode)
    if not no_cut_mode:
        # Step 3: Analyze SRT with Gemini
        step_start_time = time.time()
        print("=" * 60)
        print("STEP 3: Analyzing SRT with Gemini")
        print("=" * 60)
        try:
            gemini_response_path = audio_cleaner.process_srt_file(srt_path)
            
            if not gemini_response_path:
                print("Error: Gemini analysis failed or response file was not created")
                return
            
            # Convert to absolute path to handle relative paths correctly
            gemini_response_path = os.path.abspath(gemini_response_path)
            
            if not os.path.exists(gemini_response_path):
                print(f"Error: Gemini response file was not found at: {gemini_response_path}")
                return
            
            print(f"\n✓ Gemini analysis complete: {gemini_response_path}\n")
            print(f"Step 3 duration: {time.time() - step_start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during Gemini analysis: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 4: Convert Gemini response to cut format
        step_start_time = time.time()
        print("=" * 60)
        print("STEP 4: Converting Gemini response to cut format")
        print("=" * 60)
        try:
            json_ranges_path = convert_gemini_response_to_cut_format(gemini_response_path)
            
            if not json_ranges_path:
                print("Error: Failed to convert Gemini response to cut format")
                return
            
            # Convert to absolute path to handle relative paths correctly
            json_ranges_path = os.path.abspath(json_ranges_path)
            
            if not os.path.exists(json_ranges_path):
                print(f"Error: Ranges JSON file was not found at: {json_ranges_path}")
                return
            
            print(f"\n✓ Conversion complete: {json_ranges_path}\n")
            print(f"Step 4 duration: {time.time() - step_start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 5: Cut video
        step_start_time = time.time()
        print("=" * 60)
        print("STEP 5: Cutting video")
        print("=" * 60)
        try:
            output_video_path = cut_mp4.process_video(mp4_path, json_ranges_path)
            
            if not output_video_path:
                print("Error: Video cutting failed or output file was not created")
                return
            
            # Convert to absolute path to handle relative paths correctly
            output_video_path = os.path.abspath(output_video_path)
            
            if not os.path.exists(output_video_path):
                print(f"Error: Output video file was not found at: {output_video_path}")
                return
            
            print(f"\n✓ Video cutting complete: {output_video_path}\n")
            print(f"Step 5 duration: {time.time() - step_start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during video cutting: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 6: Correct SRT timestamps
        step_start_time = time.time()
        print("=" * 60)
        print("STEP 6: Correcting SRT timestamps")
        print("=" * 60)
        try:
            corrected_srt_path = apply_cuts_to_srt.main(srt_path, json_ranges_path)
            
            if not corrected_srt_path:
                print("Warning: SRT correction failed or file was not created")
                corrected_srt_path = "Failed"
            else:
                # Convert to absolute path
                corrected_srt_path = os.path.abspath(corrected_srt_path)
                
                if not os.path.exists(corrected_srt_path):
                    print(f"Error: Corrected SRT file was not found at: {corrected_srt_path}")
                    corrected_srt_path = "Not found"
                else:
                    print(f"\n✓ SRT correction complete: {corrected_srt_path}\n")
        except Exception as e:
            print(f"Error during SRT correction: {e}")
            import traceback
            traceback.print_exc()
            corrected_srt_path = f"Error: {e}"
        print(f"Step 6 duration: {time.time() - step_start_time:.2f} seconds")
    else:
        print("\nSkipping Steps 3, 4, 5, 6 (Analysis & Cutting) due to No Cut mode selection.")
    
    # Step 7: Generate Chapters
    step_start_time = time.time()
    print("=" * 60)
    print("STEP 7: Generating Chapters")
    print("=" * 60)
    chapters_path = "Skipped"
    
    # Only run if we have a valid corrected SRT (or at least the original one if correction failed, but usually we want corrected)
    # If correction failed, we might want to skip or use original. Let's use corrected_srt_path if valid, else srt_path
    
    valid_srt_for_chapters = None
    if corrected_srt_path and os.path.exists(corrected_srt_path):
        valid_srt_for_chapters = corrected_srt_path
    elif srt_path and os.path.exists(srt_path):
        print("Note: Using original SRT for chapters as corrected SRT is unavailable.")
        valid_srt_for_chapters = srt_path
    
    if valid_srt_for_chapters:
        try:
            chapters_path = generate_chapters.generate_chapters(valid_srt_for_chapters, language=detected_language)
            
            if not chapters_path:
                print("Warning: Chapter generation failed")
                chapters_path = "Failed"
            else:
                chapters_path = os.path.abspath(chapters_path)
                print(f"\n✓ Chapters generated: {chapters_path}\n")
        except Exception as e:
            print(f"Error during chapter generation: {e}")
            import traceback
            traceback.print_exc()
            chapters_path = f"Error: {e}"
    else:
        print("Error: No valid SRT file available for chapter generation.")
        chapters_path = "No input SRT"
    
    print(f"Step 7 duration: {time.time() - step_start_time:.2f} seconds")

    # Summary
    elapsed_time = time.time() - start_time
    time_str = common_utils.format_ms_to_srt(elapsed_time * 1000)
    
    script_end_dt = datetime.now()
    
    print("=" * 60)
    print("WORKFLOW COMPLETE!")
    print("=" * 60)
    print(f"Script started:  {script_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script finished: {script_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Original video: {mp4_path}")
    print(f"SRT file:       {srt_path}")
    print(f"Gemini response: {gemini_response_path}")
    print(f"Ranges JSON:    {json_ranges_path}")
    print(f"Cleaned video:  {output_video_path}")
    print(f"Corrected SRT:  {corrected_srt_path}")
    print(f"Chapters file:  {chapters_path}")
    print(f"Total execution time: {time_str}")
    
    total_cost = common_utils.get_total_gemini_cost()
    print(f"Total Gemini Cost:    ${total_cost:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

