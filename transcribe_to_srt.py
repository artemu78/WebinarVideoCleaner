# transcribe_to_srt.py
# Requirements:
#   pip install -U openai-whisper==20240303  (or latest whisper)
#   ffmpeg must be installed and on PATH (for whisper)
# Usage:
#   python transcribe_to_srt.py --model small

import argparse
import math
import os
import time
import subprocess
from datetime import timedelta
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

try:
    import whisper
except Exception as e:
    raise SystemExit("Please install the whisper package: pip install -U openai-whisper\nAlso ensure ffmpeg is installed.") from e

def format_timestamp(seconds: float) -> str:
    # SRT timestamp: HH:MM:SS,mmm
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def segments_to_srt(segments):
    """
    segments: list of dicts with 'start', 'end', 'text'
    returns string with SRT content
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        text = seg['text'].strip()
        # Clean text newlines to single-line blocks (players handle multi-line too)
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line
    return "\n".join(lines)

def segments_to_plain_text(segments):
    lines = []
    for seg in segments:
        text = seg['text'].strip().replace("\n", " ")
        if text:
            lines.append(text)
    return "\n".join(lines)

def process_segments(raw_segments, max_dur):
    segments = []
    for seg in raw_segments:
        start = seg['start']
        end = seg['end']
        text = seg['text'].strip()
        dur = end - start
        if dur <= max_dur:
            segments.append({'start': start, 'end': end, 'text': text})
        else:
            # split into N chunks of approx equal time boundaries by words
            words = text.split()
            if len(words) <= 1:
                # fallback: keep original
                segments.append({'start': start, 'end': end, 'text': text})
            else:
                # estimated words per chunk
                n_chunks = math.ceil(dur / max_dur)
                chunk_size = math.ceil(len(words) / n_chunks)
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i+chunk_size]
                    rel_idx_start = i / len(words)
                    rel_idx_end = min((i+chunk_size)/len(words), 1.0)
                    cstart = start + rel_idx_start * dur
                    cend = start + rel_idx_end * dur
                    segments.append({'start': cstart, 'end': cend, 'text': " ".join(chunk_words)})
    return segments

def extract_mp3_from_mp4(mp4_path):
    """
    Extract MP3 audio track from MP4 file using ffmpeg.
    Returns path to the extracted MP3 file.
    """
    print(f"Extracting audio from MP4: {mp4_path}")
    
    # Create a temporary MP3 file in the same directory as the MP4
    mp4_dir = os.path.dirname(os.path.abspath(mp4_path))
    mp4_basename = os.path.splitext(os.path.basename(mp4_path))[0]
    mp3_path = os.path.join(mp4_dir, f"{mp4_basename}_extracted.mp3")
    
    # Construct the ffmpeg command to extract audio
    command = [
        "ffmpeg",
        "-i", mp4_path,
        "-vn",  # disable video
        "-ar", "44100",  # audio sampling rate
        "-ac", "2",  # stereo audio
        "-b:a", "192k",  # audio bitrate
        "-y",  # overwrite output file if it exists
        mp3_path
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully extracted audio to: {mp3_path}")
        return mp3_path
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Error extracting audio from {mp4_path}: {e.stderr}") from e
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found. Please ensure ffmpeg is installed and on PATH.") from None

def get_language_codes_help():
    """Returns a string with common language codes for user reference."""
    common_languages = {
        'en': 'English',
        'ru': 'Russian',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'pl': 'Polish',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'uk': 'Ukrainian',
    }
    lines = ["Common language codes:"]
    for code, name in sorted(common_languages.items()):
        lines.append(f"  {code}: {name}")
    return "\n".join(lines)

def detect_language(model, audio_path):
    """
    Detect the language of an audio file using Whisper.
    Returns a tuple of (language_code, confidence) where confidence is a float between 0 and 1.
    If confidence cannot be determined, returns (language_code, None).
    """
    print(f"Detecting language from: {audio_path}")
    try:
        # Load audio and detect language (this is faster than full transcription)
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]
        print(f"Detected language: {detected_lang} (confidence: {confidence:.2%})")
        return (detected_lang, confidence)
    except Exception as e:
        # Fallback: do a quick transcribe to get language
        print(f"Using fallback method for language detection...")
        result = model.transcribe(audio_path, verbose=False, task="transcribe")
        detected_lang = result.get("language", None)
        if detected_lang:
            print(f"Detected language: {detected_lang}")
            return (detected_lang, None)  # No confidence available in fallback
        else:
            raise Exception(f"Could not detect language: {e}")

def get_segments_from_file(model, audio_path, max_dur, language=None):
    print(f"\nProcessing: {audio_path}")
    transcribe_start = time.time()
    # Use specified language or let whisper detect if None
    result = model.transcribe(audio_path, verbose=False, language=language)
    transcribe_time = time.time() - transcribe_start
    mins, secs = divmod(transcribe_time, 60)
    print(f"Transcription completed in {int(mins):02d}:{secs:05.2f}")

    # Whisper returns 'segments': list with start/end/text
    raw_segments = result.get("segments", [])
    return process_segments(raw_segments, max_dur)

def main(folder_input=None, file_input=None, model="small", max_segment_duration=8.0, use_srt=True, language=None):
    """
    Transcribe audio files to SRT format using Whisper.
    
    Args:
        folder_input (str, optional): Path to folder containing audio files to process.
        file_input (str, optional): Path to a single audio file to process.
        model (str, optional): Whisper model name (tiny, base, small, medium, large). Default: "small"
        max_segment_duration (float, optional): Maximum segment duration in seconds. Default: 8.0
        use_srt (bool, optional): Whether to output SRT format. If False, outputs plain text. Default: True
        language (str, optional): Language code (e.g., 'en', 'ru'). If None, will auto-detect. Default: None
    
    Returns:
        str: Path to the created SRT/text file, or None if no files were processed.
    
    Note:
        When called from command line, interactive prompts are used if parameters are not provided.
    """
    # When called from CLI, use argparse and interactive prompts if parameters not provided
    interactive_mode = (folder_input is None and file_input is None)
    if interactive_mode:
        parser = argparse.ArgumentParser(description="Transcribe audio to SRT using Whisper")
        parser.add_argument("--model", default="small", help="Whisper model (tiny, base, small, medium, large)")
        parser.add_argument("--max_segment_duration", type=float, default=8.0,
                            help="Optional: re-chunk long segments to this maximum duration (seconds)")
        args = parser.parse_args()
        model = args.model
        max_segment_duration = args.max_segment_duration
        
        print(f"Current working directory: {os.getcwd()}")
        folder_input = input("Which folder to process? (Press Enter for single file): ").strip()
        
        if not folder_input:
            file_input = input("Which file to process? ").strip()
            
        srt_input = input("convert to srt? (y/n): ").strip().lower()
        use_srt = (srt_input != 'n')

    print(f"Loading Whisper model '{model}' (this may take a while)...")
    start_time = time.time()
    whisper_model = whisper.load_model(model)
    model_load_time = time.time() - start_time
    mins, secs = divmod(model_load_time, 60)
    print(f"Model loaded in {int(mins):02d}:{secs:05.2f}")

    # Normalize empty strings to None
    folder_input = folder_input if folder_input else None
    file_input = file_input if file_input else None

    files_to_process = []

    if folder_input:
        if os.path.isdir(folder_input):
            for root, dirs, files in os.walk(folder_input):
                for file in files:
                    if file.lower().endswith((".mp3", ".mp4")):
                        files_to_process.append(os.path.join(root, file))
            # Sort files by modification time (oldest first)
            files_to_process.sort(key=os.path.getmtime)
        elif os.path.isfile(folder_input):
             files_to_process.append(folder_input)
        else:
            print(f"Path not found: {folder_input}")
            return None
    elif file_input:
        if os.path.exists(file_input):
            files_to_process.append(file_input)
        else:
            print(f"File not found: {file_input}")
            return None

    if not files_to_process:
        print("No files to process.")
        return None

    print(f"Found {len(files_to_process)} files to process:")
    for f in files_to_process:
        print(f" - {f}")

    # Track files we extract for language detection so we can reuse them
    reused_extracted_files = {}  # Maps original file path to extracted MP3 path
    
    # Detect language if not provided
    if language is None and interactive_mode:
        if files_to_process:
            first_file = files_to_process[0]
            # Extract MP3 if needed for language detection
            temp_file_for_detection = None
            if first_file.lower().endswith('.mp4'):
                temp_file_for_detection = extract_mp3_from_mp4(first_file)
                detection_path = temp_file_for_detection
            else:
                detection_path = first_file
            
            try:
                detected_lang, confidence = detect_language(whisper_model, detection_path)
                print(f"\nDetected language: {detected_lang}")
                
                # Skip approval if confidence is above 90%
                if confidence is not None and confidence > 0.9:
                    language = detected_lang
                    print(f"High confidence ({confidence:.2%}), automatically using detected language: {language}")
                else:
                    print(get_language_codes_help())
                    lang_input = input("\nApprove this language? (Press Enter to approve, or enter language code to change): ").strip()
                    
                    if lang_input:
                        language = lang_input.lower()
                        print(f"Using specified language: {language}")
                    else:
                        language = detected_lang
                        print(f"Using detected language: {language}")
            except Exception as e:
                print(f"Warning: Could not detect language: {e}")
                print(get_language_codes_help())
                lang_input = input("\nEnter language code or press Enter for auto-detect: ").strip()
                if lang_input:
                    language = lang_input.lower()
                    print(f"Using specified language: {language}")
                else:
                    language = None
                    print("Will auto-detect language for each file")
            finally:
                # Clean up temporary file used for detection (interactive mode doesn't reuse)
                if temp_file_for_detection and os.path.exists(temp_file_for_detection):
                    try:
                        os.remove(temp_file_for_detection)
                    except:
                        pass
    elif language is None and files_to_process:
        # Auto-detect language in non-interactive mode
        # We'll extract MP3 here and reuse it for transcription to avoid double extraction
        first_file = files_to_process[0]
        temp_file_for_detection = None
        if first_file.lower().endswith('.mp4'):
            temp_file_for_detection = extract_mp3_from_mp4(first_file)
            detection_path = temp_file_for_detection
            # Store it for reuse in the transcription loop
            reused_extracted_files[first_file] = temp_file_for_detection
        else:
            detection_path = first_file
        
        try:
            detected_lang, confidence = detect_language(whisper_model, detection_path)
            # If confidence is low (below 90%), prompt the user
            if confidence is not None and confidence < 0.9:
                print(f"\n⚠️  Confidence in language detection: {confidence:.2%}")
                print(f"Detected language: {detected_lang}")
                print(get_language_codes_help())
                lang_input = input("\nEnter language code to use, or press Enter to use detected language: ").strip()
                
                if lang_input:
                    language = lang_input.lower()
                    print(f"Using specified language: {language}")
                else:
                    language = detected_lang
                    print(f"Using detected language: {language}")
            else:
                language = detected_lang
                if confidence is not None:
                    print(f"Auto-detected language: {language} (confidence: {confidence:.2%})")
                else:
                    print(f"Auto-detected language: {language}")
        except Exception as e:
            print(f"Warning: Could not detect language: {e}, will auto-detect during transcription")
            language = None
        # Don't delete temp_file_for_detection here - we'll reuse it in the loop below

    all_segments = []
    temp_files = []  # Track temporary MP3 files for cleanup
    
    for audio_path in files_to_process:
        # Check if file is MP4 and extract MP3 first
        # Reuse the file if we already extracted it for language detection
        if audio_path.lower().endswith('.mp4'):
            if audio_path in reused_extracted_files:
                # Reuse the already-extracted file
                extracted_mp3 = reused_extracted_files[audio_path]
                print(f"Reusing previously extracted audio: {extracted_mp3}")
            else:
                extracted_mp3 = extract_mp3_from_mp4(audio_path)
            temp_files.append(extracted_mp3)
            audio_path = extracted_mp3
        
        segs = get_segments_from_file(whisper_model, audio_path, max_segment_duration, language=language)
        all_segments.extend(segs)
    
    # Clean up temporary MP3 files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")

    # Determine output filename
    if folder_input and os.path.isdir(folder_input):
        # Folder mode -> Single file named after folder
        folder_name = os.path.basename(os.path.abspath(folder_input))
        out_name = folder_name
        if not out_name: out_name = "output"
    elif files_to_process:
        # Single file mode
        filename = os.path.basename(files_to_process[0])
        out_name = os.path.splitext(filename)[0]
    else:
        return None

    ext = ".srt" if use_srt else ".txt"
    outpath = out_name + ext
    
    # Generate content
    if use_srt:
        content = segments_to_srt(all_segments)
    else:
        content = segments_to_plain_text(all_segments)
        
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote output to {outpath}")
    
    return outpath

if __name__ == "__main__":
    script_start = time.time()
    main()
    total_time = time.time() - script_start
    mins, secs = divmod(total_time, 60)
    print(f"\nTotal script execution time: {int(mins):02d}:{secs:05.2f}")
