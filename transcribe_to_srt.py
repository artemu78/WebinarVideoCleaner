# transcribe_to_srt.py
# Requirements:
#   pip install -U "openai-whisper>=20240930"  (or latest whisper)
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

def has_audio_stream(file_path):
    """
    Checks if the media file has an audio stream using ffprobe.
    Bypassing check for now because it might be hanging.
    """
    return True

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

def get_extracted_mp3_path(mp4_path):
    """
    Returns the expected path for the extracted MP3 file.
    """
    mp4_dir = os.path.dirname(os.path.abspath(mp4_path))
    mp4_basename = os.path.splitext(os.path.basename(mp4_path))[0]
    return os.path.join(mp4_dir, f"{mp4_basename}_extracted.mp3")

def extract_mp3_from_mp4(mp4_path):
    """
    Extract MP3 audio track from MP4 file using ffmpeg.
    Returns path to the extracted MP3 file.
    """
    
    mp3_path = get_extracted_mp3_path(mp4_path)

    if os.path.exists(mp3_path):
        print(f"Extracted audio file already exists: {mp3_path}")
        recreate = input("Re-create it? (y/n): ").strip().lower()
        if recreate != 'y':
            print(f"Using existing audio file: {mp3_path}")
            return mp3_path

    print(f"Extracting audio from MP4: {mp4_path}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully extracted audio to: {mp3_path}", flush=True)
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

def get_segments_from_file(model, audio_path, max_dur, language=None, initial_prompt=None):
    print(f"\nProcessing: {audio_path}")
    print("Using anti-hallucination settings: condition_on_previous_text=False, no_speech_threshold=0.6")
    if initial_prompt:
        print(f"Using initial prompt: {initial_prompt}")
    transcribe_start = time.time()
    # Use specified language or let whisper detect if None
    # Anti-hallucination settings:
    # condition_on_previous_text=False: Prevents looping phrases like "Subtitle Editor"
    # no_speech_threshold=0.6: Filters out silence better (default is 0.6, but ensuring it's set)
    # logprob_threshold=-1.0: Discards low confidence transcriptions
    result = model.transcribe(
        audio_path, 
        verbose=False, 
        language=language,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        initial_prompt=initial_prompt
    )
    transcribe_time = time.time() - transcribe_start
    mins, secs = divmod(transcribe_time, 60)
    print(f"Transcription completed in {int(mins):02d}:{secs:05.2f}")

    # Whisper returns 'segments': list with start/end/text
    raw_segments = result.get("segments", [])
    detected_lang = result.get("language")
    return process_segments(raw_segments, max_dur), detected_lang

def main(folder_input=None, file_input=None, model="turbo", max_segment_duration=8.0, use_srt=True, language=None, initial_prompt="Это запись технического вебинара или видео про программирование и AI.", webinar_topic=None, skip_if_exists=False):
    """
    Transcribe audio files to SRT format using Whisper.
    ...
    """
    # When called from CLI, use argparse and interactive prompts if parameters not provided
    interactive_mode = (folder_input is None and file_input is None)
    if interactive_mode:
        parser = argparse.ArgumentParser(description="Transcribe audio to SRT using Whisper")
        parser.add_argument("--model", default="turbo", help="Whisper model (tiny, base, small, medium, large, turbo)")
        parser.add_argument("--max_segment_duration", type=float, default=8.0,
                            help="Optional: re-chunk long segments to this maximum duration (seconds)")
        parser.add_argument("--initial_prompt", type=str, default="Это запись технического вебинара или видео про программирование и AI.", 
                            help="Optional: provide a prompt to guide the transcription and reduce hallucinations.")
        parser.add_argument("--webinar_topic", type=str, default=None,
                            help="Optional: provide a topic for the webinar to guide transcription.")
        args = parser.parse_args()
        model = args.model
        max_segment_duration = args.max_segment_duration
        initial_prompt = args.initial_prompt
        webinar_topic = args.webinar_topic
        
        print(f"Current working directory: {os.getcwd()}")
        folder_input = input("Which folder to process? (Press Enter for single file): ").strip()
        
        if not folder_input:
            file_input = input("Which file to process? ").strip()
            
        srt_input = input("convert to srt? (y/n): ").strip().lower()
        use_srt = (srt_input != 'n')

    # Normalize empty strings to None
    folder_input = folder_input if folder_input else None
    file_input = file_input if file_input else None

    # 1. Identify files to process first
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
            return None, None
    elif file_input:
        if os.path.exists(file_input):
            files_to_process.append(file_input)
        else:
            print(f"File not found: {file_input}")
            return None, None

    if not files_to_process:
        print("No files to process.")
        return None, None

    # Filter out files without audio
    # Bypassing audio check as it hangs on some files (e.g. iCloud-backed)
    valid_files = files_to_process

    files_to_process = valid_files

    if not files_to_process:
        print("No files to process.", flush=True)
        return None, None

    print(f"Found {len(files_to_process)} files to process:")
    for f in files_to_process:
        print(f" - {f}")

    # 2. Determine output filename early
    if folder_input and os.path.isdir(folder_input):
        # Folder mode -> Single file named after folder
        folder_name = os.path.basename(os.path.abspath(folder_input))
        out_name = folder_name
        if not out_name: out_name = "output"
    elif files_to_process:
        # Single file mode
        input_path = os.path.abspath(files_to_process[0])
        input_dir = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        base_name = os.path.splitext(filename)[0]
        out_name = os.path.join(input_dir, base_name)
    else:
        return None, None

    ext = ".srt" if use_srt else ".txt"
    outpath = out_name + ext

    # 3. Check if output file already exists
    if os.path.exists(outpath):
        print(f"\nOutput file already exists: {outpath}")
        
        should_skip = False
        if skip_if_exists:
            print("skip_if_exists=True: Skipping generation and using existing file.")
            should_skip = True
        elif interactive_mode:
            reproduce = input("Regenerate it? (y/n): ").strip().lower()
            if reproduce != 'y':
                print("Skipping generation and using existing file.")
                should_skip = True
        else:
            print("Non-interactive mode: skip_if_exists is False, but skipping by default for safety in non-interactive mode.")
            should_skip = True

        if should_skip:
            # If language was not provided, ask for it since we rely on it later
            if language is None:
                if interactive_mode:
                    print(get_language_codes_help())
                    lang_input = input("Enter language of existing file (press Enter for 'en'): ").strip()
                    language = lang_input.lower() if lang_input else 'en'
                else:
                    print("Language not specified. Defaulting to 'en' for existing file.")
                    language = 'en'
            return outpath, language
        else:
            print("Regenerating...")

    # 4. Load Model (only if needed)
    print(f"Whisper package version: {whisper.__version__}")
    print(f"Loading Whisper model '{model}' (this may take a while)...")
    start_time = time.time()
    whisper_model = whisper.load_model(model)
    model_load_time = time.time() - start_time
    mins, secs = divmod(model_load_time, 60)
    print(f"Model loaded in {int(mins):02d}:{secs:05.2f}")

    # Enriched initial prompt (moved here, logic remains same)
    if webinar_topic:
        initial_prompt = f"{initial_prompt} Topic: {webinar_topic}"
        print(f"Enriched initial prompt with topic: {webinar_topic}")

    # Track files we extract for language detection so we can reuse them
    reused_extracted_files = {}  # Maps original file path to extracted MP3 path
    
    # Detect language if not provided
    if language is None and interactive_mode:
        if files_to_process:
            first_file = files_to_process[0]
            # Extract MP3 if needed for language detection
            temp_file_for_detection = None
            if first_file.lower().endswith('.mp4'):
                mp3_path = get_extracted_mp3_path(first_file)
                if os.path.exists(mp3_path):
                    print(f"Using existing audio for language detection: {mp3_path}")
                    temp_file_for_detection = mp3_path
                else:
                    temp_file_for_detection = extract_mp3_from_mp4(first_file)
                detection_path = temp_file_for_detection
            else:
                detection_path = first_file
            
            try:
                detected_lang, confidence = detect_language(whisper_model, detection_path)
                print(f"\nDetected language: {detected_lang}. Confidence: {confidence:.2%}")
                
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
                # Cleanup disabled to allow reuse
                pass
                """
                if temp_file_for_detection and os.path.exists(temp_file_for_detection):
                    try:
                        os.remove(temp_file_for_detection)
                    except:
                        pass
                """
    elif language is None and files_to_process:
        # Auto-detect language in non-interactive mode
        # We'll extract MP3 here and reuse it for transcription to avoid double extraction
        first_file = files_to_process[0]
        temp_file_for_detection = None
        if first_file.lower().endswith('.mp4'):
            mp3_path = get_extracted_mp3_path(first_file)
            if os.path.exists(mp3_path):
                print(f"Using existing audio for language detection: {mp3_path}")
                temp_file_for_detection = mp3_path
            else:
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
                mp3_path = get_extracted_mp3_path(audio_path)
                if os.path.exists(mp3_path):
                    print(f"Using existing audio: {mp3_path}")
                    extracted_mp3 = mp3_path
                else:
                    extracted_mp3 = extract_mp3_from_mp4(audio_path)
            # We don't add to temp_files anymore because we want to keep extracted files to save time in future runs
            # temp_files.append(extracted_mp3)
            audio_path = extracted_mp3
        
        segs, detected_file_lang = get_segments_from_file(whisper_model, audio_path, max_segment_duration, language=language, initial_prompt=initial_prompt)
        all_segments.extend(segs)
        if language is None and detected_file_lang:
            language = detected_file_lang
    
    # Temporary files cleanup is disabled to allow reuse in future runs (save time)
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    """

    # Generate content
    if use_srt:
        content = segments_to_srt(all_segments)
    else:
        content = segments_to_plain_text(all_segments)
        
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote output to {outpath}")
    
    return outpath, language

if __name__ == "__main__":
    script_start = time.time()
    main()
    total_time = time.time() - script_start
    mins, secs = divmod(total_time, 60)
    print(f"\nTotal script execution time: {int(mins):02d}:{secs:05.2f}")
