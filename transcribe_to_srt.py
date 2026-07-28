# transcribe_to_srt.py
# Requirements:
#   pip install -U "openai-whisper>=20240930"  (or latest whisper)
#   ffmpeg must be installed and on PATH (for whisper)
# Usage:
#   python transcribe_to_srt.py --model small

import argparse
import math
import os
import sys
import time
import subprocess
from datetime import timedelta
import re
from dotenv import load_dotenv

from common_utils import get_tracked_user_input_seconds, timed_input, track_user_input_time_scope

load_dotenv()
translate_srt_model = os.getenv("TRANSLATE_SRT_MODEL", "gemini-3.1-flash-lite-preview")
translate_srt_openrouter_model = os.getenv(
    "TRANSLATE_SRT_OPENROUTER_MODEL", "google/gemini-3-flash-preview"
)
translate_srt_openrouter_inference_provider = os.getenv(
    "TRANSLATE_SRT_OPENROUTER_INFERENCE_PROVIDER", ""
).strip()

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
    raise SystemExit(
        "Failed to import the `whisper` package (openai-whisper).\n"
        f"  Python executable: {sys.executable}\n"
        f"  Import error: {type(e).__name__}: {e}\n"
        "Install for this same interpreter, then retry:\n"
        f"  {sys.executable} -m pip install -U openai-whisper\n"
        "Also ensure ffmpeg is installed and on PATH."
    ) from e


def translate_srt(srt_content, target_language, provider="gemini", api_key=None, keep_fillers=False, source_language=None, webinar_topic=None):
    """
    Translates SRT content to another language using the configured GenAI provider.
    Uses chunked translation with JSON to strictly maintain timing alignment.
    """
    try:
        from common_utils import parse_srt_to_blocks, blocks_to_srt_str, parse_json_array, generate_content
    except ImportError:
        print("Error: common_utils.py not found. Please ensure it is in the same directory.")
        return srt_content

    original_blocks = parse_srt_to_blocks(srt_content)
    if not original_blocks:
        print("Warning: No valid SRT blocks found to translate.")
        return srt_content

    # We use a system instruction and a clear prompt to ensure the output is valid JSON
    filler_instruction = "Translate the text."
    if keep_fillers:
        filler_instruction = (
            "Translate the text while PRESERVING all filler words (uh, um, ah, эээ, ммм, ну) and non-verbal cues. "
            "Enhance the narration with non-verbal cues to prevent awkward silences and sound natural. "
            "PROACTIVELY insert the following emotional tags where natural pauses, breaths, or emotional shifts occur in the sentence: "
            "[laughter], [sigh], [confirmation-en], [question-en], [question-ah], [question-oh], "
            "[question-ei], [question-yi], [surprise-ah], [surprise-oh], [surprise-wa], [surprise-yo], [dissatisfaction-hnn]. "
            "CRITICAL: Do NOT invent your own tags. ONLY use the exact tags listed above."
        )

    glossary_instruction = ""
    if target_language.lower() == 'ru' and (source_language is None or source_language.lower() == 'en'):
        terms_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TERMS_FOR_TRANSLATION.md")
        if os.path.exists(terms_file):
            print(f"Loading IT glossary from {terms_file} for EN->RU translation...")
            try:
                with open(terms_file, 'r', encoding='utf-8') as f:
                    glossary_content = f.read()
                glossary_instruction = f"\n\nUSE THE FOLLOWING GLOSSARY FOR IT TERMS:\n{glossary_content}\n"
            except Exception as e:
                print(f"Warning: Could not read terms file: {e}")

    topic_context = ""
    if webinar_topic:
        topic_context = f"\nContext/Topic: The video is about '{webinar_topic}'."

    chunk_size = 40  # Segments per chunk for translation
    translated_blocks = []
    
    selected_provider = (provider or "gemini").strip().lower()
    model = (
        translate_srt_openrouter_model
        if selected_provider == "openrouter"
        else translate_srt_model
    )
    inference_provider = (
        translate_srt_openrouter_inference_provider
        if selected_provider == "openrouter" and model == translate_srt_openrouter_model
        else None
    )
    
    print(f"Translating {len(original_blocks)} blocks in chunks of {chunk_size} via {selected_provider} ({model})...")

    # Keep backward compatibility: if caller passes gemini_api_key, prefer it.
    if api_key and selected_provider == "gemini":
        os.environ["GEMINI_API_KEY"] = api_key

    for i in range(0, len(original_blocks), chunk_size):
        batch = original_blocks[i:i + chunk_size]
        print(f"  Processing chunk {i//chunk_size + 1}/{(len(original_blocks)-1)//chunk_size + 1} ({len(batch)} blocks)...")
        
        input_payload = [{"id": b["index"], "text": b["text"]} for b in batch]
        import json
        batch_json = json.dumps(input_payload, indent=2, ensure_ascii=False)
        
        prompt = (
            f"You are a professional translator. Translate the following JSON list of subtitle lines to {target_language}. "
            f"{topic_context}\n"
            f"RULES:\n"
            f"1. {filler_instruction}\n"
            f"2. DO NOT change the 'id'.\n"
            f"3. DO NOT change the number of items.\n"
            f"4. Maintain technical context and natural flow.\n"
            f"{glossary_instruction}\n"
            f"Output ONLY the translated JSON list without any markdown formatting or extra text.\n\n"
            f"Input JSON:\n{batch_json}"
        )
        
        try:
            raw_text = generate_content(
                provider=selected_provider,
                model=model,
                prompt=prompt,
                inference_provider=inference_provider,
            )
            batch_translated = parse_json_array(raw_text)
            
            if batch_translated and len(batch_translated) > 0:
                # Create a map for quick lookup
                trans_map = {str(item.get("id")): item.get("text", "") for item in batch_translated if "id" in item}
                
                # Apply translations back to original batch structure to preserve timestamps
                for b in batch:
                    translated_text = trans_map.get(str(b["index"]))
                    if translated_text:
                        b["text"] = translated_text
                    translated_blocks.append(b)
                print(f"    ✓ Chunk {i//chunk_size + 1} success.")
            else:
                print(f"    ⚠️ Failed to parse JSON for chunk {i//chunk_size + 1}, using original text.")
                translated_blocks.extend(batch)
        except Exception as e:
            print(f"    ❌ Error during chunk {i//chunk_size + 1} translation: {e}")
            translated_blocks.extend(batch)

    return blocks_to_srt_str(translated_blocks)


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

def extract_mp3_from_mp4(mp4_path, interactive=True):
    """
    Extract MP3 audio track from MP4 file using ffmpeg.
    Returns path to the extracted MP3 file.
    If the MP3 already exists: when interactive, asks whether to re-create; otherwise
    reuses the file without prompting. Answering no (or non-interactive) skips ffmpeg.
    """
    mp3_path = get_extracted_mp3_path(mp4_path)

    if os.path.exists(mp3_path):
        print(f"Extracted audio file already exists: {mp3_path}")
        if interactive:
            recreate = timed_input("Re-create it? (y/n): ").strip().lower()
            if recreate != 'y':
                print(f"Using existing audio file: {mp3_path}")
                return mp3_path
        else:
            print(f"Using existing audio file (non-interactive): {mp3_path}")
            return mp3_path

    print(f"Extracting audio from MP4: {mp4_path}")
    
    # Use ffmpeg to extract audio
    command = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "2",
        mp3_path
    ]
    
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

def get_segments_from_file(model, audio_path, max_dur, language=None, initial_prompt=None, keep_fillers=False):
    print(f"\nProcessing: {audio_path}")
    # Anti-hallucination vs Style preservation:
    # condition_on_previous_text=False: Prevents looping phrases but might lose style/fillers after first 30s.
    # condition_on_previous_text=True: Maintains style/fillers better across windows but increases hallucination risk.
    cond_on_prev = True if keep_fillers else False
    print(f"Whisper settings: condition_on_previous_text={cond_on_prev}, no_speech_threshold=0.6")
    
    if initial_prompt:
        print(f"Using initial prompt: {initial_prompt}")
    transcribe_start = time.time()
    # Use specified language or let whisper detect if None
    result = model.transcribe(
        audio_path, 
        verbose=False, 
        language=language,
        condition_on_previous_text=cond_on_prev,
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

def main(folder_input=None, file_input=None, model="turbo", max_segment_duration=8.0, use_srt=True, language=None, initial_prompt="Это запись технического вебинара или видео про программирование и AI.", webinar_topic=None, skip_if_exists=False, translate_to=None, gemini_api_key=None, keep_fillers=False, provider="gemini"):
    """
    Transcribe audio files to SRT format using Whisper.
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
        parser.add_argument("--translate_to", type=str, default=None,
                            help="Optional: translate subtitles to another language (e.g., 'en', 'ru') using Gemini.")
        parser.add_argument("--gemini_api_key", type=str, default=None,
                            help="Optional: Gemini API key for translation.")
        parser.add_argument("--provider", type=str, default="gemini",
                            help="GenAI provider for translation: gemini or openrouter.")
        parser.add_argument("--keep_fillers", action="store_true", help="Try to keep filler words like 'uh', 'um', 'ah'.")
        parser.add_argument("--language", type=str, default=None,
                            help="Optional: language of the input file.")
        args = parser.parse_args()
        model = args.model
        max_segment_duration = args.max_segment_duration
        initial_prompt = args.initial_prompt
        webinar_topic = args.webinar_topic
        translate_to = args.translate_to
        gemini_api_key = args.gemini_api_key
        provider = args.provider
        keep_fillers = args.keep_fillers
        language = args.language
        print(f"Current working directory: {os.getcwd()}")
        folder_input = timed_input("Which folder to process? (Press Enter for single file): ").strip()
        
        if not folder_input:
            file_input = timed_input("Which file to process? ").strip()
            
        srt_input = timed_input("convert to srt? (y/n): ").strip().lower()
        use_srt = (srt_input != 'n')

        if use_srt and not translate_to:
            translate_input = timed_input("Translate to another language? (e.g. en, ru, or leave empty for no): ").strip()
            if translate_input:
                translate_to = translate_input

        if use_srt and translate_to:
            provider_input = timed_input("GenAI Provider? (1) Gemini (2) OpenRouter [1]: ").strip().lower()
            if provider_input in ("2", "openrouter", "open-router", "open router"):
                provider = "openrouter"
            else:
                provider = "gemini"
        
        if not keep_fillers:
            fillers_input = timed_input("Keep filler words (uh, um, ah...)? (y/n): ").strip().lower()
            keep_fillers = (fillers_input == 'y')

        if not language:
            language = timed_input("Enter language of the input file (press Enter for auto-detect): ").strip()
            if not language:
                language = None
            else:
                language = language.lower()
                print(f"Using specified language: {language}")
        else:
            print(f"Using detected language: {language}")

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

    # 3. Check if output file already exists (may skip Whisper only; translation still runs below)
    reuse_existing_srt = False
    if os.path.exists(outpath):
        print(f"\nOutput file already exists: {outpath}")
        
        should_skip = False
        if skip_if_exists:
            print("skip_if_exists=True: Skipping generation and using existing file.")
            should_skip = True
        elif interactive_mode:
            reproduce = timed_input("Regenerate it? (y/n): ").strip().lower()
            if reproduce != 'y':
                print("Skipping generation and using existing file.")
                should_skip = True
        else:
            print("Non-interactive mode: skip_if_exists is False, but skipping by default for safety in non-interactive mode.")
            should_skip = True

        if should_skip:
            reuse_existing_srt = True
            # If language was not provided, ask for it since we rely on it later (e.g. translation source language)
            if language is None:
                if interactive_mode:
                    print(get_language_codes_help())
                    lang_input = timed_input("Enter language of existing file (press Enter for 'ru'): ").strip()
                    language = lang_input.lower() if lang_input else 'ru'
                else:
                    print("Language not specified. Defaulting to 'ru' for existing file.")
                    language = 'ru'
        else:
            print("Regenerating...")

    if not reuse_existing_srt:
        # 4. Load Model (only if needed)
        print(f"Whisper package version: {whisper.__version__}")
        print(f"Loading Whisper model '{model}' (this may take a while)...")
        start_time = time.time()
        whisper_model = whisper.load_model(model)
        model_load_time = time.time() - start_time
        mins, secs = divmod(model_load_time, 60)
        print(f"Model loaded in {int(mins):02d}:{secs:05.2f}")

        # Enriched initial prompt
        if webinar_topic:
            initial_prompt = f"{initial_prompt} Topic: {webinar_topic}"
            print(f"Enriched initial prompt with topic: {webinar_topic}")

        if keep_fillers:
            if language == 'en' or (language is None and "Это запись" not in initial_prompt):
                 # For English, use a style-based prompt that encourages fillers
                 initial_prompt = f"Hello. Uh, welcome to this video. Um, today we are, ah, talking about software. {initial_prompt}"
            else:
                 initial_prompt = f"{initial_prompt} Keep fillers: uh, um, ah, эээ, ммм, ну."
            print(f"Enriched initial prompt to keep fillers.")

        # Track files we extract for language detection so we can reuse them
        reused_extracted_files = {}  # Maps original file path to extracted MP3 path
        
        # Detect language if not provided
        if language is None and interactive_mode:
            if files_to_process:
                first_file = files_to_process[0]
                # Extract MP3 if needed for language detection
                temp_file_for_detection = None
                if first_file.lower().endswith('.mp4'):
                    temp_file_for_detection = extract_mp3_from_mp4(
                        first_file, interactive=interactive_mode
                    )
                    reused_extracted_files[first_file] = temp_file_for_detection
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
                        lang_input = timed_input("\nApprove this language? (Press Enter to approve, or enter language code to change): ").strip()
                        
                        if lang_input:
                            language = lang_input.lower()
                            print(f"Using specified language: {language}")
                        else:
                            language = detected_lang
                            print(f"Using detected language: {language}")
                except Exception as e:
                    print(f"Warning: Could not detect language: {e}")
                    print(get_language_codes_help())
                    lang_input = timed_input("\nEnter language code or press Enter for auto-detect: ").strip()
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
                temp_file_for_detection = extract_mp3_from_mp4(first_file, interactive=False)
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
                    lang_input = timed_input("\nEnter language code to use, or press Enter to use detected language: ").strip()
                    
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
                    extracted_mp3 = extract_mp3_from_mp4(
                        audio_path, interactive=interactive_mode
                    )
                # We don't add to temp_files anymore because we want to keep extracted files to save time in future runs
                # temp_files.append(extracted_mp3)
                audio_path = extracted_mp3
            else:
                print(f"Direct audio file provided, skipping MP4->MP3 extraction: {audio_path}")
            
            segs, detected_file_lang = get_segments_from_file(whisper_model, audio_path, max_segment_duration, language=language, initial_prompt=initial_prompt, keep_fillers=keep_fillers)
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
    else:
        print(f"Using existing transcript from: {outpath}")
        with open(outpath, "r", encoding="utf-8") as f:
            content = f.read()

    if translate_to and use_srt:
        print(f"\n--- Translation Step ---")
        translated_content = translate_srt(
            content,
            translate_to,
            provider=provider,
            api_key=gemini_api_key,
            keep_fillers=keep_fillers,
            source_language=language,
            webinar_topic=webinar_topic,
        )
        
        # New output path for translation
        base, ext = os.path.splitext(outpath)
        # Avoid double extension if we're already translating a translated file (unlikely but safe)
        if base.endswith(f"_{translate_to}"):
             translated_outpath = outpath
        else:
            translated_outpath = f"{base}_{translate_to}{ext}"
        
        with open(translated_outpath, "w", encoding="utf-8") as f:
            f.write(translated_content)
            
        print(f"Wrote translated output to {translated_outpath}")
        
        # Auto-correct SRT using the same GenAI provider as translation (Gemini file upload vs OpenRouter inline).
        try:
            print("Running SRT auto-correction...")
            import correct_srt_errors

            correct_srt_errors.process_srt_correction(
                translated_outpath,
                language=translate_to,
                webinar_topic=webinar_topic,
                keep_fillers=keep_fillers,
                provider=provider,
            )
            print("SRT auto-correction completed.")
        except Exception as e:
            print(f"Warning: SRT auto-correction failed: {e}")
            
        return translated_outpath, translate_to

    return outpath, language

if __name__ == "__main__":
    with track_user_input_time_scope():
        script_start = time.perf_counter()
        main()
    total_time = time.perf_counter() - script_start - get_tracked_user_input_seconds()
    mins, secs = divmod(total_time, 60)
    print(f"\nTotal script execution time: {int(mins):02d}:{secs:05.2f}")
