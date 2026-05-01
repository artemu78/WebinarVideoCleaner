# Transcribe to SRT Subsystem

The `transcribe_to_srt.py` script is a robust transcription and translation tool designed to convert audio/video files into formatted SRT subtitles or plain text. It is optimized for technical webinars and AI-related content, utilizing OpenAI Whisper for speech-to-text and Google's Gemini for optional translation.

## Goal
To accurately extract speech from media files, generate properly timed SRT subtitles (with anti-hallucination guardrails), and optionally translate them into another language while maintaining timecode integrity.

## Architecture & Logic

1. **Input Processing & Extraction**:
   - Accepts single files or folders containing `.mp4` or `.mp3` files.
   - If an MP4 is provided, it uses `ffmpeg` (via `subprocess`) to extract a high-quality MP3 file automatically, caching it to save time on future runs.

2. **Language Detection**:
   - Uses Whisper's `detect_language` on a small snippet of the audio.
   - If confidence is < 90% (in interactive mode), it prompts the user to verify or override the detected language.

3. **Transcription (Whisper)**:
   - Processes the audio using the specified Whisper model (e.g., `turbo`, `large`).
   - Employs strict **anti-hallucination settings**:
     - `condition_on_previous_text=False`: Prevents looping phrases.
     - `no_speech_threshold=0.6`: Filters out silence better.
     - `logprob_threshold=-1.0`: Discards low-confidence transcriptions.
   - Uses an `initial_prompt` (often enriched with a `webinar_topic`) to guide Whisper toward correct technical terminology.

4. **Segment Chunking & Formatting**:
   - Checks segment durations against `max_segment_duration`.
   - If a Whisper segment is too long, `process_segments` splits it into smaller chunks based on word count to ensure subtitles remain readable on screen.
   - Converts the finalized segments into standard SRT timestamp format (`HH:MM:SS,mmm`).

5. **Optional Gemini Translation**:
   - If requested, passes the generated SRT to the Gemini API (`gemini-2.0-flash`).
   - Uses a strict system prompt to translate the text while leaving SRT timing and indices completely intact.
   - Strips markdown formatting using `common_utils.clean_srt_response`.

## Key Parameters (`main()`)

- `folder_input` / `file_input`: Target file or directory to process.
- `--model`: Whisper model size to use (default: `turbo`).
- `--max_segment_duration`: Maximum length in seconds for a single subtitle block (default: `8.0`).
- `language`: Target language code. If omitted, the script attempts auto-detection.
- `--initial_prompt`: Contextual hint for Whisper to improve accuracy (e.g., "Это запись технического вебинара...").
- `--webinar_topic`: Optional string to further refine the `initial_prompt`.
- `--translate_to`: Language code (e.g., `en`, `ru`) to translate the final SRT into using Gemini.
- `skip_if_exists`: If `True`, skips processing if the target output file already exists.
- `gemini_api_key`: Optional API key for translation (otherwise fetches from environment/utils).

## Execution
Can be run as a standalone script interactively, or imported and called via `main(...)` in broader pipelines like `main_video_editor.py`.

Requirements: `openai-whisper>=20240930`, `ffmpeg` on PATH, and optionally `google-genai` for translation.