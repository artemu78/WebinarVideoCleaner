# Webinar Video Cleaner

This project provides an automated workflow to clean and enhance webinar recordings. It uses OpenAI's Whisper for transcription and Google's Gemini AI to intelligently identify and remove irrelevant segments (such as silence, filler words, or off-topic setup), correct transcription errors, and generate content chapters.

## Features

- **Automated Transcription**: Converts MP4 video audio to SRT subtitles using the Whisper model.
- **Smart Content Analysis**: Uses Google Gemini AI to analyze the transcript and identify "useless" ranges to delete.
- **Video Cutting**: Automatically removes the identified ranges from the original MP4 file.
- **Subtitle Correction**: Uses AI to fix transcription errors and improve subtitle quality.
- **Subtitle Synchronization**: Adjusts SRT timestamps to perfectly match the edited (cut) video.
- **Chapter Generation**: Generates structured chapters with titles and summaries for the final video.
- **Cost & Time Tracking**: Monitors script execution time and estimates Gemini API costs.

## Prerequisites

- **Python 3.8+**
- **FFmpeg**: Required for video and audio processing. Any standard installation should work.
- **Google Gemini API Key**: Required for AI analysis and text processing.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    Ensure you have `pip` installed.
    ```bash
    pip install -e .
    # Or if requirements.txt is available
    # pip install -r requirements.txt
    ```

3.  **Environment Setup**:
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```bash
    GOOGLE_API_KEY=your_actual_api_key_here
    ```

## Usage

The main entry point for the application is `main_video_editor.py`.

1.  **Run the script**:
    ```bash
    python3 main_video_editor.py
    ```

2.  **Follow the interactive prompts**:
    - **Enter MP4 Path**: Paste the full path to your webinar video file.
    - **Select Mode**:
        - `1`: **Full Video Cleaner** (Transcribe + Correct + Analyze + Cut + Chapters).
        - `2`: **Transcription & Chapters Only** (Skips the video cutting step).
    - **Webinar Topic**: (Optional) Provide a topic to help the AI understand context better during correction.

3.  **Review Outputs**:
    All generated files are saved in the same directory as the original video. The script provides a summary at the end containing paths to:
    - Original SRT
    - Corrected SRT
    - Cleaned (Cut) Video
    - Chapters File
    - Usage Stats

## Workflow Details

The system operates through a cascade of specialized scripts orchestrated by `main_video_editor.py`:

1.  **Transcription** (`transcribe_to_srt.py`):
    Extracts audio and generates an initial SRT transcript.

2.  **Correction** (`correct_srt_errors.py`):
    Sends the transcript to Gemini to fix spelling, grammar, and recognition errors.

3.  **Analysis** (`audio_cleaner.py`):
    *Full Mode Only.* Analyzes the text to find start and end timestamps of segments that should be removed.

4.  **Cutting** (`cut_mp4.py`):
    *Full Mode Only.* Uses FFmpeg to physically remove the identified segments from the video file.

5.  **Re-synchronization** (`apply_cuts_to_srt.py`):
    *Full Mode Only.* Adjusts the timestamps in the subtitle file so they align with the new, shorter video.

6.  **Chapters** (`generate_chapters.py`):
    Analyzes the final content to generate a list of chapters with timestamps.

## Project Structure

- `main_video_editor.py`: Orchestrator script that manages the entire pipeline.
- `transcribe_to_srt.py`: Handles Whisper transcription.
- `audio_cleaner.py`: Interface for Gemini AI to identify cuts.
- `cut_mp4.py`: Handles video processing and cutting logic.
- `correct_srt_errors.py`: Logic for AI-based subtitle correction.
- `apply_cuts_to_srt.py`: Utilities for SRT timestamp manipulation.
- `generate_chapters.py`: Generates video chapters.
- `common_utils.py`: Shared utilities for path handling, logging, and cost calculation.
