#!/usr/bin/env python3
import os
import time
import json
import re
import urllib.error
from dotenv import load_dotenv
from common_utils import (
    get_api_key,
    calculate_gemini_cost,
    format_ms_to_srt,
    safe_upload,
    retry_gemini_request,
    generate_content,
    parse_srt_to_blocks,
    blocks_to_srt_str,
    parse_json_array,
)

load_dotenv()
correct_srt_errors_model = os.environ.get("CORRECT_SRT_ERRORS_MODEL", "gemini-3-flash-preview")
correct_srt_errors_openrouter_model = os.environ.get("CORRECT_SRT_ERRORS_OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
correct_srt_errors_openrouter_inference_provider = os.environ.get(
    "CORRECT_SRT_ERRORS_OPENROUTER_INFERENCE_PROVIDER", ""
).strip()
# Large JSON-only responses from free OpenRouter routes are more likely to be
# truncated. Keep batches small enough that their corrected JSON fits reliably.
OPENROUTER_CORRECTION_BATCH_SIZE = int(os.environ.get("OPENROUTER_CORRECTION_BATCH_SIZE", "100"))


def _parse_correction_json_array(text):
    """Parse a JSON list of {id, text} from an LLM response; tolerate markdown fences."""
    return parse_json_array(text)


def _correction_output_suffix(provider):
    p = (provider or "gemini").strip().lower()
    return "_corrected_by_openrouter" if p == "openrouter" else "_corrected_by_gemini"

# Check if google.genai is available
try:
    import google.genai as genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as e:
    print("Error: google-genai package not found.")
    print("Please install the package: pip install google-genai")
    exit(1)

def parse_srt(filepath):
    """
    Parses an SRT file into a list of dictionaries.
    Each dict: {'index': str, 'start': str, 'end': str, 'text': str}
    """
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_srt_to_blocks(content)

def write_srt(blocks, filepath):
    """
    Writes a list of block dictionaries to an SRT file.
    """
    content = blocks_to_srt_str(blocks)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def _build_correction_prompt(language, webinar_topic, keep_fillers, file_attached=True):
    topic_context = ""
    if webinar_topic:
        topic_context = (
            f"\n        Context/Topic: The video is about '{webinar_topic}'. "
            "Use this context to better understand technical terms or context-specific language."
        )

    filler_instruction = "Correct spelling, grammar, and punctuation errors."
    if keep_fillers:
        filler_instruction = (
            "Correct spelling and technical terms. "
            "IMPORTANT: PRESERVE all filler words (uh, um, ah, эээ, ммм, ну) and non-verbal cues. "
            "Do NOT remove them during grammar correction."
        )

    source = "The attached JSON file contains" if file_attached else "The following JSON contains"
    return f"""
        You are a professional transcription editor.
        {source} subtitle lines from a video.
        Language: {language}.{topic_context}

        Your task:
        1. Read the parsed JSON list.
        2. {filler_instruction}
        3. DO NOT change the 'id'.
        4. DO NOT change the number of items.
        5. Return the result as a valid JSON list ONLY (no markdown, no commentary).

        Output Format:
        [
          {{"id": "1", "text": "Corrected text here"}},
          {{"id": "2", "text": "More corrected text"}}
        ]
        """


def process_srt_correction(
    srt_path,
    language="en",
    webinar_topic=None,
    keep_fillers=False,
    provider="gemini",
    correction_model=None,
):
    """
    Parses SRT, extracts text, asks an LLM to correct it via JSON in batches,
    and reconstructs the SRT with original timestamps.
    """
    provider = (provider or "gemini").strip().lower()
    model = correction_model or (
        correct_srt_errors_openrouter_model if provider == "openrouter" else correct_srt_errors_model
    )
    inference_provider = (
        correct_srt_errors_openrouter_inference_provider
        if provider == "openrouter" and model == correct_srt_errors_openrouter_model
        else None
    )

    base, ext = os.path.splitext(srt_path)
    out_suffix = _correction_output_suffix(provider)
    output_path = f"{base}{out_suffix}{ext}"

    if os.path.exists(output_path):
        print(f"✓ Corrected SRT already exists: {output_path} (Skipping step)")
        return output_path

    print(f"Parsing SRT file: {srt_path}...")
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return None

    original_blocks = parse_srt(srt_path)
    if not original_blocks:
        print("Error: No valid blocks found in SRT file.")
        return None

    print(f"✓ Parsed {len(original_blocks)} subtitle blocks.")

    # Batch sizes: Gemini can handle more if using File API, but OpenRouter needs prompt limits.
    # We'll use a conservative batch size for both for simplicity when using text-based prompts.
    batch_size = OPENROUTER_CORRECTION_BATCH_SIZE if provider == "openrouter" else 400
    batches = [original_blocks[i : i + batch_size] for i in range(0, len(original_blocks), batch_size)]
    print(f"Split into {len(batches)} batches for processing (Provider: {provider}, Batch Size: {batch_size}).")

    corrected_map = {}
    successful_batches = 0
    
    for i, batch in enumerate(batches):
        print(f"\nProcessing Batch {i+1}/{len(batches)} ({len(batch)} items)...")
        input_payload = [{"id": b["index"], "text": b["text"]} for b in batch]
        batch_json = json.dumps(input_payload, indent=2, ensure_ascii=False)
        
        prompt = (
            _build_correction_prompt(language, webinar_topic, keep_fillers, file_attached=False).strip()
            + "\n\nInput JSON:\n"
            + batch_json
        )
        
        print(f"  Requesting correction from {provider} (model: {model})...")
        try:
            # Use unified generate_content which handles Gemini/OpenRouter and retries
            raw = generate_content(
                provider=provider, 
                model=model, 
                prompt=prompt,
                response_mime_type="application/json" if provider == "gemini" else None,
                inference_provider=inference_provider,
            )
            
            batch_corrected = _parse_correction_json_array(raw)
            if not batch_corrected:
                print(f"  ❌ Failed to parse JSON from batch {i+1} response.")
                continue
                
            for item in batch_corrected:
                corrected_map[item["id"]] = item["text"]
            successful_batches += 1
            print(f"  ✓ Batch {i+1} Success.")
            
        except urllib.error.HTTPError as error:
            print(
                f"  ❌ OpenRouter rejected batch {i+1} with HTTP {error.code}. "
                "Stopping Step 2; no corrected file will be created."
            )
            return None
        except Exception as e:
            print(f"  ❌ Error in batch {i+1}: {e}")

    if successful_batches == 0:
        print("❌ No batches were corrected. Step 2 stopped; no corrected file was created.")
        return None

    correction_count = 0
    for block in original_blocks:
        if block["index"] in corrected_map:
            new_text = corrected_map[block["index"]]
            if new_text != block["text"]:
                block["text"] = new_text
                correction_count += 1

    print(f"✓ Applied corrections to {correction_count} blocks.")

    write_srt(original_blocks, output_path)
    print(f"✓ Corrected SRT saved to: {output_path}")

    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        start_time = time.time()
        args = sys.argv[1:]
        provider = "gemini"
        if len(args) >= 2 and args[-1].lower() in ("openrouter", "gemini"):
            provider = args.pop().lower()

        srt_file = args[0]
        lang = args[1] if len(args) > 1 else "en"
        topic = args[2] if len(args) > 2 else None

        process_srt_correction(srt_file, lang, topic, provider=provider)

        elapsed_time = time.time() - start_time
        print(f"Total execution time: {format_ms_to_srt(elapsed_time * 1000)}")
    else:
        print(
            "Usage: python correct_srt_errors.py <srt_file> [language] [webinar_topic] [gemini|openrouter]\n"
            "  If the last argument is gemini or openrouter, it selects the provider "
            "(e.g. `script.py clip_ru.srt ru openrouter`)."
        )
