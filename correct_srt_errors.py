#!/usr/bin/env python3
import os
import time
import json
import re
from dotenv import load_dotenv
from common_utils import get_api_key, clean_srt_response, calculate_gemini_cost, format_ms_to_srt

load_dotenv()

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
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get blocks (handling potential CRLF)
    blocks = re.split(r'\n\s*\n', content.strip())
    
    parsed_blocks = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # First line is index
            index = lines[0].strip()
            # Second line is timestamp
            timestamps = lines[1].strip()
            # The rest is text
            text = "\n".join(lines[2:])
            
            # Extract start and end times
            try:
                start, end = timestamps.split(' --> ')
                parsed_blocks.append({
                    'index': index,
                    'start': start,
                    'end': end,
                    'text': text
                })
            except ValueError:
                print(f"Warning: Skipping malformed block: {lines[0]}")
                continue
                
    return parsed_blocks

def write_srt(blocks, filepath):
    """
    Writes a list of block dictionaries to an SRT file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for block in blocks:
            f.write(f"{block['index']}\n")
            f.write(f"{block['start']} --> {block['end']}\n")
            f.write(f"{block['text']}\n\n")

def process_srt_correction(srt_path, language="en", webinar_topic=None):
    """
    Parses SRT, extracts text, asks Gemini to correct it via JSON in batches, 
    and reconstructs the SRT with original timestamps.
    """
    
    # Step 1: Initialize Client
    print(f"Initializing Gemini client for SRT correction (Language: {language})...")
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    
    # Step 2: Parse SRT locally
    print(f"Parsing SRT file: {srt_path}...")
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return None
        
    original_blocks = parse_srt(srt_path)
    if not original_blocks:
        print("Error: No valid blocks found in SRT file.")
        return None
        
    print(f"✓ Parsed {len(original_blocks)} subtitle blocks.")
    
    # Step 3: Batch Processing
    # We split into batches of 50 to avoid hitting output token limits (usually 8192)
    BATCH_SIZE = 50
    batches = [original_blocks[i:i + BATCH_SIZE] for i in range(0, len(original_blocks), BATCH_SIZE)]
    print(f"Split into {len(batches)} batches for processing (Batch Size: {BATCH_SIZE}).")
    
    corrected_map = {}
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    temp_json_path = srt_path + ".temp_batch.json"

    for i, batch in enumerate(batches):
        print(f"\nProcessing Batch {i+1}/{len(batches)} ({len(batch)} items)...")
        
        # Prepare batch payload
        input_payload = [
            {'id': b['index'], 'text': b['text']} 
            for b in batch
        ]
        
        # Save batch to temp file
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(input_payload, f, indent=2, ensure_ascii=False)

        # Upload batch
        uploaded_file = None
        try:
            print("  Uploading batch...")
            uploaded_file = client.files.upload(
                file=temp_json_path,
                config=types.UploadFileConfig(mime_type="text/plain")
            )
        except Exception as e:
            print(f"  ❌ Error uploading batch {i+1}: {e}")
            continue

        # Wait for processing
        try:
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(0.5)
                uploaded_file = client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                print(f"  ❌ Batch {i+1} processing failed.")
                try: client.files.delete(name=uploaded_file.name)
                except: pass
                continue
        except Exception as e:
            print(f"  ❌ Error checking status for batch {i+1}: {e}")
            try: client.files.delete(name=uploaded_file.name)
            except: pass
            continue

        # Define Prompt
        topic_context = ""
        if webinar_topic:
            topic_context = f"\n        Context/Topic: The video is about '{webinar_topic}'. Use this context to better understand technical terms or context-specific language."

        prompt = f"""
        You are a professional transcription editor.
        The attached JSON file contains subtitle lines from a video.
        Language: {language}.{topic_context}
        
        Your task:
        1. Read the parsed JSON list.
        2. Correct spelling, grammar, and punctuation errors in the 'text' field.
        3. DO NOT change the 'id'.
        4. DO NOT change the number of items.
        5. Return the result as a valid JSON list.
        
        Output Format:
        [
          {{"id": "1", "text": "Corrected text here"}},
          {{"id": "2", "text": "More corrected text"}}
        ]
        """

        # Call Gemini
        print("  Requesting correction from Gemini (model: gemini-2.5-flash)...")
        response = None
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
                contents=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type
                    ),
                    prompt
                ]
            )
            
            # Cost tracking
            cost, in_tok, out_tok = calculate_gemini_cost(response)
            total_cost += cost
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            print(f"  ✓ Batch {i+1} Success. Cost: ${cost:.6f} | Tokens: {in_tok} in / {out_tok} out")

            # Parse Response
            batch_corrected = json.loads(response.text)
            for item in batch_corrected:
                corrected_map[item['id']] = item['text']
                
        except Exception as e:
            print(f"  ❌ Error in batch {i+1}: {e}")
            if response and hasattr(response, 'text'):
                 print(f"  Response text preview: {response.text[:200]}...")
        
        finally:
            # Cleanup
            if uploaded_file:
                try: client.files.delete(name=uploaded_file.name)
                except: pass

    # Cleanup temp file
    if os.path.exists(temp_json_path):
        os.remove(temp_json_path)

    print(f"\nTotal Session Cost: ${total_cost:.6f}")

    # Process Final Results
    correction_count = 0
    for block in original_blocks:
        if block['index'] in corrected_map:
            new_text = corrected_map[block['index']]
            if new_text != block['text']:
                block['text'] = new_text
                correction_count += 1
        else:
            # Warning only if we expected this batch to work?
            # Actually, silence is better here unless we want to spam warnings for failed batches
            pass
            
    print(f"✓ Applied corrections to {correction_count} blocks.")
    
    # Step 9: Save result
    base, ext = os.path.splitext(srt_path)
    output_path = f"{base}_corrected_by_gemini{ext}"
    
    write_srt(original_blocks, output_path)
    print(f"✓ Corrected SRT saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        start_time = time.time()
        srt_file = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) > 2 else "en"
        topic = sys.argv[3] if len(sys.argv) > 3 else None
        
        process_srt_correction(srt_file, lang, topic)
        
        elapsed_time = time.time() - start_time
        print(f"Total execution time: {format_ms_to_srt(elapsed_time * 1000)}")
    else:
        print("Usage: python correct_srt_errors.py <srt_file> [language] [webinar_topic]")
