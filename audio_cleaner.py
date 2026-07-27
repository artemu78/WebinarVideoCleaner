#!/Users/artemreva/projects/whisper/venv/bin/python3
import os
import time
from dotenv import load_dotenv
from common_utils import get_api_key, calculate_gemini_cost, safe_upload, retry_gemini_request

load_dotenv()
audio_cleaner_model = "gemini-3-flash-preview"
audio_cleaner_openrouter_model = "google/gemini-2.5-flash"

# Check if google.genai is available
try:
    import google.genai as genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as e:
    genai = None
    types = None
    ClientError = Exception

def process_srt_file(srt_path, audio_path=None, keep_fillers=False, provider="gemini", model=None):
    """Upload SRT file (and optional audio) to Gemini or send text to OpenRouter."""
    provider = (provider or "gemini").strip().lower()
    model = model or (audio_cleaner_openrouter_model if provider == "openrouter" else audio_cleaner_model)
    
    # Determine output path early to check if it already exists
    output_filename = os.path.splitext(srt_path)[0] + "_gemini_response.txt"
    if os.path.exists(output_filename):
        print(f"✓ AI response file already exists: {output_filename} (Skipping step)")
        return output_filename

    # Step 1: Prepare Content & Prompt
    print(f"Step 1: Preparing analysis (Provider: {provider}, Model: {model}, Keep Fillers: {keep_fillers})...")
    
    filler_cleaning_instruction = "2. Sections with filler words (uh, um) not captured in SRT."
    if keep_fillers:
        filler_cleaning_instruction = "2. [DISABLED] DO NOT identify or remove sections with filler words."

    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return None
        
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    if provider == "openrouter":
        print("Using OpenRouter (Text-based SRT analysis)...")
        prompt = f"""
        Analyze the following SRT subtitles:
        
        {srt_content}

        Identify all ranges that should be removed. 
        Focus on:
        1. Long silences (over 2 seconds).
        {filler_cleaning_instruction}
        3. Errors or repeated takes.

        Return ONLY a JSON object with a list of ranges to delete.
        Example format:
        {{
          "ranges_to_delete": [
            {{"start": "00:00:05,000", "end": "00:00:08,500", "reason": "silence"}},
            {{"start": "00:01:12,200", "end": "00:01:15,000", "reason": "filler words"}}
          ]
        }}
        """
        from common_utils import generate_content
        try:
            response_text = generate_content(provider="openrouter", model=model, prompt=prompt)
            if not response_text:
                print("Error: Empty response from OpenRouter")
                return None
        except Exception as e:
            print(f"\n❌ Error calling OpenRouter: {e}")
            return None
            
    else:
        # Gemini Path (File Uploads)
        print("Using Google Gemini (File-based analysis)...")
        api_key = get_api_key()
        client = genai.Client(api_key=api_key)
        
        # Define wrapped methods for retries
        retry_generate_content = retry_gemini_request(client.models.generate_content)
        retry_get_file = retry_gemini_request(client.files.get)
        retry_delete_file = retry_gemini_request(client.files.delete)
        
        try:
            uploaded_srt = safe_upload(client, srt_path, "text/plain")
            print(f"✓ SRT file uploaded successfully. File URI: {uploaded_srt.uri}")
        except Exception as e:
            print(f"\n❌ Error uploading SRT file: {e}")
            return None

        # Upload Audio if provided
        uploaded_audio = None
        if audio_path:
            print(f"Step 2b: Uploading Audio file: {audio_path}...")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}. Proceeding with SRT only.")
            else:
                try:
                    mime_type = "audio/mpeg"
                    if audio_path.lower().endswith(".mp3"): mime_type = "audio/mp3"
                    elif audio_path.lower().endswith(".wav"): mime_type = "audio/wav"
                    uploaded_audio = safe_upload(client, audio_path, mime_type)
                    print(f"✓ Audio file uploaded successfully. File URI: {uploaded_audio.uri}")
                except Exception as e:
                    print(f"\n❌ Error uploading Audio file: {e}. Proceeding with SRT only.")
                    uploaded_audio = None
        
        # Wait for processing
        print("Waiting for Gemini to process files...")
        files_to_wait = [uploaded_srt]
        if uploaded_audio: files_to_wait.append(uploaded_audio)

        for f_obj in files_to_wait:
            while f_obj.state.name == "PROCESSING":
                time.sleep(2)
                f_obj = retry_get_file(name=f_obj.name)
            if f_obj.state.name == "FAILED":
                print(f"Error: File processing failed for {f_obj.name}")
                return None
        
        audio_clause = " AND the audio file. Use the audio to confirm silences, identify non-verbal cues, and filler words not present in the text" if uploaded_audio else ""
        prompt = f"""
        Analyze the uploaded SRT subtitles{audio_clause}:
        
        Identify all ranges that should be removed. 
        Focus on:
        1. Long silences (over 2 seconds).
        {filler_cleaning_instruction}
        3. Errors or repeated takes.

        Return ONLY a JSON object with a list of ranges to delete.
        Example format:
        {{
          "ranges_to_delete": [
            {{"start": "00:00:05,000", "end": "00:00:08,500", "reason": "silence"}},
            {{"start": "00:01:12,200", "end": "00:01:15,000", "reason": "filler words"}}
          ]
        }}
        """
        
        content_parts = []
        content_parts.append(types.Part.from_uri(file_uri=uploaded_srt.uri, mime_type=uploaded_srt.mime_type))
        if uploaded_audio:
            content_parts.append(types.Part.from_uri(file_uri=uploaded_audio.uri, mime_type=uploaded_audio.mime_type))
        content_parts.append(prompt)

        try:
            response = retry_generate_content(model=model, contents=content_parts)
            from common_utils import calculate_gemini_cost
            cost, input_tokens, output_tokens = calculate_gemini_cost(response)
            print(f"✓ Response received from Gemini (Cost: ${cost:.6f} | Tokens: {input_tokens} in / {output_tokens} out)")
            response_text = response.text
        except Exception as e:
            print(f"\n❌ Error generating content: {e}")
            return None
        finally:
            # Clean up
            try:
                retry_delete_file(name=uploaded_srt.name)
                if uploaded_audio: retry_delete_file(name=uploaded_audio.name)
            except: pass

    # Step 3: Save response as txt file
    if not response_text or not response_text.strip():
        print("Error: AI response is empty")
        return None
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"✓ Response saved to: {output_filename}")
    
    print("\n=== Process completed successfully ===")
    return output_filename


if __name__ == "__main__":
    # Ask user for SRT file path
    srt_file = input("Enter the path to the SRT file: ").strip()
    
    # Remove quotes if user pasted a path with quotes
    if srt_file.startswith('"') and srt_file.endswith('"'):
        srt_file = srt_file[1:-1]
    elif srt_file.startswith("'") and srt_file.endswith("'"):
        srt_file = srt_file[1:-1]
    
    if not srt_file:
        print("Error: No file path provided.")
        exit(1)
    
    if os.path.exists(srt_file):
        print(f"\nProcessing SRT file: {srt_file}\n")
        output_file = process_srt_file(srt_file)
        if output_file:
            print(f"\nOutput file: {output_file}")
    else:
        print(f"Error: SRT file not found: {srt_file}")
        print("Please ensure the SRT file exists.")