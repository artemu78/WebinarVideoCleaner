#!/usr/bin/env python3
import os
import time
from dotenv import load_dotenv

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

def get_api_key(filepath="gemini_key.txt"):
    """
    Reads the Gemini API key.
    Priority:
    1. Environment variable GEMINI_API_KEY
    2. Local file (default: gemini_key.txt)
    3. User input
    """
    # 1. Check environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key.strip()
    
    # 2. Check local file (legacy support)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
    
    # 3. Prompt user (only if running directly, not when imported usually)
    # But since this is a utility script, we might not want to prompt if imported.
    # We'll assume the main script handles env setup or we prompt if missing.
    print(f"Gemini API key not found in environment or file: {filepath}")
    api_key = input("Please enter your Gemini API key: ").strip()
    
    if not api_key:
        print("Error: No API key provided.")
        exit(1)
            
    return api_key

def clean_srt_response(text):
    """
    Clean the response from Gemini to extract just the SRT content.
    """
    # Remove markdown code blocks
    text = text.replace("```srt", "").replace("```", "")
    return text.strip()

def process_srt_correction(srt_path, language="en"):
    """
    Upload SRT file to Gemini and ask for transcription corrections.
    Returns path to the corrected SRT file.
    """
    
    # Step 1: Initialize Client
    print(f"Initializing Gemini client for SRT correction (Language: {language})...")
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    
    # Step 2: Upload SRT file
    print(f"Uploading SRT file: {srt_path}...")
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return None
    
    try:
        uploaded_file = client.files.upload(
            file=srt_path,
            config=types.UploadFileConfig(mime_type="text/plain")
        )
        print(f"✓ File uploaded successfully. URI: {uploaded_file.uri}")
    except Exception as e:
        print(f"\n❌ Error uploading file: {e}")
        return None
    
    # Step 3: Wait for processing
    print("Waiting for file processing...")
    try:
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            print("Error: File processing failed")
            return None
    except Exception as e:
        print(f"\n❌ Error checking file status: {e}")
        return None
    
    # Step 4: Define Prompt
    prompt = f"""
    You are an expert transcriptionist. 
    The attached file is an SRT subtitle file derived from a video.
    The detected language is: {language}.
    
    Your task is to correct TRANSCRIPTION ERRORS in the text.
    
    Rules:
    1. Only correct spelling mistakes, wrong words, and grammar errors that are clearly transcription mistakes.
    2. Do NOT change the timing (timestamps) of the subtitles.
    3. Do NOT merge or split subtitles. Keep the structure exactly the same.
    4. Only make changes if you are VERY SURE it is a mistake.
    5. Output the result clearly as a valid SRT file.
    6. Do not include any explanation, only the SRT content.
    """
    
    # Step 5: Call Gemini
    print("Requesting transcription correction from Gemini (model: gemini-3-flash-preview)...")
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type
                ),
                prompt
            ]
        )
    except Exception as e:
        print(f"\n❌ Error generating content: {e}")
        # Cleanup
        try: client.files.delete(name=uploaded_file.name)
        except: pass
        return None
    
    # Step 6: Save response
    if not response.text or not response.text.strip():
        print("Error: Empty response from Gemini")
        # Cleanup
        try: client.files.delete(name=uploaded_file.name)
        except: pass
        return None
        
    cleaned_srt_content = clean_srt_response(response.text)
    
    # Create new filename
    base, ext = os.path.splitext(srt_path)
    output_path = f"{base}_corrected_transcription{ext}"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_srt_content)
    
    print(f"✓ Corrected SRT saved to: {output_path}")
    
    # Step 7: Cleanup
    try:
        client.files.delete(name=uploaded_file.name)
    except:
        pass
        
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        srt_file = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) > 2 else "en"
        process_srt_correction(srt_file, lang)
    else:
        print("Usage: python correct_transcription.py <srt_file> [language]")
