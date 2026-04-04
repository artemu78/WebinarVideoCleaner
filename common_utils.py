import os
import time

def get_api_key(filepath="gemini_key.txt"):
    """
    Reads the Gemini API key.
    Priority:
    1. Environment variable GEMINI_API_KEY
    2. Local file (default: gemini_key.txt)
    3. User input (and optionally saves to .env)
    """
    # 1. Check environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key.strip()
    
    # 2. Check local file (legacy support)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
    
    # 3. Prompt user
    print(f"Gemini API key not found in environment or file: {filepath}")
    api_key = input("Please enter your Gemini API key: ").strip()
    
    if not api_key:
        print("Error: No API key provided.")
        exit(1)
    
    # Ask if user wants to save to .env
    save_env = input("Save to .env file? (y/n, default y): ").strip().lower()
    if save_env != 'n':
        try:
            with open(".env", "a") as f:
                f.write(f"\nGEMINI_API_KEY={api_key}\n")
            print("✓ API key saved to .env")
        except Exception as e:
            print(f"Warning: Could not save API key to .env: {e}")
            
    return api_key

def parse_time_to_ms(time_str):
    """
    Parses timestamp to milliseconds.
    Supports 'HH:MM:SS,mmm' (SRT) and 'HH:MM:SS' (Simple).
    """
    time_str = time_str.strip()
    # SRT format
    if ',' in time_str:
        hms, ms = time_str.split(',')
    elif '.' in time_str:
         hms, ms = time_str.split('.')
    else:
        hms = time_str
        ms = '000'

    parts = hms.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h = 0
        m, s = map(int, parts)
    else:
        return 0
    
    return (h * 3600 + m * 60 + s) * 1000 + int(ms)

def format_ms_to_srt(ms):
    """
    Formats milliseconds to 'HH:MM:SS,mmm'.
    """
    ms = int(ms)
    seconds = ms // 1000
    milliseconds = ms % 1000
    minutes = seconds // 60
    hours = minutes // 60
    
    seconds %= 60
    minutes %= 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def clean_srt_response(text):
    """
    Clean the response from Gemini to extract just the SRT content.
    """
    # Remove markdown code blocks
    text = text.replace("```srt", "").replace("```", "")
    return text.strip()

# Cost tracking
_TOTAL_GEMINI_COST = 0.0

def calculate_gemini_cost(response):
    """
    Calculates the cost of a Gemini request based on token usage.
    Pricing (Gemini 3 Flash preview):
    - Input: $0.50 / 1M tokens
    - Output: $3.00 / 1M tokens
    
    Updates the global total cost.
    Returns: Tuple (cost, input_tokens, output_tokens)
    """
    global _TOTAL_GEMINI_COST
    
    if not hasattr(response, 'usage_metadata'):
        return 0.0, 0, 0
        
    usage = response.usage_metadata
    
    # Check if usage is valid/populated
    if not usage:
        return 0.0, 0, 0
        
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    
    # Pricing
    input_price_per_million = 0.50
    output_price_per_million = 3.00
    
    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    
    total_request_cost = input_cost + output_cost
    
    _TOTAL_GEMINI_COST += total_request_cost
    
    return total_request_cost, input_tokens, output_tokens

def get_total_gemini_cost():
    """Returns the total accumulated Gemini cost."""
    global _TOTAL_GEMINI_COST
    return _TOTAL_GEMINI_COST

# Try to import google.genai for safe_upload typing/config
try:
    from google.genai import types
except ImportError:
    types = None

import shutil
import tempfile
import functools

def retry_gemini_request(func):
    """
    Decorator that catches Gemini API errors, asks the user to check 
    their connection, and retries the request upon confirmation.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"\n[Gemini API Error] {e}")
                print("Connection failed. Please check your internet connection and API status.")
                input("Press Enter to retry the request...")
                print("Retrying...")
    return wrapper

@retry_gemini_request
def safe_upload(client, file_path, mime_type):
    """
    Uploads a file to Gemini using a safe temporary ASCII filename to avoid Unicode errors.
    
    Args:
        client: The initialized Gemini client.
        file_path (str): The absolute path to the file to upload.
        mime_type (str): The MIME type of the file.
        
    Returns:
        The uploaded file object from Gemini.
    """
    if types is None:
        raise ImportError("google.genai package is required for safe_upload")

    ext = os.path.splitext(file_path)[1]
    # Create a temp file with the same extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="gemini_upload_") as tmp:
        temp_path = tmp.name
    
    try:
        # Copy original file content to temp file
        shutil.copy2(file_path, temp_path)
        
        # Upload the temp file
        print(f"  (Uploading safe copy: {temp_path})...")
        uploaded_file = client.files.upload(
            file=temp_path,
            config=types.UploadFileConfig(mime_type=mime_type)
        )
        return uploaded_file
    finally:
        # cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
