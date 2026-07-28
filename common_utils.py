import builtins
import contextlib
import os
import time
import json
import urllib.request
import re
import shutil
import tempfile
import functools

# Try to import google.genai for safe_upload typing/config
try:
    import google.genai as genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# When enabled, `timed_input` accumulates wall time spent waiting on the user (for execution metrics).
_track_user_input_time = False
_tracked_user_input_seconds = 0.0


def track_user_input_time_begin():
    """Start excluding user typing time from measured script execution (resets the accumulator)."""
    global _track_user_input_time, _tracked_user_input_seconds
    _tracked_user_input_seconds = 0.0
    _track_user_input_time = True


def track_user_input_time_end():
    """Stop accumulating user-input time (does not clear the last total; call get_tracked_user_input_seconds first if needed)."""
    global _track_user_input_time
    _track_user_input_time = False


def get_tracked_user_input_seconds():
    """Seconds spent inside `timed_input` since the last `track_user_input_time_begin`."""
    return _tracked_user_input_seconds


@contextlib.contextmanager
def track_user_input_time_scope():
    """Begin tracking user input time for this block; always ends tracking on exit."""
    track_user_input_time_begin()
    try:
        yield
    finally:
        track_user_input_time_end()


def timed_input(prompt=""):
    """Like built-in input; when tracking is on, adds wait time to the user-input accumulator."""
    t0 = time.perf_counter()
    try:
        return builtins.input(prompt)
    finally:
        if _track_user_input_time:
            global _tracked_user_input_seconds
            _tracked_user_input_seconds += time.perf_counter() - t0

def get_gemini_api_key(filepath="gemini_key.txt"):
    """
    Reads the Gemini API key.
    Priority:
    1. Environment variable GEMINI_API_KEY (or GOOGLE_API_KEY)
    2. Local file (default: gemini_key.txt)
    3. User input (and optionally saves to .env)
    """
    # 1. Check environment variables
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key.strip()
    
    # 2. Check local file (legacy support)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read().strip()
    
    # 3. Prompt user
    print(f"Gemini API key not found in environment or file: {filepath}")
    api_key = timed_input("Please enter your Gemini API key: ").strip()
    
    if not api_key:
        print("Error: No API key provided.")
        exit(1)
    
    # Ask if user wants to save to .env
    save_env = timed_input("Save to .env file? (y/n, default y): ").strip().lower()
    if save_env != 'n':
        try:
            with open(".env", "a") as f:
                f.write(f"\nGEMINI_API_KEY={api_key}\n")
            print("✓ API key saved to .env")
        except Exception as e:
            print(f"Warning: Could not save API key to .env: {e}")
            
    return api_key

# Alias for backward compatibility
get_api_key = get_gemini_api_key

def get_openrouter_api_key(filepath="openrouter_key.txt"):
    """
    Reads the OpenRouter API key.
    Priority:
    1. Environment variable OPENROUTER_API_KEY
    2. Local file (default: openrouter_key.txt)
    3. User input (and optionally saves to .env)
    """
    # 1. Check environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key.strip()

    # 2. Check local file
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    # 3. Prompt user
    print(f"OpenRouter API key not found in environment or file: {filepath}")
    api_key = timed_input("Please enter your OpenRouter API key: ").strip()
    
    if not api_key:
        print("Error: No OpenRouter API key provided.")
        exit(1)
    
    # Ask if user wants to save to .env
    save_env = timed_input("Save to .env file? (y/n, default y): ").strip().lower()
    if save_env != 'n':
        try:
            with open(".env", "a") as f:
                f.write(f"\nOPENROUTER_API_KEY={api_key}\n")
            print("✓ API key saved to .env")
        except Exception as e:
            print(f"Warning: Could not save API key to .env: {e}")
            
    return api_key

# Alias for internal use if needed
_get_openrouter_api_key = get_openrouter_api_key

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

def parse_srt_to_blocks(content):
    """
    Parses SRT content into a list of dictionaries.
    Each dict: {'index': str, 'start': str, 'end': str, 'text': str}
    """
    # Split by double newlines to get blocks (handling potential CRLF and varying whitespace)
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
                if ' --> ' in timestamps:
                    start, end = timestamps.split(' --> ')
                    parsed_blocks.append({
                        'index': index,
                        'start': start,
                        'end': end,
                        'text': text
                    })
            except ValueError:
                continue
                
    return parsed_blocks

def blocks_to_srt_str(blocks):
    """
    Writes a list of block dictionaries to an SRT string.
    """
    lines = []
    for block in blocks:
        lines.append(f"{block['index']}")
        lines.append(f"{block['start']} --> {block['end']}")
        lines.append(f"{block['text']}")
        lines.append("")
    return "\n".join(lines)

def parse_json_array(text):
    """
    Robustly parse a JSON list from an LLM response, tolerating markdown and extra text.
    """
    t = (text or "").strip()
    # Remove markdown fences
    for prefix in ("```json", "```JSON", "```"):
        if t.startswith(prefix):
            t = t[len(prefix) :].lstrip()
            break
    if "```" in t:
        t = t.split("```")[0].strip()
    
    # Find the JSON array part
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        t = t[start : end + 1]
    
    try:
        return json.loads(t)
    except Exception:
        # If it's not a valid JSON array, return None so the caller can handle it
        return None

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
                print(f"\n[AI API Error] {e}")
                print("Connection failed. Please check your internet connection and API status.")
                timed_input("Press Enter to retry the request...")
                print("Retrying...")
    return wrapper


def retry_openrouter_request(func, attempts=3, initial_delay_seconds=1.0):
    """Retry OpenRouter transport failures without blocking for terminal input."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as error:
            last_error = error
            if attempt == attempts:
                break

            delay = initial_delay_seconds * (2 ** (attempt - 1))
            print(
                f"[OpenRouter request failed] Attempt {attempt}/{attempts}: {error}. "
                f"Retrying automatically in {delay:.0f} second(s)..."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"OpenRouter request failed after {attempts} attempts: {last_error}"
    ) from last_error

def safe_upload(client, file_path, mime_type):
    """
    Uploads a file to Gemini using a safe temporary ASCII filename to avoid Unicode errors.
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
        retry_upload = retry_gemini_request(client.files.upload)
        uploaded_file = retry_upload(
            file=temp_path,
            config=types.UploadFileConfig(mime_type=mime_type)
        )
        return uploaded_file
    finally:
        # cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def generate_content(provider, model, prompt, response_mime_type=None, temperature=0.1):
    """
    Generate text content via configured provider.

    Args:
        provider (str): "gemini" or "openrouter"
        model (str): model identifier for the selected provider
        prompt (str): text prompt
        response_mime_type (str): Optional MIME type for response (Gemini only)
        temperature (float): Sampling temperature

    Returns:
        str: model response text
    """
    provider_key = (provider or "gemini").strip().lower()

    if provider_key == "gemini":
        if genai is None:
            raise ImportError("google-genai package not found. Install with `pip install google-genai`.")
        api_key = get_gemini_api_key()
        client = genai.Client(api_key=api_key)
        
        config = None
        if response_mime_type or temperature != 0.1:
            config = types.GenerateContentConfig(
                response_mime_type=response_mime_type,
                temperature=temperature
            )
            
        retry_gen = retry_gemini_request(client.models.generate_content)
        response = retry_gen(model=model, contents=prompt, config=config)
        
        # Track cost
        calculate_gemini_cost(response)
        
        return (getattr(response, "text", "") or "").strip()

    if provider_key == "openrouter":
        api_key = get_openrouter_api_key()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        inference_provider = os.environ.get("OPENROUTER_INFERENCE_PROVIDER", "").strip()
        if inference_provider:
            payload["provider"] = {"only": [inference_provider]}
        
        request = urllib.request.Request(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost",
                "X-Title": "VideoCleaner",
            },
            method="POST",
        )
        
        def _make_request():
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        # Retry transport failures only. A completed response with no usable final
        # message will not become valid by submitting the same prompt again.
        body = retry_openrouter_request(_make_request)

        if not isinstance(body, dict):
            raise RuntimeError("OpenRouter returned an invalid response body.")
        if "error" in body:
            raise RuntimeError(f"OpenRouter API Error: {body['error']}")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter returned no completion choices.")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("OpenRouter returned an invalid completion choice.")

        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            finish_reason = choice.get("finish_reason", "unknown")
            response_model = body.get("model", model)
            raise RuntimeError(
                "OpenRouter returned no final message content "
                f"(model={response_model!r}, finish_reason={finish_reason!r}). "
                "The model or provider may have returned reasoning only or rejected the request."
            )

        return content.strip()

    raise ValueError(f"Unsupported provider: {provider}")
