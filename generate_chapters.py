#!/Users/artemreva/projects/whisper/venv/bin/python3
import os
import time
from dotenv import load_dotenv
from common_utils import get_api_key, calculate_gemini_cost, get_total_gemini_cost, format_ms_to_srt, safe_upload, retry_gemini_request

load_dotenv()
generate_chapters_model = "gemini-3-flash-preview"
generate_chapters_openrouter_model = "google/gemini-2.5-flash"

# Check if google.genai is available
try:
    import google.genai as genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as e:
    genai = None
    types = None
    ClientError = Exception


def _build_chapters_prompt(language=None, webinar_topic=None):
    lang_instruction = ""
    if language:
        lang_instruction = f"The input is in {language} language. Please generate the response in {language}."

    topic_instruction = ""
    if webinar_topic:
        topic_instruction = f"The topic of this webinar is: '{webinar_topic}'. Use this context to create more accurate and meaningful chapter titles."

    return f"""
    Analyze the provided SRT subtitles for this webinar/video.
    {topic_instruction}
    {lang_instruction}
    
    Your task is to create a list of timestamps (chapters) that summarize the entire content.
    
    1. Break down the content into logical chapters.
    2. For each chapter, provide the Start Time (HH:MM:SS) and a Concise Title/Summary.
    3. Ensure the chapters cover the flow of the entire video.
    
    Output format:
    00:00:00 - Introduction
    00:05:30 - Topic A description
    00:12:45 - Key takeaway about B
    ...
    
    Do not add any other conversational text, just the list of timecodes and titles.
    """


def _build_qa_timeline_prompt(language=None, webinar_topic=None):
    lang_instruction = ""
    if language:
        lang_instruction = f"The input is in {language} language. Please generate the response in {language}."

    topic_instruction = ""
    if webinar_topic:
        topic_instruction = f"The topic of this webinar is: '{webinar_topic}'. Use this context when identifying question-answer pairs and side topics."

    return f"""
    Analyze the provided SRT subtitles for this webinar/video.
    {topic_instruction}
    {lang_instruction}

    Your task is to produce a timed Question/Answer sequence for a Q/A webinar.
    Infer question and answer boundaries from the transcript text only.
    Do not rely on speaker labels.

    Rules:
    1. Extract meaningful Q/A pairs in chronological order.
    2. Timestamp each pair using the question start time in HH:MM:SS.
    3. Include a concise Question summary and Answer summary.
    4. If the answer contains side topics, include them; otherwise write "None".
    5. Keep each summary concise and useful for quick navigation.

    Output format (repeat this block for each pair):
    [00:00:00]
    Question: ...
    Answer: ...
    Side topics: topic 1; topic 2

    Do not add any extra commentary outside these Q/A blocks.
    """


def generate_chapters(srt_path, language=None, webinar_topic=None, output_mode="chapters", provider="gemini", model=None):
    """Upload SRT file to Gemini or send text to OpenRouter to get chapters or Q/A timeline."""
    provider = (provider or "gemini").strip().lower()
    model = model or (generate_chapters_openrouter_model if provider == "openrouter" else generate_chapters_model)

    if output_mode not in {"chapters", "qa_timeline"}:
        raise ValueError(f"Invalid output_mode: {output_mode}")
    
    # Determine output path early to check if it already exists
    suffix = "_chapters.txt" if output_mode == "chapters" else "_qa_timeline.txt"
    output_filename = os.path.splitext(srt_path)[0] + suffix
    if os.path.exists(output_filename):
        print(f"✓ AI response file already exists: {output_filename} (Skipping step)")
        return output_filename

    # Step 1: Prepare Content & Prompt
    print(f"Step 1: Preparing {output_mode} (Provider: {provider}, Model: {model})...")
    
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return None
        
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    if output_mode == "qa_timeline":
        prompt_builder = _build_qa_timeline_prompt
    else:
        prompt_builder = _build_chapters_prompt
        
    prompt_base = prompt_builder(language=language, webinar_topic=webinar_topic)

    response_text = ""
    if provider == "openrouter":
        print("Using OpenRouter (Text-based analysis)...")
        prompt = f"{prompt_base}\n\nInput SRT content:\n{srt_content}"
        from common_utils import generate_content
        try:
            response_text = generate_content(provider="openrouter", model=model, prompt=prompt)
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
        
        uploaded_file = None
        try:
            uploaded_file = safe_upload(client, srt_path, "text/plain")
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = retry_get_file(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                print("Error: File processing failed")
                return None
                
            response = retry_generate_content(
                model=model,
                contents=[
                    types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
                    prompt_base
                ]
            )
            
            from common_utils import calculate_gemini_cost
            cost, input_tokens, output_tokens = calculate_gemini_cost(response)
            print(f"✓ Response received from Gemini (Cost: ${cost:.6f} | Tokens: {input_tokens} in / {output_tokens} out)")
            response_text = response.text
            
        except Exception as e:
            print(f"\n❌ Error calling Gemini: {e}")
            return None
        finally:
            if uploaded_file:
                try: retry_delete_file(name=uploaded_file.name)
                except: pass

    # Step 6: Save response
    if not response_text or not response_text.strip():
        print("Error: AI response is empty")
        return None
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"✓ Timeline saved to: {output_filename}")
    
    print("\n=== Timeline generation completed successfully ===")
    return output_filename

if __name__ == "__main__":
    # Ask user for SRT file path
    srt_file = input("Enter the path to the Corrected SRT file: ").strip()
    language = input("Enter the language of the SRT file: ").strip()
    webinar_topic = input("Enter the topic of the webinar: ").strip()

    # Remove quotes
    if srt_file.startswith('"') and srt_file.endswith('"'):
        srt_file = srt_file[1:-1]
    elif srt_file.startswith("'") and srt_file.endswith("'"):
        srt_file = srt_file[1:-1]
    
    if not srt_file:
        print("Error: No file path provided.")
        exit(1)
    
    if os.path.exists(srt_file):
        start_time = time.time()
        print(f"\nGenerating chapters for: {srt_file}\n")
        output_file = generate_chapters(srt_file, language, webinar_topic)
        if output_file:
            print(f"\nOutput file: {output_file}")
            
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        formatted_time = format_ms_to_srt(execution_time_ms)
        total_cost = get_total_gemini_cost()

        print(f"Total execution time: {formatted_time}")
        print(f"Total cost: ${total_cost:.6f}")
    else:
        print(f"Error: File not found: {srt_file}")