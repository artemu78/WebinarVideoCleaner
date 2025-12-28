#!/Users/artemreva/projects/whisper/venv/bin/python3
import os
import time
from dotenv import load_dotenv
from common_utils import get_api_key, calculate_gemini_cost, get_total_gemini_cost, format_ms_to_srt

load_dotenv()

# Check if google.genai is available
try:
    import google.genai as genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as e:
    print("Error: google-genai package not found.")
    print("Please run the script using one of these methods:")
    print("  1. ./generate_chapters.py")
    print("  2. venv/bin/python generate_chapters.py")
    print("  3. source venv/bin/activate && python generate_chapters.py")
    print("\nOr install the package: pip install google-genai")
    exit(1)



def generate_chapters(srt_path, language=None, webinar_topic=None):
    """Upload SRT file to Gemini and get chapters/timecodes."""
    
    # Step 1: Initialize Client
    print("Step 1: Initializing Gemini client...")
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    print("✓ Client initialized successfully")
    
    # Step 2: Upload SRT file
    print(f"Step 2: Uploading SRT file: {srt_path}...")
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return None
    
    try:
        uploaded_file = client.files.upload(
            file=srt_path,
            config=types.UploadFileConfig(mime_type="text/plain")
        )
        print(f"✓ File uploaded successfully. File URI: {uploaded_file.uri}")
    except ClientError as e:
        error_message = str(e)
        if "location is not supported" in error_message.lower() or "FAILED_PRECONDITION" in error_message:
            print("\n❌ Error: Geographic restriction")
            print("The Gemini API is not available in your current location.")
        else:
            print(f"\n❌ Error uploading file: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error uploading file: {e}")
        return None
    
    # Step 3: Wait for file to be processed
    print("Step 3: Waiting for file to be processed...")
    try:
        while uploaded_file.state.name == "PROCESSING":
            print("  File is still processing, waiting...")
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            print("Error: File processing failed")
            return None
    except Exception as e:
        print(f"\n❌ Error during file processing: {e}")
        return None
    
    print("✓ File processed successfully")
    
    # Step 4: Define the Prompt
    print("Step 4: Preparing prompt for chapters...")
    lang_instruction = ""
    if language:
        # Map common codes to full names if needed, or just use the code as is.
        # Gemini understands codes.
        lang_instruction = f"The input is in {language} language. Please generate the response in {language}."

    topic_instruction = ""
    if webinar_topic:
        topic_instruction = f"The topic of this webinar is: '{webinar_topic}'. Use this context to create more accurate and meaningful chapter titles."

    prompt = f"""
    Analyze the uploaded SRT subtitles for this webinar/video.
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
    
    # Step 5: Call Gemini 3 with uploaded file
    print("Step 5: Requesting chapters from Gemini 3 Flash model...")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type
                ),
                prompt
            ]
        )
        # Calculate and print cost
        cost, input_tokens, output_tokens = calculate_gemini_cost(response)
        print(f"✓ Response received from Gemini (Cost: ${cost:.6f} | Tokens: {input_tokens} in / {output_tokens} out)")
    except Exception as e:
        print(f"\n❌ Error generating content: {e}")
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        return None
    
    # Step 6: Save response
    print("Step 6: Saving chapters to text file...")
    
    if not hasattr(response, 'text') or not response.text:
        print("Error: Gemini response has no text content")
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        return None
    
    # Determine output filename: same basename as SRT but with _chapters.txt
    output_filename = os.path.splitext(srt_path)[0] + "_chapters.txt"
    # If the input was already _corrected.srt, it will become _corrected_chapters.txt
    # If we want it cleaner, we could strip _corrected, but keeping it is explicit.
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"✓ Chapters saved to: {output_filename}")
    
    # Step 7: Clean up uploaded file
    print("Step 7: Cleaning up uploaded file...")
    try:
        client.files.delete(name=uploaded_file.name)
        print("✓ Uploaded file deleted from Gemini")
    except Exception as e:
        print(f"  Warning: Could not delete uploaded file: {e}")
    
    print("\n=== Chapter generation completed successfully ===")
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
