#!/Users/artemreva/projects/whisper/venv/bin/python3
import os
import time
from dotenv import load_dotenv
from common_utils import get_api_key, calculate_gemini_cost

load_dotenv()
audio_cleaner_model = "gemini-3-flash-preview"
# Check if google.genai is available
try:
    import google.genai as genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as e:
    print("Error: google-genai package not found.")
    print("Please run the script using one of these methods:")
    print("  1. ./audio_cleaner.py")
    print("  2. venv/bin/python audio_cleaner.py")
    print("  3. source venv/bin/activate && python audio_cleaner.py")
    print("\nOr install the package: pip install google-genai")
    exit(1)



def process_srt_file(srt_path, audio_path=None):
    """Upload SRT file (and optional audio) to Gemini and get response, saving it as txt file."""
    
    # Step 1: Initialize Client
    print("Step 1: Initializing Gemini client...")
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    print("✓ Client initialized successfully")
    
    # Step 2: Upload SRT file
    print(f"Step 2: Uploading SRT file: {srt_path}...")
    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found: {srt_path}")
        return
    
    try:
        uploaded_srt = client.files.upload(
            file=srt_path,
            config=types.UploadFileConfig(mime_type="text/plain")
        )
        print(f"✓ SRT file uploaded successfully. File URI: {uploaded_srt.uri}")
    except ClientError as e:
        error_message = str(e)
        if "location is not supported" in error_message.lower() or "FAILED_PRECONDITION" in error_message:
            print("\n❌ Error: Geographic restriction")
            print("The Gemini API is not available in your current location.")
            print("This is a policy restriction from Google and cannot be bypassed.")
            print("\nPossible solutions:")
            print("  1. Use a VPN to connect from a supported region")
            print("  2. Use the API from a supported geographic location")
            print("  3. Check Google's Gemini API availability in your region")
        else:
            print(f"\n❌ Error uploading SRT file: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error uploading SRT file: {e}")
        return None

    # Step 2b: Upload Audio file (if provided)
    uploaded_audio = None
    if audio_path:
        print(f"Step 2b: Uploading Audio file: {audio_path}...")
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}. Proceeding with SRT only.")
        else:
            try:
                # Determine mime type based on extension
                mime_type = "audio/mpeg"  # default
                if audio_path.lower().endswith(".mp3"):
                    mime_type = "audio/mp3"
                elif audio_path.lower().endswith(".wav"):
                    mime_type = "audio/wav"
                elif audio_path.lower().endswith(".aac"):
                    mime_type = "audio/aac"
                # Add more if needed or rely on default

                uploaded_audio = client.files.upload(
                    file=audio_path,
                    config=types.UploadFileConfig(mime_type=mime_type)
                )
                print(f"✓ Audio file uploaded successfully. File URI: {uploaded_audio.uri}")
            except Exception as e:
                print(f"\n❌ Error uploading Audio file: {e}. Proceeding with SRT only.")
                uploaded_audio = None
    
    # Step 3: Wait for files to be processed
    print("Step 3: Waiting for files to be processed...")
    files_to_wait = [uploaded_srt]
    if uploaded_audio:
        files_to_wait.append(uploaded_audio)

    for f_obj in files_to_wait:
        try:
            while f_obj.state.name == "PROCESSING":
                print(f"  File {f_obj.name} is still processing, waiting...")
                time.sleep(2)
                f_obj = client.files.get(name=f_obj.name)
            
            if f_obj.state.name == "FAILED":
                print(f"Error: File processing failed for {f_obj.name}")
                return None
        except ClientError as e:
            print(f"\n❌ Error during file processing: {e}")
            return None
        except Exception as e:
            print(f"\n❌ Unexpected error during file processing: {e}")
            return None
    
    print("✓ Files processed successfully")
    
    # Step 4: Define the Prompt
    print("Step 4: Preparing prompt...")
    
    base_prompt = """
    Analyze the uploaded SRT subtitles{audio_clause}:
    
    Identify all ranges that should be removed. 
    Focus on:
    1. Long silences (over 2 seconds).
    2. Sections with filler words (uh, um) not captured in SRT.
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
    
    audio_clause = ""
    if uploaded_audio:
        audio_clause = " AND the audio file. Use the audio to confirm silences, identify non-verbal cues, and filler words not present in the text"
    
    prompt = base_prompt.format(audio_clause=audio_clause)
    print("✓ Prompt prepared")
    
    # Step 5: Call Gemini with uploaded files
    print(f"Step 5: Requesting analysis from {audio_cleaner_model} model...")
    content_parts = []
    
    # Add SRT
    content_parts.append(types.Part.from_uri(
        file_uri=uploaded_srt.uri,
        mime_type=uploaded_srt.mime_type
    ))
    
    # Add Audio if available
    if uploaded_audio:
        content_parts.append(types.Part.from_uri(
            file_uri=uploaded_audio.uri,
            mime_type=uploaded_audio.mime_type
        ))
        
    # Add Prompt
    content_parts.append(prompt)

    try:
        response = client.models.generate_content(
            model=audio_cleaner_model,
            contents=content_parts
        )
        # Calculate and print cost
        cost, input_tokens, output_tokens = calculate_gemini_cost(response)
        print(f"✓ Response received from Gemini (Cost: ${cost:.6f} | Tokens: {input_tokens} in / {output_tokens} out)")
    except ClientError as e:
        print(f"\n❌ Error generating content: {e}")
        # Try to clean up the uploaded file even if generation failed
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error generating content: {e}")
        # Try to clean up the uploaded file even if generation failed
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        return None
    
    # Step 6: Save response as txt file
    print("Step 6: Saving response to text file...")
    
    # Validate that response.text exists and is not empty
    if not hasattr(response, 'text') or response.text is None:
        print("Error: Gemini response has no text content")
        # Try to clean up the uploaded file
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        return None
    
    if not response.text.strip():
        print("Warning: Gemini response text is empty")
        # Try to clean up the uploaded file
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
        return None
    
    output_filename = os.path.splitext(srt_path)[0] + "_gemini_response.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"✓ Response saved to: {output_filename}")
    
    # Step 7: Clean up uploaded files
    print("Step 7: Cleaning up uploaded files...")
    
    files_to_delete = [uploaded_srt]
    if uploaded_audio:
        files_to_delete.append(uploaded_audio)

    for f_obj in files_to_delete:
        try:
            client.files.delete(name=f_obj.name)
            print(f"✓ Uploaded file {f_obj.name} deleted from Gemini")
        except Exception as e:
            print(f"  Warning: Could not delete uploaded file {f_obj.name}: {e}")
    
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