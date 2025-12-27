#!/Users/artemreva/projects/whisper/venv/bin/python3
import os
import json
import time
from dotenv import load_dotenv
from common_utils import get_api_key

load_dotenv()

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



def process_srt_file(srt_path):
    """Upload SRT file to Gemini and get response, saving it as txt file."""
    
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
            print("This is a policy restriction from Google and cannot be bypassed.")
            print("\nPossible solutions:")
            print("  1. Use a VPN to connect from a supported region")
            print("  2. Use the API from a supported geographic location")
            print("  3. Check Google's Gemini API availability in your region")
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
    except ClientError as e:
        print(f"\n❌ Error during file processing: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error during file processing: {e}")
        return None
    
    print("✓ File processed successfully")
    
    # Step 4: Define the Prompt
    print("Step 4: Preparing prompt...")
    prompt = """
    Analyze the uploaded SRT subtitles:
    
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
    print("✓ Prompt prepared")
    
    # Step 5: Call Gemini 3 with uploaded file
    print("Step 5: Requesting analysis from Gemini 3 Flash model...")
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
        print("✓ Response received from Gemini")
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
    
    # Step 7: Clean up uploaded file
    print("Step 7: Cleaning up uploaded file...")
    try:
        client.files.delete(name=uploaded_file.name)
        print("✓ Uploaded file deleted from Gemini")
    except Exception as e:
        print(f"  Warning: Could not delete uploaded file: {e}")
    
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