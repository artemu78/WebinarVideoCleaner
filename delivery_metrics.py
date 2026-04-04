#!/usr/bin/env python3
import os
import time
import json
import re
from dotenv import load_dotenv
from common_utils import get_api_key, calculate_gemini_cost, get_total_gemini_cost, format_ms_to_srt, safe_upload, parse_time_to_ms, retry_gemini_request

load_dotenv()
delivery_metrics_model = "gemini-3-flash-preview"

# Check if google.genai is available
try:
    import google.genai as genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError as e:
    print("Error: google-genai package not found.")
    print("Please install the package: pip install google-genai")
    exit(1)

def parse_srt_for_metrics(filepath):
    """
    Parses an SRT file into a list of dictionaries with timestamps in ms.
    Each dict: {'index': str, 'start': int, 'end': int, 'text': str}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    
    parsed_blocks = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0].strip()
            timestamps = lines[1].strip()
            text = "\n".join(lines[2:])
            
            try:
                start_str, end_str = timestamps.split(' --> ')
                start_ms = parse_time_to_ms(start_str)
                end_ms = parse_time_to_ms(end_str)
                parsed_blocks.append({
                    'index': index,
                    'start': start_ms,
                    'end': end_ms,
                    'text': text
                })
            except ValueError:
                continue
                
    return parsed_blocks

def calculate_manual_metrics(srt_blocks):
    """
    Calculates metrics that can be derived directly from the SRT.
    - Dead Air Index (Gaps > 3 seconds)
    - Pace (WPM)
    """
    if not srt_blocks:
        return {
            'total_duration_ms': 0,
            'dead_air_count': 0,
            'total_dead_air_ms': 0,
            'dead_air_percentage': "0.0%",
            'average_wpm': "0.0",
            'dead_air_intervals': []
        }

    total_duration_ms = srt_blocks[-1]['end'] - srt_blocks[0]['start']
    
    dead_air_threshold_ms = 3000
    dead_air_intervals = []
    
    for i in range(len(srt_blocks) - 1):
        gap = srt_blocks[i+1]['start'] - srt_blocks[i]['end']
        if gap > dead_air_threshold_ms:
            dead_air_intervals.append({
                'start': srt_blocks[i]['end'],
                'end': srt_blocks[i+1]['start'],
                'duration': gap
            })
            
    total_dead_air_ms = sum(gap['duration'] for gap in dead_air_intervals)
    dead_air_percentage = (total_dead_air_ms / total_duration_ms) * 100 if total_duration_ms > 0 else 0
    
    # Pace calculation (Words Per Minute)
    total_words = sum(len(block['text'].split()) for block in srt_blocks)
    speaking_duration_ms = total_duration_ms - total_dead_air_ms
    wpm = (total_words / (speaking_duration_ms / 60000)) if speaking_duration_ms > 0 else 0
    
    return {
        'total_duration_ms': total_duration_ms,
        'dead_air_count': len(dead_air_intervals),
        'total_dead_air_ms': total_dead_air_ms,
        'dead_air_percentage': f"{dead_air_percentage:.1f}%",
        'average_wpm': f"{wpm:.1f}",
        'dead_air_intervals': dead_air_intervals
    }

def read_chapters(chapters_path):
    """Reads chapters from the generated chapters file."""
    if not chapters_path or not os.path.exists(chapters_path):
        return "No chapters provided."
    
    with open(chapters_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_delivery_metrics(srt_path, chapters_path, language="en", webinar_topic=None):
    """
    Calculates delivery metrics using a combination of manual processing
    and Gemini analysis.
    """
    # Determine output path early to check if it already exists
    base_path = os.path.splitext(srt_path)[0]
    output_path = f"{base_path}_delivery_metrics.html"
    if os.path.exists(output_path):
        print(f"✓ Delivery metrics report already exists: {output_path} (Skipping step)")
        return output_path

    print(f"Calculating delivery metrics for: {srt_path}")

    # Step 1: Initialize Client
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    
    # Define wrapped methods for retries
    retry_generate_content = retry_gemini_request(client.models.generate_content)
    retry_get_file = retry_gemini_request(client.files.get)
    retry_delete_file = retry_gemini_request(client.files.delete)
    
    # Step 2: Manual Metrics
    srt_blocks = parse_srt_for_metrics(srt_path)
    manual_metrics = calculate_manual_metrics(srt_blocks)
    
    # Step 3: Upload SRT to Gemini for deeper analysis
    print(f"Uploading SRT file for linguistic and speaker analysis...")
    uploaded_file = None
    try:
        uploaded_file = safe_upload(client, srt_path, "text/plain")
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = retry_get_file(name=uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            print("Error: File processing failed")
            return None
    except Exception as e:
        print(f"Error uploading SRT: {e}")
        return None

    # Step 4: Define Prompt
    chapters_content = read_chapters(chapters_path)
    topic_context = f"Webinar Topic: {webinar_topic}\n" if webinar_topic else ""
    
    # We combine manual metrics to give Gemini more context
    dead_air_intervals_str = ""
    if manual_metrics['dead_air_intervals']:
        dead_air_intervals_str = "\n    Long Silence Intervals:\n"
        for interval in manual_metrics['dead_air_intervals'][:10]:
            dead_air_intervals_str += f"    - {format_ms_to_srt(interval['start'])} to {format_ms_to_srt(interval['end'])} ({interval['duration']/1000:.1f}s)\n"

    manual_context = f"""
    Manual analysis results:
    - Total Duration: {format_ms_to_srt(manual_metrics['total_duration_ms'])}
    - Average WPM: {manual_metrics['average_wpm']}
    - Dead Air Percentage: {manual_metrics['dead_air_percentage']}
    - Dead Air Count: {manual_metrics['dead_air_count']}
    - Total Dead Air: {format_ms_to_srt(manual_metrics['total_dead_air_ms'])}
    {dead_air_intervals_str}
    """

    prompt = f"""
    You are an expert educational consultant and speech coach.
    Analyze the attached SRT subtitles and the provided chapters for a webinar.
    
    IMPORTANT: Write the entire report in the following language: {language}.
    
    {topic_context}
    {manual_context}
    
    Chapters:
    {chapters_content}
    
    Evaluate the following metrics based on the transcript and context:
    
    1. Student-to-Teacher Talk Ratio: 
       Identify speaker turns. Estimate the percentage of time the teacher is speaking vs students/audience. Identify the frequency and duration of student interruptions or questions.
    
    2. Pace Variability: 
       Identify where the pace spikes (rushing) or slows down significantly. Correlate these with topics.
    
    3. Clarity vs. Jargon Density: 
       Measure the ratio of specialized terms to plain language. List specific technical terms that were not explained well.
    
    4. Sentiment & Energy Tracking: 
       Infer the vocal profile and energy levels from the text, punctuation, and pacing. Identify "flat" moments vs "tonal spikes" when new concepts are introduced.
    
    5. Cognitive Load Indicators: 
       Identify "rephrase loops" where the same concept is explained multiple times consecutively. 
       CRITICAL: Provide the exact video timing (e.g., [00:12:34]) for each identified rephrase loop.
    
    6. Structural Alignment & Tangent Tracking: 
       Compare the transcript against the chapters. Identify tangents that took too long or deviations from the main topic.
    
    7. Question Responsiveness: 
       Measure the latency from a student question to the teacher's response.
    
    Output Format:
    Return only the HTML body content (no <html>, <head>, or <body> tags) in a structured format (no CSS) with the following sections:
    - Executive Summary (Overall grade/score)
    - Engagement Metrics (Talk Ratio, Dead Air)
    - Content & Clarity (Jargon, Cognitive Load)
    - Performance & Structure (Pace, Tangents, Responsiveness)
    - Recommendations for Improvement (CRITICAL: Include specific video timings for each example or suggested area of change)
    - Appendix: Raw Data Metrics (Translate the manual analysis results provided above into {language})
    
    Be objective and critical.
    """

    print(f"Requesting deep analysis from Gemini (model: {delivery_metrics_model})...")
    try:
        response = retry_generate_content(
            model=delivery_metrics_model,
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type
                ),
                prompt
            ]
        )
        
        cost, in_tok, out_tok = calculate_gemini_cost(response)
        print(f"✓ Analysis complete. Cost: ${cost:.6f} | Tokens: {in_tok} in / {out_tok} out")
        
        # Step 5: Save the report
        report_body = response.text
        
        # Clean up possible markdown code blocks if Gemini includes them
        report_body = re.sub(r'^```html\n?', '', report_body)
        report_body = re.sub(r'\n?```$', '', report_body)

        full_html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="utf-8">
    <title>Delivery Metrics - {webinar_topic or "Analysis"}</title>
</head>
<body style="font-family: sans-serif; line-height: 1.5; max-width: 800px; margin: 2em auto; padding: 0 1em;">
{report_body}
</body>
</html>"""
        
        base_path = os.path.splitext(srt_path)[0]
        output_path = f"{base_path}_delivery_metrics.html"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)
            
        print(f"✓ Delivery metrics report saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during Gemini analysis: {e}")
        return None
    finally:
        if uploaded_file:
            try: retry_delete_file(name=uploaded_file.name)
            except: pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        srt_file = sys.argv[1]
        chapters_file = sys.argv[2] if len(sys.argv) > 2 else None
        lang = sys.argv[3] if len(sys.argv) > 3 else "en"
        topic = sys.argv[4] if len(sys.argv) > 4 else None
        
        generate_delivery_metrics(srt_file, chapters_file, lang, topic)
    else:
        print("Usage: python delivery_metrics.py <srt_file> [chapters_file] [language] [webinar_topic]")
