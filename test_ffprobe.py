import subprocess
import os

path = "/Users/artemreva/Documents/otus_webinars/AI-Agents/7-AI_Agents_2026_02_Практика_промптинга._Проработка_требован/AI_Agents_2026_02_Практика_промптинга._Проработка_требован.mp4"

print(f"Testing path: {path}")

command = [
    "ffprobe",
    "-v", "error",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-show_format",
    path
]

print(f"Running command: {' '.join(command)}")
try:
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    print(f"Return code: {result.returncode}")
    print(f"Stdout: '{result.stdout.strip()}'")
    print(f"Stderr: '{result.stderr.strip()}'")
except Exception as e:
    print(f"Exception: {e}")
