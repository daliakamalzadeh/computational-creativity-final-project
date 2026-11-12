import os
import sys
import time
import requests
import platform
import numpy as np
import librosa
import soundfile as sf

# --- Configuration (Wizard of Oz Inputs) ---
FREESOUND_API_KEY = "YOUR_API_KEY_HERE"
INPUT_AUDIO_PATH = "my_song_segment.wav"  # Ensure this file exists on your machine
SEARCH_QUERY = "high guitar loop"  # You are acting as the LLM here!


# --- 1. Audacity Pipe Setup ---
def get_pipe_names():
    """Returns the correct named pipe paths based on the OS."""
    if platform.system() == 'Windows':
        return r'\\.\pipe\ToSrvPipe', r'\\.\pipe\FromSrvPipe'
    else:
        uid = os.getuid()
        return f'/tmp/audacity_script_pipe.to.{uid}', f'/tmp/audacity_script_pipe.from.{uid}'


def send_audacity_command(command):
    """Sends a command to Audacity and reads the response."""
    write_name, read_name = get_pipe_names()

    if not os.path.exists(write_name) and platform.system() != 'Windows':
        print("Error: Audacity pipe not found. Is Audacity running with mod-script-pipe enabled?")
        sys.exit(1)

    # Write command to Audacity
    with open(write_name, 'w') as fp:
        fp.write(command + '\n')

    # Read response from Audacity
    response = ""
    with open(read_name, 'r') as fp:
        while True:
            line = fp.readline()
            if line == '\n' or line == "":
                break
            response += line
    return response


# --- 2. Analyze Input Audio ---
print(f"Analyzing input audio: {INPUT_AUDIO_PATH}")
y_in, sr_in = librosa.load(INPUT_AUDIO_PATH)
bpm_in, _ = librosa.beat.beat_track(y=y_in, sr=sr_in)

# Handle Librosa version differences (sometimes returns array, sometimes float)
bpm_in = float(bpm_in[0]) if isinstance(bpm_in, np.ndarray) else float(bpm_in)
print(f"Detected Input BPM: {bpm_in:.2f}")

# --- 3. Freesound Search (Wizard of Oz execution) ---
print(f"Searching Freesound for: '{SEARCH_QUERY}'...")
search_url = "https://freesound.org/apiv2/search/"
params = {
    "query": SEARCH_QUERY,
    "token": FREESOUND_API_KEY,
    "fields": "id,name,previews",
    "page_size": 1  # Grab the top result
}

response = requests.get(search_url, params=params).json()

if not response.get("results"):
    print(f"No results found on Freesound for '{SEARCH_QUERY}'. Try a different query.")
    sys.exit()

sound_info = response["results"][0]
# Use the HQ mp3 preview to avoid needing complex OAuth2 implementation for raw wav files
preview_url = sound_info["previews"]["preview-hq-mp3"]
print(f"Found match: '{sound_info['name']}'")

# Download the sample
download_path = "downloaded_sample.mp3"
with open(download_path, "wb") as f:
    f.write(requests.get(preview_url).content)

# --- 4. Edit BPM to Match ---
print("Analyzing downloaded sample BPM...")
y_samp, sr_samp = librosa.load(download_path)
bpm_samp, _ = librosa.beat.beat_track(y=y_samp, sr=sr_samp)
bpm_samp = float(bpm_samp[0]) if isinstance(bpm_samp, np.ndarray) else float(bpm_samp)

# Calculate the speed adjustment ratio
rate = bpm_in / bpm_samp
print(f"Sample BPM: {bpm_samp:.2f}. Stretching by factor of {rate:.2f} to match {bpm_in:.2f}...")

# Time-stretch the audio using Librosa
y_stretched = librosa.effects.time_stretch(y_samp, rate=rate)
output_path = os.path.abspath("stretched_sample.wav")

# Save the stretched version as a WAV file
sf.write(output_path, y_stretched, sr_samp)

# --- 5. Insert into Audacity ---
print(f"Inserting stretched file into active Audacity project...")
# The Audacity macro command to import a file
command = f'Import2: Filename="{output_path}"'
result = send_audacity_command(command)
print("Audacity response:\n", result)
print("Success! Check your Audacity window.")