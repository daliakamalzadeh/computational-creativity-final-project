# COCONUT

Preview Freesound samples over a section of your song in Audacity.

## Run the demo on Windows

1. Use **Python 3.12** and **Audacity 3.7.8**. In Audacity, enable **mod-script-pipe** under **Edit → Preferences → Modules**, then restart Audacity.
2. Open a copy of `Practice Track A (126BPM).aup3` as your only open project and highlight 5–10 seconds of audio.
3. Get a [Freesound API key](https://freesound.org/apiv2/apply) from the **Client secret/Api key** column. Run the following in PowerShell from this folder and paste the key when prompted.

```powershell
py -3.12 -m pip install -r requirements-clap.txt
$env:FREESOUND_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "Paste your Freesound API key" -AsSecureString)).Password
py -3.12 .\main.py --query "A short, dry shaker loop with steady eighth notes" --search "shaker loop"
```

CLAP downloads its model on the first run. When **Preview** appears, press **Space** in Audacity to listen. Stop playback before choosing **K** to keep, **R** to reject and end, or **N** to try the next result. Save in Audacity to keep your edits.

## Quick test without an API key

Use the same Audacity setup and selection. This uses the bundled Track B as a demo sample.

```powershell
python -m pip install -r requirements.txt
python .\main.py --query "rhythmic accompaniment" --local "Practice Track B (125BPM).wav" --skip-clap
```