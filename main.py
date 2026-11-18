import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

from audacity_workflow import AudacityError, AudacityPipe, Selection, audition, sample_preview

CLAP_MODEL = "laion/clap-htsat-unfused"
BEAT_SAMPLE_RATE = 22050   # beat tracker always works at this rate, whatever the file's rate
CLAP_BATCH_SIZE = 4        # small batches keep CPU memory use down
CLAP_MAX_SECONDS = 10      # CLAP only listens to the first 10 seconds of each candidate

@dataclass(frozen=True)
class AudioAnalysis:
    duration: float
    bpm: Optional[float]
    beat_times: Tuple[float, ...]

@dataclass(frozen=True)
class Candidate:
    path: Path
    name: str
    url: str = ""
    author: str = ""
    license: str = ""


# ---------------------------------------------------------------------------
# Step 1: analyze the audio we want to match (tempo and beats)
# ---------------------------------------------------------------------------

def refine_bpm(bpm, beats):
    """Average the beat gaps that agree with the tempo, to cancel frame rounding error."""
    import numpy as np

    intervals = np.diff(beats)
    period = 60 / bpm
    consistent = intervals[np.abs(intervals - period) < 0.2 * period]
    if len(consistent) >= 3:
        return float(60 / np.mean(consistent))
    return bpm


def analyze_audio(audio, sample_rate):
    import librosa
    import numpy as np

    mono = librosa.to_mono(audio)
    if not mono.size or not np.isfinite(mono).all():
        raise ValueError("Audio is empty or contains invalid samples.")

    mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=BEAT_SAMPLE_RATE)
    tempo, frames = librosa.beat.beat_track(y=mono, sr=BEAT_SAMPLE_RATE)
    bpm = float(np.asarray(tempo).item())
    beats = librosa.frames_to_time(frames, sr=BEAT_SAMPLE_RATE)

    if len(beats) < 2 or not math.isfinite(bpm) or bpm <= 0:
        bpm = None  # not enough beats to trust the tempo
    elif len(beats) >= 4:
        bpm = refine_bpm(bpm, beats)

    return AudioAnalysis(
        duration=audio.shape[-1] / sample_rate,
        bpm=bpm,
        beat_times=tuple(float(beat) for beat in beats),
    )


# ---------------------------------------------------------------------------
# Step 2: collect candidate samples (from Freesound or local files)
# ---------------------------------------------------------------------------

def search_freesound(query, token, limit):
    """Return the raw Freesound search results (1 to 30 second sounds)."""
    import requests

    try:
        response = requests.get(
            "https://freesound.org/apiv2/search/",
            headers={"Authorization": f"Token {token}"},
            params={
                "query": query,
                "fields": "id,name,url,username,license,previews",
                "filter": "duration:[1 TO 30]",
                "page_size": limit,
            },
            timeout=20,
        )
        if response.status_code in (401, 403):
            raise ValueError("Freesound rejected FREESOUND_API_KEY.")
        response.raise_for_status()
        return response.json()["results"]
    except (requests.RequestException, KeyError) as exc:
        raise ValueError(f"Freesound search failed: {exc}") from exc


def freesound_candidates(query, token, directory, limit):
    """Search Freesound and download each result's preview MP3 into `directory`."""
    import requests

    candidates = []
    for result in search_freesound(query, token, limit):
        preview_url = result.get("previews", {}).get("preview-hq-mp3")
        if not preview_url:
            continue
        try:
            response = requests.get(preview_url, timeout=20)
            response.raise_for_status()
        except requests.RequestException:
            print(f"Skipping unavailable Freesound preview {result['id']}.")
            continue

        path = directory / f"freesound-{int(result['id'])}.mp3"
        path.write_bytes(response.content)
        candidates.append(Candidate(
            path, result["name"], result.get("url", ""),
            result.get("username", ""), result.get("license", ""),
        ))
    return candidates


def load_usable_candidates(candidates):
    """Load each candidate's audio, skipping files that are broken, empty or silent."""
    import librosa
    import numpy as np
    import soundfile as sf

    loaded = []
    for candidate in candidates:
        try:
            audio, rate = librosa.load(candidate.path, sr=None, mono=False)
            if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
                raise ValueError("empty, silent, or invalid audio")
            if audio.ndim == 2 and audio.shape[0] > 2:
                audio = librosa.to_mono(audio)  # more than stereo: fold down to mono
        except (sf.LibsndfileError, OSError, ValueError) as exc:
            print(f"Skipping {candidate.name}: {exc}")
            continue
        loaded.append((candidate, audio, rate))

    if not loaded:
        raise ValueError("No usable candidates. Try a different search or local sample files.")
    return loaded


# ---------------------------------------------------------------------------
# Step 3: rank candidates by how well they match the text description
# ---------------------------------------------------------------------------

def clap_scores(description, audio_files):
    """Score each (audio, sample_rate) pair against the description. Higher is a better match."""
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model = ClapModel.from_pretrained(CLAP_MODEL).eval()
    rate = processor.feature_extractor.sampling_rate
    text_inputs = processor.tokenizer(
        description, truncation=True, max_length=77, return_tensors="pt",
    )

    scores = []
    for start in range(0, len(audio_files), CLAP_BATCH_SIZE):
        batch = audio_files[start:start + CLAP_BATCH_SIZE]
        clips = [
            librosa.resample(
                librosa.to_mono(audio)[..., :sample_rate * CLAP_MAX_SECONDS],
                orig_sr=sample_rate, target_sr=rate,
            )
            for audio, sample_rate in batch
        ]
        audio_inputs = processor.feature_extractor(
            clips, sampling_rate=rate, return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**audio_inputs, **text_inputs).logits_per_text[0]
        scores.extend(logits.tolist())
    return scores


# ---------------------------------------------------------------------------
# Step 4: adjust a candidate so it fits the target (tempo and length)
# ---------------------------------------------------------------------------

def prepare_sample(audio, sample_rate, target):
    """Tempo-match a clip when a plausible beat rate was detected and fit the selection."""
    import librosa
    import numpy as np

    analysis = analyze_audio(audio, sample_rate)

    # Stretch to the target tempo, if both tempos are known.
    rate = None
    if target.bpm and analysis.bpm:
        # Beat tracking can report the pulse at half or double the musical tempo,
        # so pick whichever reading is closest to the target.
        tempos = (analysis.bpm / 2, analysis.bpm, analysis.bpm * 2)
        closest = min(tempos, key=lambda bpm: abs(math.log(target.bpm / bpm)))
        rate = target.bpm / closest
        if abs(rate - 1) > 0.005:  # ignore tiny differences
            audio = librosa.effects.time_stretch(audio, rate=rate)

    # Trim to the selection's length, with a 10 ms fade-out to avoid a click.
    count = max(1, round(target.duration * sample_rate))
    if audio.shape[-1] > count:
        audio = audio[..., :count].copy()
        fade = min(count, max(1, round(sample_rate * 0.01)))
        audio[..., -fade:] *= np.linspace(1, 0, fade)

    # Time stretching can create peaks beyond the WAV range, so scale them back.
    peak = float(np.max(np.abs(audio)))
    if peak > 1:
        audio = audio / peak

    return audio, analysis.bpm, rate


def prepare_candidates(candidates, description, target, output, limit, use_clap):
    """Rank candidates, prepare the best ones, write them as WAVs, and return the suggestions."""
    import soundfile as sf

    loaded = load_usable_candidates(candidates)

    if use_clap:
        print(f"Ranking {len(loaded)} samples with CLAP...", flush=True)
        scores = clap_scores(description, [(audio, rate) for _, audio, rate in loaded])
        ranked = sorted(zip(scores, loaded), key=lambda item: item[0], reverse=True)
    else:
        ranked = [(None, item) for item in loaded]  # keep the original order

    suggestions = []
    for number, (score, (candidate, audio, sample_rate)) in enumerate(ranked[:limit], 1):
        prepared, bpm, speed = prepare_sample(audio, sample_rate, target)

        output_path = output / f"suggestion-{number:02d}.wav"
        with output_path.open("xb") as wav:
            sf.write(wav, prepared.T, sample_rate, format="WAV", subtype="PCM_24")

        details = f"CLAP {score:.2f}, " if score is not None else ""
        details += f"{bpm:.1f} BPM, speed {speed:.3f}x" if speed is not None else "original speed"
        print(f"{number}. {candidate.name} ({details})")
        if candidate.url:
            print(f"   {candidate.author} | {candidate.url} | {candidate.license}")

        suggestions.append((candidate.name, sample_preview, output_path))
    return suggestions

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Analyze a file instead of the Audacity selection.")
    parser.add_argument("--query", help="Sound description to rank against and to search on Freesound.")
    parser.add_argument("--search", help="Optional shorter Freesound search terms.")
    parser.add_argument("--local", type=Path, action="append", default=[],
                        help="Use local candidates. Repeat for each file.")
    parser.add_argument("--output", type=Path, help="Save WAVs here instead of previewing in Audacity.")
    parser.add_argument("--pool", type=int, choices=range(1, 31), default=12, metavar="1-30")
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=3, metavar="1-10")
    parser.add_argument("--skip-clap", action="store_true",
                        help="Use search or input order to test the audio workflow without CLAP.")
    args = parser.parse_args()

    if (args.local or args.search or args.output) and not args.query:
        parser.error("--local, --search and --output require --query")
    if args.query and args.input and not args.output:
        parser.error("add --output when matching against a file, or omit --input to preview against the Audacity selection")
    if args.query and not args.query.strip():
        parser.error("--query must contain a description")

    args.token = os.environ.get("FREESOUND_API_KEY", "").strip()
    if args.query and not args.local and not args.token:
        parser.error("set FREESOUND_API_KEY in your environment or supply local files with --local")
    return parser, args


def main():
    parser, args = parse_args()

    try:
        import librosa

        with TemporaryDirectory() as directory:
            temp = Path(directory)

            # Get the audio to match: a file, or the selection exported from Audacity.
            if args.input:
                source = args.input
            else:
                with AudacityPipe() as pipe:
                    selection = Selection.capture(pipe)
                    source = selection.export(pipe, temp / "selection.wav")

            audio, sample_rate = librosa.load(source, sr=None, mono=False)
            analysis = analyze_audio(audio, sample_rate)
            print(json.dumps(asdict(analysis), indent=2))

            if not args.query:
                return

            # Find candidates, then rank and prepare them.
            if args.local:
                candidates = [Candidate(path, path.stem) for path in args.local]
            else:
                candidates = freesound_candidates(args.search or args.query, args.token, temp, args.pool)

            output = args.output or temp
            output.mkdir(parents=True, exist_ok=True)
            suggestions = prepare_candidates(
                candidates, args.query, analysis, output, args.limit, not args.skip_clap,
            )

            # Without --output, preview the suggestions inside Audacity.
            if not args.output:
                with AudacityPipe() as pipe:
                    if Selection.capture(pipe) != selection:
                        raise AudacityError("The Audacity selection changed. Rerun on the intended section.")
                    audition(pipe, selection, suggestions)

    except (AudacityError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except (EOFError, KeyboardInterrupt):
        parser.exit(130, "\nCancelled.\n")

if __name__ == "__main__":
    main()