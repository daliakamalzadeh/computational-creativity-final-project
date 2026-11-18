import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

# Effects we can preview, mapped to the Audacity command that applies them.
EFFECTS = {"fade-in": "FadeIn:", "fade-out": "FadeOut:"}

# Accepted answers at the Keep / Reject / Another prompt.
ANSWERS = {
    "k": "keep", "keep": "keep",
    "r": "reject", "reject": "reject",
    "n": "another", "another": "another", "suggest another": "another",
}


class AudacityError(RuntimeError):
    pass


class AudacityPipe:
    """Connection to Audacity's scripting pipe (use with a `with` block)."""

    def __enter__(self):
        if sys.platform == "win32":
            to_audacity, from_audacity = r"\\.\pipe\ToSrvPipe", r"\\.\pipe\FromSrvPipe"
            self.ending = b"\r\n\0"
        else:
            prefix = "/tmp/audacity_script_pipe"
            to_audacity = f"{prefix}.to.{os.getuid()}"
            from_audacity = f"{prefix}.from.{os.getuid()}"
            self.ending = b"\n"

        self.writer = self.reader = None
        try:
            self.writer = open(to_audacity, "wb")
            self.reader = open(from_audacity, "rb")
        except OSError as exc:
            self._close()
            raise AudacityError(
                "Open one project in Audacity, enable mod-script-pipe in Preferences "
                "→ Modules, and restart Audacity."
            ) from exc
        return self

    def __exit__(self, *exc_info):
        self._close()

    def _close(self):
        for handle in (self.reader, self.writer):
            if handle:
                handle.close()

    def command(self, command):
        """Send one command and return its output; raise if Audacity reports a failure."""
        self.writer.write(command.encode("utf-8") + self.ending)
        self.writer.flush()

        # The reply ends with a blank line, after at least one line of output.
        lines = []
        while True:
            raw = self.reader.readline()
            if not raw:
                raise AudacityError("Audacity closed the scripting connection.")
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
            elif lines:
                break

        if lines[-1] != "BatchCommand finished: OK":
            name = command.split(":", 1)[0]
            raise AudacityError(f"{name} failed: {' '.join(lines)}")
        return "\n".join(lines[:-1])

    def info(self, kind):
        """Ask Audacity for Tracks, Clips or Selection info, parsed from JSON."""
        return json.loads(self.command(f"GetInfo: Type={kind} Format=JSON"))


def quote(value):
    """Wrap a value in quotes for an Audacity command, rejecting unsafe characters."""
    value = str(value)
    if any(character in value for character in '"\r\n\0'):
        raise AudacityError("The filename or track name contains unsupported characters.")
    return f'"{value}"'


@dataclass(frozen=True)
class Selection:
    start: float
    end: float
    tracks: tuple  # indexes of the selected tracks

    @classmethod
    def capture(cls, pipe):
        """Read the current selection from Audacity and check it is usable."""
        bounds = pipe.info("Selection")
        tracks = tuple(
            index for index, track in enumerate(pipe.info("Tracks")) if track["selected"]
        )
        if bounds["End"] <= bounds["Start"] or not tracks:
            raise AudacityError("Highlight a time range on an audio track in Audacity.")

        has_audio = any(
            clip["track"] in tracks
            and clip["start"] < bounds["End"]
            and clip["end"] > bounds["Start"]
            for clip in pipe.info("Clips")
        )
        if not has_audio:
            raise AudacityError("The highlighted range does not contain an audio clip.")
        return cls(bounds["Start"], bounds["End"], tracks)

    def restore(self, pipe):
        """Re-select the original time range and tracks."""
        pipe.command(f"SelectTime: Start={self.start} End={self.end} RelativeTo=ProjectStart")
        for position, track in enumerate(self.tracks):
            mode = "Add" if position else "Set"  # first track replaces, the rest are added
            pipe.command(f"Select: Track={track} TrackCount=1 Mode={mode}")

    def export(self, pipe, destination):
        """Save the selected audio to a new .wav file."""
        destination = Path(destination).resolve()
        if destination.suffix.lower() != ".wav" or destination.exists():
            raise AudacityError("Choose a new .wav file for the selection export.")

        # Audacity can export to its temp directory even when Windows blocks
        # the Audacity process from writing directly to the project folder.
        with TemporaryDirectory() as directory:
            staged = Path(directory) / "selection.wav"
            pipe.command(f"Export2: Filename={quote(staged)} NumChannels=2")
            if not staged.is_file() or staged.stat().st_size <= 44:  # 44 bytes = empty WAV header
                raise AudacityError("Audacity did not produce a usable selection export.")
            with staged.open("rb") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
        return destination


def sample_preview(pipe, selection, path):
    """Import a sample file and place it at the start of the selection."""
    marker = f"COCONUT-preview-{uuid4().hex}"  # unique name so we can find the new track

    def find_track():
        for index, track in enumerate(pipe.info("Tracks")):
            if track["name"] == marker:
                return index
        raise AudacityError("The preview track could not be found in this project.")

    def finish(keep):
        track = find_track()
        pipe.command(f"Select: Track={track} TrackCount=1 Mode=Set")
        if keep:
            pipe.command(f"SetTrack: Name={quote(path.stem)}")  # give it a proper name
        else:
            pipe.command("RemoveTracks:")
        selection.restore(pipe)

    if not path.is_file():
        raise AudacityError(f"Sample file does not exist: {path}")
    quote(path.stem)  # fail early if the name can't be used as a track name

    with TemporaryDirectory() as directory:
        # Import a copy named with the marker, so the new track gets that name.
        staged = Path(directory) / f"{marker}{path.suffix}"
        shutil.copyfile(path, staged)
        tracks_before = len(pipe.info("Tracks"))
        pipe.command(f"Import2: Filename={quote(staged.resolve())}")

        try:
            track = find_track()
            clips = [clip for clip in pipe.info("Clips") if clip["track"] == track]
            if len(pipe.info("Tracks")) != tracks_before + 1 or len(clips) != 1:
                raise AudacityError("Use a sample file containing a single mono or stereo clip.")

            # Move the clip so it starts where the selection starts.
            middle_of_clip = (clips[0]["start"] + clips[0]["end"]) / 2
            pipe.command(f"Select: Track={track} TrackCount=1 Mode=Set")
            pipe.command(f"SetClip: At={middle_of_clip} Start={selection.start}")
            selection.restore(pipe)
        except (AudacityError, OSError, ValueError):
            # Setup failed: remove the imported track so the project is left clean.
            try:
                finish(False)
            except AudacityError as cleanup_error:
                raise AudacityError(
                    f"Preview setup failed. Inspect Audacity before continuing: {cleanup_error}"
                ) from cleanup_error
            raise
    return finish


def effect_preview(pipe, selection, effect):
    """Apply an effect to the selection; rejecting it simply undoes it."""
    selection.restore(pipe)
    pipe.command(EFFECTS[effect])

    def finish(keep):
        if not keep:
            pipe.command("Undo:")
        selection.restore(pipe)

    return finish


def ask_decision():
    """Prompt until the user gives a valid answer: keep, reject or another."""
    while True:
        answer = input("[K]eep  [R]eject  [N] Suggest another: ").strip().lower()
        if answer in ANSWERS:
            return ANSWERS[answer]


def audition(pipe, selection, candidates):
    print("In Audacity, press Space to listen. Stop playback before deciding.")
    print("Leave the project and its edits alone while a preview is pending.")

    for label, preview, value in candidates:
        print(f"Preview: {label}")
        finish = preview(pipe, selection, value)
        decision = "reject"  # if anything interrupts the prompt, undo the preview
        try:
            decision = ask_decision()
        finally:
            finish(decision == "keep")

        if decision != "another":
            print(f"{'Kept' if decision == 'keep' else 'Rejected'}: {label}")
            return
    print("No more suggestions available.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Save the selected audio to a new WAV.")
    parser.add_argument("--sample", action="append", type=Path, default=[],
                        help="A local sample to preview. Repeat for multiple suggestions.")
    parser.add_argument("--effect", action="append", choices=EFFECTS, default=[],
                        help="An effect to preview. Repeat for multiple suggestions.")
    options = parser.parse_args()

    # Each candidate is (label, function that sets up the preview, its argument).
    candidates = [(path.stem, sample_preview, path) for path in options.sample]
    candidates += [(effect, effect_preview, effect) for effect in options.effect]
    if not options.export and not candidates:
        parser.error("provide --export, --sample, or --effect")

    try:
        with AudacityPipe() as pipe:
            selection = Selection.capture(pipe)
            print(f"Selected {selection.start:.3f} to {selection.end:.3f} seconds.")
            if options.export:
                print(f"Exported {selection.export(pipe, options.export)}")
            if candidates:
                audition(pipe, selection, candidates)
    except (AudacityError, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except (EOFError, KeyboardInterrupt):
        parser.exit(130, "\nPreview cancelled.\n")


if __name__ == "__main__":
    main()