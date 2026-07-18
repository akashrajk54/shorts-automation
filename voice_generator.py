"""Convert narration text to an MP3 voiceover using edge-tts (free, no API key)."""
import asyncio
from pathlib import Path

import config  # noqa: F401  (imported first to configure SSL trust before edge_tts)
import edge_tts

# The most natural, human-sounding English voices from edge-tts.
# Multilingual neural voices sound noticeably more lifelike and expressive.
# Other great options to try:
#   en-US-AndrewMultilingualNeural (warm male), en-US-EmmaMultilingualNeural,
#   en-US-BrianMultilingualNeural, en-GB-RyanNeural, en-IN-PrabhatNeural
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"


async def _synthesize(text: str, out_path: Path, voice: str,
                      rate: str = "+3%", volume: str = "+10%", pitch: str = "+0Hz") -> None:
    # Slightly slower + fuller volume reads more naturally and clearly.
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(str(out_path))


def generate_voice(text: str, voice: str = DEFAULT_VOICE, filename: str = "voice.mp3") -> Path:
    """Generate an MP3 from text and return its path."""
    out_path = config.OUTPUT_DIR / filename
    asyncio.run(_synthesize(text, out_path, voice))
    return out_path


# --- Story mode: two child-like voices for the girl and the boy ---
# Ana is a genuine child (girl) voice. For the boy we pitch a young male voice up
# so it reads like a little boy. Both are free edge-tts voices.
GIRL_VOICE = {"voice": "en-US-AnaNeural", "rate": "+8%", "pitch": "+0Hz"}
BOY_VOICE = {"voice": "en-GB-RyanNeural", "rate": "+6%", "pitch": "+35Hz"}


def _voice_for(speaker: str) -> dict:
    return BOY_VOICE if str(speaker).lower().startswith("b") else GIRL_VOICE


def generate_dialogue_voice(dialogue: list[dict], filename: str = "voice.mp3"):
    """Synthesize a two-speaker dialogue into one MP3.

    Returns (audio_path, segments) where each segment is
    {"text", "speaker", "start", "duration"} for caption syncing.
    """
    from moviepy.editor import AudioFileClip, concatenate_audioclips

    out_path = config.OUTPUT_DIR / filename
    tmp_paths: list[Path] = []
    clips = []
    segments = []
    start = 0.0

    for i, turn in enumerate(dialogue):
        speaker = turn.get("speaker", "girl")
        line = (turn.get("line") or "").strip()
        if not line:
            continue
        v = _voice_for(speaker)
        tmp = config.OUTPUT_DIR / f"_line_{i}.mp3"
        asyncio.run(_synthesize(line, tmp, v["voice"], rate=v["rate"], pitch=v["pitch"]))
        clip = AudioFileClip(str(tmp))
        duration = max(0.1, clip.duration - 0.03)  # trim a hair to avoid mp3 over-read
        clip = clip.subclip(0, duration)
        segments.append({"text": line, "speaker": speaker, "start": start, "duration": duration})
        start += duration
        clips.append(clip)
        tmp_paths.append(tmp)

    if not clips:
        raise ValueError("No dialogue lines to synthesize.")

    final = concatenate_audioclips(clips)
    final.write_audiofile(str(out_path), logger=None)

    for c in clips:
        c.close()
    final.close()
    for p in tmp_paths:
        try:
            p.unlink()
        except OSError:
            pass

    return out_path, segments


if __name__ == "__main__":
    p = generate_voice("This is a quick test of the free edge text to speech engine.")
    print(f"Saved: {p}")
