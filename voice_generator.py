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

# --- Per-language voices (all free edge-tts neural voices) ---
# Each entry: narrator (tips mode) + girl/boy (story mode). The boy uses a male
# voice pitched up so it reads like a little boy.
LANGUAGE_VOICES = {
    "english": {"narrator": "en-US-AvaMultilingualNeural",
                "girl": "en-US-AnaNeural", "boy": "en-GB-RyanNeural"},
    "hindi": {"narrator": "hi-IN-SwaraNeural",
              "girl": "hi-IN-SwaraNeural", "boy": "hi-IN-MadhurNeural"},
    "bengali": {"narrator": "bn-IN-TanishaaNeural",
                "girl": "bn-IN-TanishaaNeural", "boy": "bn-IN-BashkarNeural"},
    "tamil": {"narrator": "ta-IN-PallaviNeural",
              "girl": "ta-IN-PallaviNeural", "boy": "ta-IN-ValluvarNeural"},
    "telugu": {"narrator": "te-IN-ShrutiNeural",
               "girl": "te-IN-ShrutiNeural", "boy": "te-IN-MohanNeural"},
    "marathi": {"narrator": "mr-IN-AarohiNeural",
                "girl": "mr-IN-AarohiNeural", "boy": "mr-IN-ManoharNeural"},
    "spanish": {"narrator": "es-ES-ElviraNeural",
                "girl": "es-ES-ElviraNeural", "boy": "es-ES-AlvaroNeural"},
    "french": {"narrator": "fr-FR-DeniseNeural",
               "girl": "fr-FR-DeniseNeural", "boy": "fr-FR-HenriNeural"},
    "german": {"narrator": "de-DE-KatjaNeural",
               "girl": "de-DE-KatjaNeural", "boy": "de-DE-ConradNeural"},
    "arabic": {"narrator": "ar-EG-SalmaNeural",
               "girl": "ar-EG-SalmaNeural", "boy": "ar-EG-ShakirNeural"},
    "japanese": {"narrator": "ja-JP-NanamiNeural",
                 "girl": "ja-JP-NanamiNeural", "boy": "ja-JP-KeitaNeural"},
    "portuguese": {"narrator": "pt-BR-FranciscaNeural",
                   "girl": "pt-BR-FranciscaNeural", "boy": "pt-BR-AntonioNeural"},
}


def _voices_for_language(language: str) -> dict:
    """Return the voice set for a language, falling back to English if unknown."""
    return LANGUAGE_VOICES.get((language or "").strip().lower(), LANGUAGE_VOICES["english"])


async def _synthesize(text: str, out_path: Path, voice: str,
                      rate: str = "+3%", volume: str = "+10%", pitch: str = "+0Hz") -> list[dict]:
    """Stream TTS to out_path and return per-word timings.

    Each word dict: {"text", "start", "duration"} in seconds (relative to this clip).
    """
    communicate = edge_tts.Communicate(
        text, voice, rate=rate, volume=volume, pitch=pitch, boundary="WordBoundary"
    )
    words: list[dict] = []
    with open(out_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # edge-tts reports offsets/durations in 100-nanosecond units.
                words.append({
                    "text": chunk["text"],
                    "start": chunk["offset"] / 1e7,
                    "duration": chunk["duration"] / 1e7,
                })
    return words


def generate_voice(text: str, voice: str = None, filename: str = "voice.mp3",
                   language: str = None):
    """Generate an MP3 and return (path, words) with per-word timings for captions."""
    if voice is None:
        voice = _voices_for_language(language or config.VIDEO_LANGUAGE)["narrator"]
    out_path = config.OUTPUT_DIR / filename
    words = asyncio.run(_synthesize(text, out_path, voice))
    for w in words:
        w["speaker"] = "narrator"
    return out_path, words


def _voice_for(speaker: str, language: str = None) -> dict:
    """Return {voice, rate, pitch} for a story speaker in the given language."""
    voices = _voices_for_language(language or config.VIDEO_LANGUAGE)
    if str(speaker).lower().startswith("b"):
        # Boy: male voice pitched up to sound like a little boy.
        return {"voice": voices["boy"], "rate": "+6%", "pitch": "+35Hz"}
    return {"voice": voices["girl"], "rate": "+8%", "pitch": "+0Hz"}


def generate_dialogue_voice(dialogue: list[dict], filename: str = "voice.mp3",
                            language: str = None):
    """Synthesize a two-speaker dialogue into one MP3.

    Returns (audio_path, words) where each word is
    {"text", "speaker", "start", "duration"} (global timings) for karaoke captions.
    """
    from moviepy.editor import AudioFileClip, concatenate_audioclips

    language = language or config.VIDEO_LANGUAGE
    out_path = config.OUTPUT_DIR / filename
    tmp_paths: list[Path] = []
    clips = []
    words: list[dict] = []
    start = 0.0

    for i, turn in enumerate(dialogue):
        speaker = turn.get("speaker", "girl")
        line = (turn.get("line") or "").strip()
        if not line:
            continue
        v = _voice_for(speaker, language)
        tmp = config.OUTPUT_DIR / f"_line_{i}.mp3"
        line_words = asyncio.run(
            _synthesize(line, tmp, v["voice"], rate=v["rate"], pitch=v["pitch"])
        )
        clip = AudioFileClip(str(tmp))
        duration = max(0.1, clip.duration - 0.03)  # trim a hair to avoid mp3 over-read
        clip = clip.subclip(0, duration)
        for w in line_words:
            words.append({
                "text": w["text"],
                "speaker": speaker,
                "start": start + w["start"],
                "duration": w["duration"],
            })
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

    return out_path, words


if __name__ == "__main__":
    p, words = generate_voice("This is a quick test of the free edge text to speech engine.")
    print(f"Saved: {p} | {len(words)} word timings")
