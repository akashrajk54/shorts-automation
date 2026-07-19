"""Convert narration text to an MP3 voiceover using edge-tts (free, no API key)."""
import asyncio
from pathlib import Path

import config  # noqa: F401  (imported first to configure SSL trust before edge_tts)
import edge_tts
import numpy as np

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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (handles English + Indic danda '।')."""
    import re
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence_segments(text: str, words: list[dict], speaker: str = "narrator") -> list[dict]:
    """Group per-word timings into per-sentence segments for image syncing."""
    sentences = _split_sentences(text)
    segments: list[dict] = []
    wi = 0
    for i, sent in enumerate(sentences):
        n = len(sent.split())
        grp = words[wi:wi + n] if n else []
        wi += n
        if not grp:
            continue
        start = grp[0]["start"]
        # Last sentence extends to the final word so nothing is left uncovered.
        end_words = words[-1] if i == len(sentences) - 1 else grp[-1]
        end = end_words["start"] + end_words["duration"]
        segments.append({"start": start, "duration": max(0.1, end - start),
                         "text": sent, "speaker": speaker})
    return segments


def generate_voice(text: str, voice: str = None, filename: str = "voice.mp3",
                   language: str = None):
    """Generate an MP3 and return (path, words, segments).

    words = per-word timings (karaoke); segments = per-sentence timings (image sync).
    """
    if voice is None:
        voice = _voices_for_language(language or config.VIDEO_LANGUAGE)["narrator"]
    out_path = config.OUTPUT_DIR / filename
    words = asyncio.run(_synthesize(text, out_path, voice))
    for w in words:
        w["speaker"] = "narrator"
    segments = _sentence_segments(text, words) if words else []
    return out_path, words, segments


def _voice_for(speaker: str, language: str = None) -> dict:
    """Return {voice, rate, pitch} for a story speaker in the given language.

    Tuned so the two speakers sound like real young North-Indian (Delhi) kids
    chatting in natural Hindi: brighter pitch, slightly quicker, playful pace.
    """
    voices = _voices_for_language(language or config.VIDEO_LANGUAGE)
    if str(speaker).lower().startswith("b"):
        # Boy: male Hindi voice pitched well up + a touch faster => little boy.
        return {"voice": voices["boy"], "rate": "+10%", "pitch": "+45Hz"}
    # Girl: female Hindi voice brightened + livelier => sweet little girl.
    return {"voice": voices["girl"], "rate": "+13%", "pitch": "+30Hz"}


def _speech_end(clip, tail: float = 0.05) -> float:
    """Return the timestamp where actual speech ends, so we can drop the long
    trailing silence edge-tts pads onto each line. Keeps a small `tail` so words
    aren't clipped. Falls back to the full duration if analysis fails."""
    try:
        chunks = list(clip.iter_chunks(fps=22050, quantize=False, nbytes=2, chunksize=22050))
        arr = np.concatenate(chunks, axis=0)
    except Exception:  # noqa: BLE001
        return clip.duration
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    amp = np.abs(arr)
    peak = amp.max()
    if peak <= 0:
        return clip.duration
    idx = np.where(amp > peak * 0.02)[0]  # 2% of peak = speech threshold
    if len(idx) == 0:
        return clip.duration
    end = (idx[-1] + 1) / 22050 + tail
    return float(min(clip.duration, max(0.2, end)))


def generate_dialogue_voice(dialogue: list[dict], filename: str = "voice.mp3",
                            language: str = None):
    """Synthesize a two-speaker dialogue into one MP3.

    Returns (audio_path, words, segments):
      words    = per-word timings (karaoke captions),
      segments = per-dialogue-line timings (one image per line).
    All timings are global {"text", "speaker", "start", "duration"}.
    """
    from moviepy.editor import AudioFileClip, concatenate_audioclips

    language = language or config.VIDEO_LANGUAGE
    out_path = config.OUTPUT_DIR / filename
    tmp_paths: list[Path] = []
    clips = []
    words: list[dict] = []
    segments: list[dict] = []
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
        # Trim the long trailing silence edge-tts pads on, so the gap when the
        # voice switches (girl <-> boy) stays short and snappy.
        duration = _speech_end(clip)
        clip = clip.subclip(0, duration)
        for w in line_words:
            words.append({
                "text": w["text"],
                "speaker": speaker,
                "start": start + w["start"],
                "duration": w["duration"],
            })
        # One segment (=> one image) per dialogue line.
        segments.append({"text": line, "speaker": speaker,
                         "start": start, "duration": duration})
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

    return out_path, words, segments


if __name__ == "__main__":
    p, words, segs = generate_voice("This is a quick test. It has two sentences today.")
    print(f"Saved: {p} | {len(words)} words | {len(segs)} segments")
