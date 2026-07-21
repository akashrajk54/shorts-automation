"""Convert narration text to an MP3 voiceover using edge-tts (free, no API key)."""
import asyncio
import shutil
import subprocess
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


# ffmpeg filter chain that makes raw TTS sound fuller and more "produced"/human:
#   highpass  -> removes low rumble/muddiness
#   acompressor -> evens out volume so it sounds steady and confident (less robotic)
#   loudnorm  -> broadcast loudness (EBU R128) so it's clear + consistent like real VO
# All filters are gain/tone only (no time-stretch), so word timings stay valid.
_ENHANCE_FILTER = (
    "highpass=f=85,"
    "acompressor=threshold=-18dB:ratio=3:attack=5:release=120,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)


def _enhance_audio(path: Path) -> None:
    """Post-process the MP3 in-place to sound warmer/fuller and more human.

    Best-effort: if ffmpeg is missing or the filter fails, the original file is
    left untouched so the pipeline never breaks.
    """
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".enh.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", str(path), "-af", _ENHANCE_FILTER,
        "-c:a", "libmp3lame", "-q:a", "2", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        try:
            tmp.unlink()
        except OSError:
            pass


def _shift_time(t: float, cuts: list[tuple[float, float]]) -> float:
    """Map a timestamp from the ORIGINAL timeline to the timeline after `cuts`
    (silence regions) have been removed, so word/segment timings stay in sync."""
    removed = 0.0
    for c0, c1 in cuts:
        if c1 <= t:
            removed += c1 - c0
        elif c0 < t:
            removed += t - c0
    return max(0.0, t - removed)


def _reduce_internal_pauses(path: Path, words: list[dict], max_gap: float = 0.18,
                            lead_keep: float = 0.05, tail_keep: float = 0.12):
    """Physically shorten the long silences edge-tts inserts between sentences of
    the SAME speaker, then shift word timings by the removed amount so karaoke
    captions stay perfectly aligned.

    Returns (new_words, cuts). Best-effort: on any failure the original audio and
    words are returned unchanged so the pipeline never breaks.
    """
    if not words:
        return words, []
    try:
        from moviepy.audio.AudioClip import AudioArrayClip
        from moviepy.editor import AudioFileClip

        clip = AudioFileClip(str(path))
        fps = int(clip.fps or 44100)
        arr = clip.to_soundarray(fps=fps)
        clip.close()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = len(arr)
        clip_dur = n / fps

        cuts: list[tuple[float, float]] = []
        # Trim excess leading silence before the first word.
        first = words[0]["start"]
        if first > lead_keep:
            cuts.append((lead_keep, first))
        # Cap the gap between every consecutive pair of words.
        for i in range(len(words) - 1):
            end_i = words[i]["start"] + words[i]["duration"]
            nxt = words[i + 1]["start"]
            if nxt - end_i > max_gap:
                cuts.append((end_i + max_gap, nxt))
        # Trim excess trailing silence after the last word.
        last_end = words[-1]["start"] + words[-1]["duration"]
        if clip_dur - last_end > tail_keep:
            cuts.append((last_end + tail_keep, clip_dur))

        if not cuts:
            return words, []

        mask = np.ones(n, dtype=bool)
        for c0, c1 in cuts:
            a = max(0, min(n, int(round(c0 * fps))))
            b = max(0, min(n, int(round(c1 * fps))))
            if b > a:
                mask[a:b] = False
        new_arr = arr[mask]

        new_clip = AudioArrayClip(new_arr, fps=fps)
        new_clip.write_audiofile(str(path), logger=None)
        new_clip.close()

        new_words = [{**w, "start": _shift_time(w["start"], cuts)} for w in words]
        return new_words, cuts
    except Exception:  # noqa: BLE001
        return words, []


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
    words, _ = _reduce_internal_pauses(out_path, words)  # tighten sentence gaps
    segments = _sentence_segments(text, words) if words else []
    _enhance_audio(out_path)  # warmer, fuller, more human-sounding VO
    return out_path, words, segments


def _voice_for(speaker: str, language: str = None) -> dict:
    """Return {voice, rate, pitch} for a story speaker in the given language.

    Tuned so the two speakers sound like real young North-Indian (Delhi) kids
    chatting in natural Hindi: brighter pitch, slightly quicker, playful pace.
    """
    voices = _voices_for_language(language or config.VIDEO_LANGUAGE)
    if str(speaker).lower().startswith("b"):
        # Boy: male Hindi voice pitched GENTLY up => reads young but stays natural
        # (heavy pitch-shift = chipmunky/robotic, which kills realism).
        return {"voice": voices["boy"], "rate": "+7%", "pitch": "+18Hz"}
    # Girl: female Hindi voice brightened slightly => sweet but still human.
    return {"voice": voices["girl"], "rate": "+8%", "pitch": "+12Hz"}


def _speech_bounds(clip, head: float = 0.04, tail: float = 0.06):
    """Return (start, end) seconds of actual speech in the clip, so we can drop
    BOTH the leading and trailing silence edge-tts pads onto each line. This keeps
    the gap when the speaker switches short AND removes the little pause before a
    kid starts talking. Small head/tail margins avoid clipping words. Falls back
    to the full clip if analysis fails."""
    try:
        chunks = list(clip.iter_chunks(fps=22050, quantize=False, nbytes=2, chunksize=22050))
        arr = np.concatenate(chunks, axis=0)
    except Exception:  # noqa: BLE001
        return 0.0, clip.duration
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    amp = np.abs(arr)
    peak = amp.max()
    if peak <= 0:
        return 0.0, clip.duration
    idx = np.where(amp > peak * 0.02)[0]  # 2% of peak = speech threshold
    if len(idx) == 0:
        return 0.0, clip.duration
    start = max(0.0, idx[0] / 22050 - head)
    end = min(clip.duration, (idx[-1] + 1) / 22050 + tail)
    return start, max(start + 0.2, end)


def _tighten(line: str) -> str:
    """Remove punctuation that makes edge-tts pause dramatically mid-line
    (ellipses and dashes), so a kid's sentence flows without long gaps."""
    import re
    line = line.replace("\u2026", " ")                 # ellipsis char
    line = re.sub(r"\.\.\.+", " ", line)               # "..."
    line = re.sub(r"\s*[\u2014\u2013]\s*", ", ", line)  # em/en dash -> short comma pause
    line = re.sub(r"\s+", " ", line).strip()
    return line


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
        line = _tighten((turn.get("line") or "").strip())
        if not line:
            continue
        v = _voice_for(speaker, language)
        tmp = config.OUTPUT_DIR / f"_line_{i}.mp3"
        line_words = asyncio.run(
            _synthesize(line, tmp, v["voice"], rate=v["rate"], pitch=v["pitch"])
        )
        clip = AudioFileClip(str(tmp))
        # Trim BOTH leading and trailing silence edge-tts pads on, so the gap when
        # the voice switches (girl <-> boy) AND the pause before each kid starts
        # talking stay short and snappy.
        lead, end = _speech_bounds(clip)
        clip = clip.subclip(lead, end)
        duration = end - lead
        for w in line_words:
            # Shift word timings by the trimmed lead so karaoke stays in sync.
            words.append({
                "text": w["text"],
                "speaker": speaker,
                "start": start + max(0.0, w["start"] - lead),
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

    # Tighten long same-speaker sentence gaps, keeping word + segment sync.
    words, cuts = _reduce_internal_pauses(out_path, words)
    if cuts:
        for seg in segments:
            s0 = _shift_time(seg["start"], cuts)
            s1 = _shift_time(seg["start"] + seg["duration"], cuts)
            seg["start"] = s0
            seg["duration"] = max(0.1, s1 - s0)
    _enhance_audio(out_path)  # warmer, fuller, more human-sounding VO
    return out_path, words, segments


if __name__ == "__main__":
    p, words, segs = generate_voice("This is a quick test. It has two sentences today.")
    print(f"Saved: {p} | {len(words)} words | {len(segs)} segments")
