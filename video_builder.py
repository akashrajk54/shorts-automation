"""Assemble a vertical Shorts video: AI-image slideshow + voiceover + captions."""
import random
from pathlib import Path

import config  # noqa: F401  (imported first to configure SSL trust)
import numpy as np
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont

# Pillow 10+ removed Image.ANTIALIAS, which moviepy 1.0.3 still references.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


def _fit_cover(clip: ImageClip) -> ImageClip:
    """Resize an image clip so it fully covers the vertical frame."""
    ratio = max(config.VIDEO_WIDTH / clip.w, config.VIDEO_HEIGHT / clip.h)
    return clip.resize(ratio)


def _ken_burns(image_path: Path, duration: float, start_time: float = 0.0,
               envelope=None, zoom: float = 0.10, pulse: float = 0.05) -> CompositeVideoClip:
    """Slow Ken Burns zoom that also gently PULSES with the voice's loudness."""
    base = ImageClip(str(image_path)).set_duration(duration)
    base = _fit_cover(base)

    def scale(t):
        s = 1 + zoom * (t / max(duration, 0.1))
        if envelope is not None:
            s += pulse * envelope(start_time + t)  # louder speech -> slightly bigger
        return s

    zoomed = base.resize(scale).set_position(("center", "center"))
    return CompositeVideoClip(
        [zoomed], size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
    ).set_duration(duration)


def _audio_envelope(voice_path: Path, fps: int = 50):
    """Return a callable env(t)->0..1 giving the (smoothed) loudness at time t."""
    try:
        clip = AudioFileClip(str(voice_path))
        # NOTE: clip.to_soundarray() is broken on moviepy 1.0.3 + numpy 2.x
        # (it passes a generator to np.vstack), so collect chunks into a list.
        chunks = list(clip.iter_chunks(fps=22050, quantize=False, nbytes=2, chunksize=22050))
        clip.close()
        arr = np.concatenate(chunks, axis=0)
    except Exception:  # noqa: BLE001
        return lambda t: 0.0
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    per = max(1, int(22050 / fps))
    n = len(arr) // per
    if n == 0:
        return lambda t: 0.0
    rms = np.sqrt(np.mean(arr[:n * per].reshape(n, per) ** 2, axis=1))
    peak = rms.max() or 1.0
    vals = np.clip(rms / peak, 0.0, 1.0)
    times = np.arange(n) / fps

    def env(t: float) -> float:
        return float(np.interp(t, times, vals, left=vals[0], right=vals[-1]))

    return env


# Script-specific fonts so captions render correctly in non-Latin languages.
_SUP = "/System/Library/Fonts/Supplemental/"
LANGUAGE_FONTS = {
    "hindi": [_SUP + "Devanagari Sangam MN.ttc", _SUP + "DevanagariMT.ttc"],
    "marathi": [_SUP + "Devanagari Sangam MN.ttc", _SUP + "DevanagariMT.ttc"],
    "bengali": [_SUP + "Bangla Sangam MN.ttc", _SUP + "Bangla MN.ttc"],
    "tamil": [_SUP + "Tamil Sangam MN.ttc", _SUP + "Tamil MN.ttc"],
    "telugu": [_SUP + "Telugu Sangam MN.ttc", _SUP + "Telugu MN.ttc"],
    "gujarati": [_SUP + "Gujarati Sangam MN.ttc"],
    "kannada": [_SUP + "Kannada Sangam MN.ttc"],
    "malayalam": [_SUP + "Malayalam Sangam MN.ttc"],
    "punjabi": [_SUP + "Gurmukhi Sangam MN.ttc"],
    "arabic": [_SUP + "GeezaPro.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "japanese": ["/System/Library/Fonts/Hiragino Sans GB.ttc"],
    "chinese": ["/System/Library/Fonts/PingFang.ttc"],
}

# Latin bold fonts used for English + as the final fallback.
_LATIN_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a font that supports the current VIDEO_LANGUAGE's script."""
    lang = (getattr(config, "VIDEO_LANGUAGE", "english") or "english").lower()
    candidates = LANGUAGE_FONTS.get(lang, []) + _LATIN_FONTS
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_by_pixels(draw, text, font, max_width, stroke):
    """Wrap text into lines that fit within max_width pixels."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke)
        if (bbox[2] - bbox[0]) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _segment_narration(text: str, max_words: int = 5) -> list[str]:
    """Split narration into short subtitle chunks (a few words each)."""
    import re

    phrases = re.split(r"(?<=[.!?,;:])\s+", text.strip())
    segments: list[str] = []
    for phrase in phrases:
        words = phrase.split()
        if not words:
            continue
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words]).strip()
            if chunk:
                segments.append(chunk)
    return segments or [text]


def _render_caption_image(text: str, duration: float, color: str = "white") -> ImageClip:
    """Render one centered, stroked caption chunk that always fits the frame."""
    margin = 90
    max_width = config.VIDEO_WIDTH - 2 * margin
    max_block_height = int(config.VIDEO_HEIGHT * 0.35)
    stroke = 6

    img = Image.new("RGBA", (config.VIDEO_WIDTH, config.VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Auto-shrink the font until the wrapped block fits width and height.
    font_size = 96
    while font_size >= 40:
        font = _load_font(font_size)
        lines = _wrap_by_pixels(draw, text, font, max_width, stroke)
        line_height = int(font_size * 1.2)
        block_height = line_height * len(lines)
        widest = max(
            (draw.textbbox((0, 0), ln, font=font, stroke_width=stroke)[2] for ln in lines),
            default=0,
        )
        if widest <= max_width and block_height <= max_block_height:
            break
        font_size -= 6

    # Vertically center the block around 68% of the height (lower third).
    start_y = int(config.VIDEO_HEIGHT * 0.68) - block_height // 2
    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        line_width = bbox[2] - bbox[0]
        x = (config.VIDEO_WIDTH - line_width) // 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=color,
            stroke_width=stroke,
            stroke_fill="black",
        )
        y += line_height

    return ImageClip(np.array(img)).set_duration(duration)


# Speaker colors for story mode (readable over the black stroke).
SPEAKER_COLORS = {"girl": "#FF9ED8", "boy": "#8FD3FF", "narrator": "white"}
# Colour of the word currently being spoken (karaoke highlight).
HIGHLIGHT_COLOR = "#FFE14D"
MAX_WORDS_PER_GROUP = 5


def _layout_words(draw, words: list[str], font, max_width: int):
    """Wrap words into lines that fit max_width. Returns list of lines,
    each a list of (word, local_index)."""
    space_w = draw.textlength(" ", font=font)
    lines, current, current_w = [], [], 0.0
    for idx, word in enumerate(words):
        ww = draw.textlength(word, font=font)
        add = ww if not current else space_w + ww
        if current and current_w + add > max_width:
            lines.append(current)
            current, current_w = [], 0.0
            add = ww
        current.append((word, idx))
        current_w += add
    if current:
        lines.append(current)
    return lines


def _render_karaoke_image(words: list[str], active_idx: int, base_color: str) -> np.ndarray:
    """Render a group of words centered, with the active word highlighted."""
    margin = 90
    max_width = config.VIDEO_WIDTH - 2 * margin
    max_block_height = int(config.VIDEO_HEIGHT * 0.35)
    stroke = 6

    img = Image.new("RGBA", (config.VIDEO_WIDTH, config.VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Auto-shrink until the wrapped block fits.
    font_size = 96
    while font_size >= 40:
        font = _load_font(font_size)
        lines = _layout_words(draw, words, font, max_width)
        line_height = int(font_size * 1.2)
        block_height = line_height * len(lines)
        widest = max(
            (draw.textlength(" ".join(w for w, _ in ln), font=font) for ln in lines),
            default=0,
        )
        if widest <= max_width and block_height <= max_block_height:
            break
        font_size -= 6

    space_w = draw.textlength(" ", font=font)
    start_y = int(config.VIDEO_HEIGHT * 0.68) - block_height // 2
    y = start_y
    for line in lines:
        line_w = sum(draw.textlength(w, font=font) for w, _ in line) + space_w * (len(line) - 1)
        x = (config.VIDEO_WIDTH - line_w) // 2
        for word, idx in line:
            fill = HIGHLIGHT_COLOR if idx == active_idx else base_color
            draw.text((x, y), word, font=font, fill=fill,
                      stroke_width=stroke, stroke_fill="black")
            x += draw.textlength(word, font=font) + space_w
        y += line_height

    return np.array(img)


def _group_words(words: list[dict], max_size: int = MAX_WORDS_PER_GROUP):
    """Group consecutive words (same speaker, up to max_size) for on-screen chunks."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        if current and (len(current) >= max_size
                        or current[0].get("speaker") != w.get("speaker")):
            groups.append(current)
            current = []
        current.append(w)
    if current:
        groups.append(current)
    return groups


def _make_karaoke_clips(words: list[dict], total_duration: float) -> list[ImageClip]:
    """Build word-synced karaoke caption clips from per-word timings."""
    clips: list[ImageClip] = []
    for group in _group_words(words):
        base_color = SPEAKER_COLORS.get(str(group[0].get("speaker", "")).lower(), "white")
        texts = [w["text"] for w in group]
        # Pre-render each highlight state once per group (cheap + avoids duplicates).
        rendered = {k: _render_karaoke_image(texts, k, base_color) for k in range(len(group))}
        group_end = min(group[-1]["start"] + group[-1]["duration"], total_duration)
        for k, w in enumerate(group):
            start = min(w["start"], total_duration)
            end = group[k + 1]["start"] if k + 1 < len(group) else group_end
            dur = max(0.08, min(end, total_duration) - start)
            if dur <= 0:
                continue
            clips.append(ImageClip(rendered[k]).set_duration(dur).set_start(start))
    return clips


def _make_caption_clips(text: str, total_duration: float) -> list[ImageClip]:
    """Build timed subtitle chunks spread across the whole video duration."""
    segments = _segment_narration(text)
    total_words = sum(len(s.split()) for s in segments) or 1

    clips, start = [], 0.0
    for i, seg in enumerate(segments):
        share = len(seg.split()) / total_words
        seg_duration = total_duration * share
        # Make the final chunk fill any remaining time.
        if i == len(segments) - 1:
            seg_duration = max(0.1, total_duration - start)
        clip = _render_caption_image(seg, seg_duration).set_start(start)
        clips.append(clip)
        start += seg_duration
    return clips


def _gradient_bg() -> np.ndarray:
    """Build a soft vertical purple-blue gradient as a fallback background."""
    top = np.array([40, 30, 70])     # deep purple
    bottom = np.array([15, 20, 45])  # dark blue
    h, w = config.VIDEO_HEIGHT, config.VIDEO_WIDTH
    ramp = np.linspace(0, 1, h)[:, None]
    col = (top[None, :] * (1 - ramp) + bottom[None, :] * ramp).astype(np.uint8)
    return np.repeat(col[:, None, :], w, axis=1)


def _pick_music_file() -> Path | None:
    """Return a random royalty-free music file from assets/music, if any exist."""
    if not config.BACKGROUND_MUSIC or not config.MUSIC_DIR.exists():
        return None
    tracks = [
        p for p in config.MUSIC_DIR.iterdir()
        if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg")
    ]
    return random.choice(tracks) if tracks else None


def _mix_background_music(voice: AudioFileClip, duration: float) -> CompositeAudioClip | AudioFileClip:
    """Layer soft background music under the voiceover, looped/trimmed to duration."""
    track = _pick_music_file()
    if track is None:
        return voice
    try:
        music = AudioFileClip(str(track))
        if music.duration < duration:
            music = audio_loop(music, duration=duration)
        else:
            music = music.subclip(0, duration)
        music = music.volumex(config.MUSIC_VOLUME).audio_fadein(0.6).audio_fadeout(0.8)
        print(f"[music] mixing background track: {track.name}")
        return CompositeAudioClip([voice, music]).set_duration(duration)
    except Exception as exc:  # noqa: BLE001
        print(f"[music] failed to mix '{track.name}', continuing without music: {exc}")
        return voice


def build_video(narration: str, voice_path: Path, image_paths: list[Path],
                filename: str = "output.mp4", word_segments: list[dict] = None) -> Path:
    """Create the final Shorts MP4 from an AI-image slideshow and return its path.

    If word_segments (per-word timings) are given, captions are rendered as
    word-by-word karaoke (color-coded per speaker); otherwise the narration is
    auto-split into timed chunks. Scenes also gently pulse with the voice loudness.
    """
    out_path = config.OUTPUT_DIR / filename

    audio = AudioFileClip(str(voice_path))
    # Cap duration slightly under the audio length to avoid moviepy's
    # MP3 over-read bug (reading frames past the decoded end).
    duration = max(0.1, audio.duration - 0.05)
    audio = audio.set_duration(duration)

    # Loudness envelope drives the audio-reactive zoom pulse.
    envelope = _audio_envelope(voice_path)

    if image_paths:
        per = duration / len(image_paths)
        slides = [
            _ken_burns(p, per, start_time=idx * per, envelope=envelope)
            for idx, p in enumerate(image_paths)
        ]
        bg = concatenate_videoclips(slides, method="compose").set_duration(duration)
    else:
        # Fallback: a soft vertical gradient (nicer than flat black) if no images.
        bg = ImageClip(_gradient_bg()).set_duration(duration)

    if word_segments:
        caption_clips = _make_karaoke_clips(word_segments, duration)
    else:
        caption_clips = _make_caption_clips(narration, duration)

    final_audio = _mix_background_music(audio, duration)
    final = CompositeVideoClip([bg, *caption_clips]).set_audio(final_audio)
    final = final.set_duration(duration)

    final.write_videofile(
        str(out_path),
        fps=config.VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
    )
    return out_path


if __name__ == "__main__":
    from image_generator import generate_images
    from voice_generator import generate_voice

    config.validate()
    demo = "Here are three AI tools that will save you hours every single week."
    vp, words = generate_voice(demo)
    imgs = generate_images([
        "Futuristic AI dashboard glowing on a laptop, cinematic, vertical",
        "Robot hand typing on keyboard, neon blue lighting, vertical",
    ])
    p = build_video(demo, vp, imgs, word_segments=words)
    print(f"Saved: {p}")
