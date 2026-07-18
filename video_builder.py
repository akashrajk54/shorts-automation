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


def _ken_burns(image_path: Path, duration: float, zoom: float = 0.10) -> CompositeVideoClip:
    """Create a slow zoom (Ken Burns) slide from an image, cropped to the frame."""
    base = ImageClip(str(image_path)).set_duration(duration)
    base = _fit_cover(base)
    zoomed = (
        base.resize(lambda t: 1 + zoom * (t / max(duration, 0.1)))
        .set_position(("center", "center"))
    )
    return CompositeVideoClip(
        [zoomed], size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
    ).set_duration(duration)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a bold TrueType font, falling back across common macOS paths."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
    ]
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
SPEAKER_COLORS = {"girl": "#FF9ED8", "boy": "#8FD3FF"}


def _make_caption_clips_from_segments(segments: list[dict]) -> list[ImageClip]:
    """Build caption clips from pre-timed dialogue segments, color-coded per speaker.

    Each line is further split into short chunks that share the line's time slot.
    """
    clips: list[ImageClip] = []
    for seg in segments:
        color = SPEAKER_COLORS.get(str(seg.get("speaker", "")).lower(), "white")
        chunks = _segment_narration(seg["text"])
        total_words = sum(len(c.split()) for c in chunks) or 1
        t = seg["start"]
        for j, chunk in enumerate(chunks):
            share = len(chunk.split()) / total_words
            d = seg["duration"] * share
            if j == len(chunks) - 1:
                d = max(0.1, seg["start"] + seg["duration"] - t)
            clips.append(_render_caption_image(chunk, d, color).set_start(t))
            t += d
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
                filename: str = "output.mp4", caption_segments: list[dict] = None) -> Path:
    """Create the final Shorts MP4 from an AI-image slideshow and return its path.

    If caption_segments is given (story mode), captions are rendered synced + color-coded
    per speaker; otherwise the narration is auto-split into timed chunks.
    """
    out_path = config.OUTPUT_DIR / filename

    audio = AudioFileClip(str(voice_path))
    # Cap duration slightly under the audio length to avoid moviepy's
    # MP3 over-read bug (reading frames past the decoded end).
    duration = max(0.1, audio.duration - 0.05)
    audio = audio.set_duration(duration)

    if image_paths:
        per = duration / len(image_paths)
        slides = [_ken_burns(p, per) for p in image_paths]
        bg = concatenate_videoclips(slides, method="compose").set_duration(duration)
    else:
        # Fallback: a soft vertical gradient (nicer than flat black) if no images.
        bg = ImageClip(_gradient_bg()).set_duration(duration)

    if caption_segments:
        caption_clips = _make_caption_clips_from_segments(caption_segments)
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
    vp = generate_voice(demo)
    imgs = generate_images([
        "Futuristic AI dashboard glowing on a laptop, cinematic, vertical",
        "Robot hand typing on keyboard, neon blue lighting, vertical",
    ])
    p = build_video(demo, vp, imgs)
    print(f"Saved: {p}")
