"""Daily orchestrator: generate -> voice -> video -> upload (or notify for manual)."""
import traceback
from datetime import datetime
from pathlib import Path

import config
import notifier
from content_generator import generate_content
from image_generator import generate_images
from video_builder import build_video
from voice_generator import generate_dialogue_voice, generate_voice


def _cleanup_output(keep: Path) -> None:
    """Delete every file in output/ except the given video to keep only the latest."""
    keep = keep.resolve()
    for f in config.OUTPUT_DIR.iterdir():
        if f.is_file() and f.resolve() != keep:
            try:
                f.unlink()
            except OSError as exc:  # noqa: BLE001
                print(f"[cleanup] could not delete {f.name}: {exc}")


def _format_script(content: dict) -> str:
    """Human-readable script for Telegram (dialogue with speakers, or narration)."""
    if content.get("style") == "story" and content.get("dialogue"):
        lines = []
        for turn in content["dialogue"]:
            who = "\U0001F467 Boy" if str(turn.get("speaker", "")).lower().startswith("b") else "\U0001F469 Girl"
            lines.append(f"{who}: {turn.get('line', '')}")
        return "\n".join(lines)
    return content.get("narration", "")


def _send_upload_pack(content: dict) -> None:
    """Send the YouTube metadata as SEPARATE messages so each can be copied alone."""
    title = content.get("title", "")
    description = content.get("description", "")
    tags = ", ".join(content.get("tags", []))
    hashtags = " ".join(content.get("hashtags", []))

    notifier.notify(
        "\U0001F4E6 YouTube upload pack below \u2014 each part is a separate message "
        "so you can copy them one at a time."
    )
    notifier.notify("\U0001F4CC TITLE \u2b07\ufe0f (tap & copy)")
    notifier.notify(title)
    notifier.notify("\U0001F4DD DESCRIPTION \u2b07\ufe0f (tap & copy)")
    notifier.notify(description)
    notifier.notify("\U0001F3F7\ufe0f TAGS \u2b07\ufe0f (tap & copy)")
    notifier.notify(tags or "(none)")
    notifier.notify("#\ufe0f\u20e3 HASHTAGS \u2b07\ufe0f (tap & copy)")
    notifier.notify(hashtags or "(none)")


def run() -> None:
    config.validate()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Run started {stamp} | niche: {config.NICHE} ===")

    style = config.PROMPT_STYLE
    language = config.VIDEO_LANGUAGE.capitalize()

    # Notify: run started
    notifier.notify(
        "\U0001F3AC Starting a new Short!\n"
        f"\U0001F30D Language: {language}\n"
        f"\U0001F3AD Style: {style}\n"
        f"\U0001F3F7\ufe0f Niche: {config.NICHE}"
    )

    try:
        # 1. Content
        notifier.notify("\U0001F50D Researching what's trending and writing the script...")
        content = generate_content()
        print(f"Title: {content['title']}")
        print(f"Narration: {content['narration']}")
        notifier.notify(
            f"\U0001F4A1 Topic: {content.get('topic', '')}\n"
            f"\U0001F3AF Title: {content['title']}"
        )
        notifier.notify(f"\U0001F4DD Script:\n\n{_format_script(content)}")

        # 2. Voice (story mode -> two kid voices; tips mode -> single narrator)
        caption_segments = None
        if content.get("style") == "story" and content.get("dialogue"):
            notifier.notify(
                f"\U0001F399\ufe0f Recording the story in {language} with two kid voices "
                f"(girl + boy)..."
            )
            voice_path, caption_segments = generate_dialogue_voice(
                content["dialogue"], filename=f"voice_{stamp}.mp3"
            )
        else:
            notifier.notify(f"\U0001F399\ufe0f Recording the voiceover in {language}...")
            voice_path = generate_voice(content["narration"], filename=f"voice_{stamp}.mp3")
        print(f"Voice saved: {voice_path}")
        notifier.notify("\u2705 Voiceover ready.")

        # 3. AI images (Pollinations - free, no key)
        n_scenes = len(content.get("image_prompts", []))
        notifier.notify(
            f"\U0001F5BC\ufe0f Generating {n_scenes} AI scene images (a few seconds each; "
            f"slow scenes retry on backup models)..."
        )
        image_paths = generate_images(content["image_prompts"])
        print(f"Images generated: {len(image_paths)}")
        if image_paths:
            notifier.notify(f"\u2705 Got {len(image_paths)}/{n_scenes} images. Building the video...")
        else:
            notifier.notify(
                "\u26a0\ufe0f Image service failed for all scenes \u2014 using a nice "
                "gradient background instead. Building the video..."
            )

        # 4. Video
        video_path = build_video(
            narration=content["narration"],
            voice_path=voice_path,
            image_paths=image_paths,
            filename=f"short_{stamp}.mp4",
            caption_segments=caption_segments,
        )
        print(f"Video built: {video_path}")
        notifier.notify("✅ Video built successfully! Sending it to you on Telegram...")
    except Exception as exc:  # noqa: BLE001
        print("Video creation failed:")
        traceback.print_exc()
        notifier.notify(f"❌ Failed while creating today's Short: {exc}")
        raise

    # 5. Push the finished video to Telegram (so you can watch/download it there).
    caption = f"{content['title']}\n\nTopic: {content.get('topic', '')}"
    sent = notifier.send_video(video_path, caption)
    if sent:
        print("Video sent to Telegram.")
        # Keep only the most recent video in output/ (delete intermediates + old videos).
        _cleanup_output(video_path)
        notifier.notify(
            "📲 Video delivered — download it from Telegram.\n"
            "🧹 output/ cleaned: only the latest video is kept."
        )
    else:
        print("Telegram send failed; keeping all output files so nothing is lost.")
        notifier.notify(
            f"⚠️ Couldn't send the video to Telegram. It's saved locally:\n{video_path}"
        )

    # 6. Send the YouTube upload pack as separate, easy-to-copy messages.
    _send_upload_pack(content)

    # 7. Optional auto-upload (off by default; upload manually using the pack above).
    if not config.AUTO_UPLOAD:
        print("AUTO_UPLOAD disabled - upload manually using the Telegram pack.")
        return

    try:
        from youtube_uploader import upload_video  # imported lazily so build works without OAuth

        notifier.notify("⬆️ Uploading to YouTube...")
        url = upload_video(
            video_path=video_path,
            title=content["title"],
            description=content["description"],
            tags=content.get("tags", []),
        )
        notifier.notify(f"🎉 Done! Uploaded to YouTube:\n{content['title']}\n{url}")
        print(f"Uploaded: {url}")
    except Exception as exc:  # noqa: BLE001
        print("Upload failed, falling back to manual:")
        traceback.print_exc()
        notifier.notify(
            f"⚠️ YouTube upload FAILED ({exc}).\nUse the video + pack already sent to Telegram to upload manually."
        )


if __name__ == "__main__":
    run()
