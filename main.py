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


def _build_upload_pack(content: dict) -> str:
    """Format a copy-paste-ready YouTube upload pack for manual uploading."""
    hashtags = " ".join(content.get("hashtags", []))
    tags = ", ".join(content.get("tags", []))
    description = content.get("description", "")
    full_description = f"{description}\n\n{hashtags}".strip()
    return (
        "🎬 YOUR YOUTUBE UPLOAD PACK (copy-paste ready)\n"
        "───────────────\n"
        f"📌 TITLE:\n{content.get('title', '')}\n\n"
        f"📝 DESCRIPTION:\n{full_description}\n\n"
        f"🏷️ TAGS:\n{tags}\n\n"
        f"#️⃣ HASHTAGS:\n{hashtags}\n"
        "───────────────\n"
        "Tip: Upload as a Short, use the title above, paste the description, "
        "add the tags, and post at a peak hour for best reach."
    )


def run() -> None:
    config.validate()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Run started {stamp} | niche: {config.NICHE} ===")

    # Notify: run started
    notifier.notify(f"🎬 Started creating today's Short\nNiche: {config.NICHE}")

    try:
        # 1. Content
        content = generate_content()
        print(f"Title: {content['title']}")
        print(f"Narration: {content['narration']}")
        notifier.notify(
            f"📝 Topic chosen: {content['title']}\n\n"
            f"Script:\n{content['narration']}\n\nGenerating voiceover next..."
        )

        # 2. Voice (story mode -> two kid voices; tips mode -> single narrator)
        caption_segments = None
        if content.get("style") == "story" and content.get("dialogue"):
            voice_path, caption_segments = generate_dialogue_voice(
                content["dialogue"], filename=f"voice_{stamp}.mp3"
            )
        else:
            voice_path = generate_voice(content["narration"], filename=f"voice_{stamp}.mp3")
        print(f"Voice saved: {voice_path}")
        notifier.notify("🎙️ Voiceover ready. Generating AI images...")

        # 3. AI images (Pollinations - free, no key)
        image_paths = generate_images(content["image_prompts"])
        print(f"Images generated: {len(image_paths)}")
        notifier.notify(f"🖼️ Generated {len(image_paths)} AI images. Building the video...")

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

    # 6. Send the copy-paste-ready YouTube upload pack (title/description/tags/hashtags).
    notifier.notify(_build_upload_pack(content))

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
