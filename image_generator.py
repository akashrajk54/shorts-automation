"""Generate AI images from text prompts using Pollinations.ai (free, no API key)."""
import hashlib
import io
import time
from pathlib import Path
from urllib.parse import quote

import config  # noqa: F401  (imported first to configure SSL trust)
import requests
from PIL import Image

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Reject obviously broken/low-quality downloads.
MIN_BYTES = 15_000
QUALITY_SUFFIX = (
    ", ultra detailed, high resolution, sharp focus, professional photography, 8k, "
    "cinematic lighting, highly detailed"
)


def _is_valid_image(data: bytes) -> bool:
    """Verify the bytes are a complete, adequately-sized image (not broken)."""
    if not data or len(data) < MIN_BYTES:
        return False
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # detects truncated/corrupt files
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        # Must be reasonably large and roughly vertical.
        return w >= config.VIDEO_WIDTH * 0.75 and h >= config.VIDEO_HEIGHT * 0.75
    except Exception:  # noqa: BLE001
        return False


def generate_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    """Download one high-quality AI-generated vertical image, retrying if broken."""
    base_seed = seed if seed is not None else int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 100000
    full_prompt = f"{prompt}{QUALITY_SUFFIX}"
    url = POLLINATIONS_URL.format(prompt=quote(full_prompt))
    dest = config.OUTPUT_DIR / f"img_{index}.jpg"

    for attempt in range(4):
        params = {
            "width": config.VIDEO_WIDTH,
            "height": config.VIDEO_HEIGHT,
            "seed": base_seed + attempt * 101,  # new seed each retry for a fresh image
            "nologo": "true",
            "enhance": "true",
            "model": "flux",
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=150)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if ctype.startswith("image") and _is_valid_image(resp.content):
                dest.write_bytes(resp.content)
                return dest
            print(f"[image] scene {index} attempt {attempt + 1}: invalid/low-quality, retrying")
        except requests.RequestException as exc:
            print(f"[image] scene {index} attempt {attempt + 1} failed: {exc}")
        time.sleep(3)
    return None


def generate_images(prompts: list[str]) -> list[Path]:
    """Generate images for a list of prompts, returning the successful paths."""
    paths: list[Path] = []
    for i, prompt in enumerate(prompts):
        path = generate_image(prompt, i)
        if path:
            print(f"[image] saved scene {i}: {path.name}")
            paths.append(path)
    return paths


if __name__ == "__main__":
    imgs = generate_images([
        "A futuristic AI robot working on a laptop, cinematic neon lighting, vertical",
        "Abstract glowing neural network, blue and purple, high tech, vertical",
    ])
    print(f"Generated {len(imgs)} images")
