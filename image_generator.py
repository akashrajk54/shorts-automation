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
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://shorts-automation.local",
}

# Reject obviously broken downloads, but don't be overly strict.
MIN_BYTES = 8_000
QUALITY_SUFFIX = (
    ", ultra detailed, high resolution, sharp focus, cinematic lighting, "
    "vibrant colors, highly detailed, 4k"
)

# Providers tried in order per scene: fast + reliable first, quality-boost last.
# Each is a Pollinations model variant (all free, no API key).
PROVIDERS = (
    {"model": "flux", "enhance": False},
    {"model": "turbo", "enhance": False},
    {"model": "flux", "enhance": True},
)


def _is_valid_image(data: bytes) -> bool:
    """Verify the bytes are a complete, reasonably-sized image (not broken)."""
    if not data or len(data) < MIN_BYTES:
        return False
    try:
        Image.open(io.BytesIO(data)).verify()  # detects truncated/corrupt files
        w, h = Image.open(io.BytesIO(data)).size
        return w >= 512 and h >= 512
    except Exception:  # noqa: BLE001
        return False


def generate_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    """Download one vertical image, trying multiple providers/seeds until one works."""
    base_seed = seed if seed is not None else int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 100000
    url = POLLINATIONS_URL.format(prompt=quote(f"{prompt}{QUALITY_SUFFIX}"))
    dest = config.OUTPUT_DIR / f"img_{index}.jpg"
    timeout = getattr(config, "IMAGE_TIMEOUT", 90)

    attempt = 0
    for provider in PROVIDERS:
        for retry in range(2):
            attempt += 1
            params = {
                "width": config.VIDEO_WIDTH,
                "height": config.VIDEO_HEIGHT,
                "seed": base_seed + attempt * 101,  # fresh image each try
                "nologo": "true",
                "model": provider["model"],
                "referrer": "shorts-automation",
            }
            if provider["enhance"]:
                params["enhance"] = "true"
            try:
                start = time.time()
                resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if ctype.startswith("image") and _is_valid_image(resp.content):
                    dest.write_bytes(resp.content)
                    print(f"[image] scene {index}: ok via {provider['model']}"
                          f"{'+enhance' if provider['enhance'] else ''} in {time.time() - start:.1f}s")
                    return dest
                print(f"[image] scene {index} ({provider['model']}) try {retry + 1}: "
                      f"invalid response ({ctype}, {len(resp.content)}B), retrying")
            except requests.RequestException as exc:
                print(f"[image] scene {index} ({provider['model']}) try {retry + 1} failed: {exc}")
            time.sleep(2)
    print(f"[image] scene {index}: all providers failed")
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
