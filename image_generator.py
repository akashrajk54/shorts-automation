"""Generate AI images from text prompts using Pollinations.ai (free, no API key)."""
import hashlib
import io
import random
import time
from pathlib import Path
from urllib.parse import quote

import config  # noqa: F401  (imported first to configure SSL trust)
import requests
from PIL import Image

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
# Free fallback when Pollinations fails/runs out of Pollen (works from cloud IPs).
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://shorts-automation.local",
}


def _auth_headers() -> dict:
    """Add a Bearer token if configured (raises Pollinations rate limits)."""
    h = dict(HEADERS)
    token = getattr(config, "POLLINATIONS_TOKEN", "")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

# Reject obviously broken downloads, but don't be overly strict.
MIN_BYTES = 8_000
# When rate-limited, Pollinations sometimes returns HTTP 200 with a ~1.3MB
# placeholder image instead of a real one. Reject it by its known MD5 so we
# retry/back off instead of shipping a broken frame.
RATE_LIMIT_HASHES = {"2090a5dc21c32952cbf8496339752bd1"}
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
    """Verify the bytes are a complete, reasonably-sized image (not broken and not
    the rate-limit placeholder Pollinations serves with HTTP 200)."""
    if not data or len(data) < MIN_BYTES:
        return False
    if hashlib.md5(data).hexdigest() in RATE_LIMIT_HASHES:
        return False  # rate-limit placeholder disguised as a real 200 response
    try:
        Image.open(io.BytesIO(data)).verify()  # detects truncated/corrupt files
        w, h = Image.open(io.BytesIO(data)).size
        return w >= 512 and h >= 512
    except Exception:  # noqa: BLE001
        return False


def generate_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    """Download one vertical image, trying multiple providers/seeds until one works.

    Pollinations' free tier rate-limits bursts (HTTP 429), so we back off
    exponentially (with jitter) on 429 and cycle through backup models.
    """
    base_seed = seed if seed is not None else int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 100000
    url = POLLINATIONS_URL.format(prompt=quote(f"{prompt}{QUALITY_SUFFIX}"))
    dest = config.OUTPUT_DIR / f"img_{index}.jpg"
    timeout = getattr(config, "IMAGE_TIMEOUT", 90)

    attempt = 0
    backoff = 4.0  # grows on repeated 429s
    for provider in PROVIDERS:
        for retry in range(3):
            attempt += 1
            params = {
                "width": config.VIDEO_WIDTH,
                "height": config.VIDEO_HEIGHT,
                "seed": base_seed + attempt * 101,  # fresh image each try
                "nologo": "true",
                "model": provider["model"],
                "referrer": getattr(config, "POLLINATIONS_REFERRER", "shorts-automation"),
            }
            token = getattr(config, "POLLINATIONS_TOKEN", "")
            if token:
                # Send under both param names for compatibility: newer Pollinations
                # uses ?key=, older builds used ?token= (Bearer header also set).
                params["key"] = token
                params["token"] = token
            if provider["enhance"]:
                params["enhance"] = "true"
            try:
                start = time.time()
                resp = requests.get(url, params=params, headers=_auth_headers(), timeout=timeout)
                # 402 = "Queue full for IP" (per-IP rate limit); treat like 429.
                if resp.status_code in (429, 402):
                    wait = backoff + random.uniform(0, 2)
                    backoff = min(backoff * 2, 40)
                    print(f"[image] scene {index} ({provider['model']}): rate-limited "
                          f"({resp.status_code}), waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
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
    print(f"[image] scene {index}: all Pollinations providers failed")
    return _fallback_image(prompt, index, base_seed)


# --- Free, KEYLESS image fallbacks (work from cloud/CI IPs, zero signup) ---
# A descriptive UA is required by Wikimedia and appreciated by the others.
_API_UA = "shorts-automation/1.0 (+https://github.com/akashrajk54/shorts-automation)"


def _keywords(prompt: str, n: int = 6) -> str:
    """Reduce a long AI-art prompt to a short keyword query for stock/CC search."""
    first_clause = prompt.split(",")[0]  # drop the quality suffix / extra clauses
    words = [w for w in first_clause.split() if w.isalpha() and len(w) > 2]
    return " ".join(words[:n]) or "technology"


def _download_valid(img_url: str, dest: Path, timeout: int) -> bool:
    """Download a URL and keep it only if it is a real, complete image."""
    try:
        data = requests.get(img_url, headers={"User-Agent": _API_UA}, timeout=timeout).content
        if _is_valid_image(data):
            dest.write_bytes(data)
            return True
    except requests.RequestException:
        pass
    return False


def _pexels_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    """Optional Pexels stock photo (only used if PEXELS_API_KEY is set)."""
    key = getattr(config, "PEXELS_API_KEY", "")
    if not key:
        return None
    query = _keywords(prompt)
    dest = config.OUTPUT_DIR / f"img_{index}.jpg"
    timeout = getattr(config, "IMAGE_TIMEOUT", 90)
    try:
        resp = requests.get(
            PEXELS_SEARCH_URL,
            params={"query": query, "orientation": "portrait", "per_page": 15,
                    "page": 1 + (index % 3)},
            headers={"Authorization": key},
            timeout=timeout,
        )
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        if photos:
            photo = photos[(index + (seed or 0)) % len(photos)]
            src = photo.get("src", {})
            img_url = src.get("original") or src.get("portrait") or src.get("large2x")
            if img_url and _download_valid(img_url, dest, timeout):
                print(f"[image] scene {index}: ok via pexels ('{query}')")
                return dest
    except requests.RequestException as exc:
        print(f"[image] scene {index}: pexels failed: {exc}")
    return None


def _openverse_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    """Keyword search over openly-licensed images (keyless, cloud-friendly)."""
    query = _keywords(prompt)
    dest = config.OUTPUT_DIR / f"img_{index}.jpg"
    timeout = getattr(config, "IMAGE_TIMEOUT", 90)
    try:
        r = requests.get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": 20, "aspect_ratio": "tall", "mature": "false"},
            headers={"User-Agent": _API_UA}, timeout=timeout,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        for off in range(len(results)):
            hit = results[(index + (seed or 0) + off) % len(results)]
            # thumbnail is Openverse-proxied (no hotlink blocks); url is full-res.
            for img_url in (hit.get("thumbnail"), hit.get("url")):
                if img_url and _download_valid(img_url, dest, timeout):
                    print(f"[image] scene {index}: ok via openverse ('{query}')")
                    return dest
    except requests.RequestException as exc:
        print(f"[image] scene {index}: openverse failed: {exc}")
    return None


def _wikimedia_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    """Keyword search over Wikimedia Commons (keyless, cloud-friendly)."""
    query = _keywords(prompt, 4)
    dest = config.OUTPUT_DIR / f"img_{index}.jpg"
    timeout = getattr(config, "IMAGE_TIMEOUT", 90)
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={"action": "query", "generator": "search", "gsrsearch": query,
                    "gsrnamespace": 6, "gsrlimit": 10, "prop": "imageinfo",
                    "iiprop": "url", "iiurlwidth": config.VIDEO_WIDTH, "format": "json"},
            headers={"User-Agent": _API_UA}, timeout=timeout,
        )
        r.raise_for_status()
        pages = list(r.json().get("query", {}).get("pages", {}).values())
        urls = [p["imageinfo"][0].get("thumburl") for p in pages if p.get("imageinfo")]
        urls = [u for u in urls if u]
        for off in range(len(urls)):
            u = urls[(index + (seed or 0) + off) % len(urls)]
            if _download_valid(u, dest, timeout):
                print(f"[image] scene {index}: ok via wikimedia ('{query}')")
                return dest
    except requests.RequestException as exc:
        print(f"[image] scene {index}: wikimedia failed: {exc}")
    return None


def _picsum_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    """Guaranteed keyless image: a real (random) photo, always available."""
    dest = config.OUTPUT_DIR / f"img_{index}.jpg"
    timeout = getattr(config, "IMAGE_TIMEOUT", 90)
    s = (seed or 0) + index
    url = f"https://picsum.photos/seed/{s}/{config.VIDEO_WIDTH}/{config.VIDEO_HEIGHT}"
    if _download_valid(url, dest, timeout):
        print(f"[image] scene {index}: ok via picsum (seed {s})")
        return dest
    return None


# Fallback order (only reached if Pollinations AI generation fails):
#   1. Pexels  - the "earlier" key-based source. Best prompt match, but only runs
#      if PEXELS_API_KEY is set; otherwise it self-skips instantly (returns None),
#      so the pipeline still works with no keys at all.
#   2. Wikimedia Commons - keyless, keyword-relevant, reliable from cloud IPs.
#   3. Openverse - keyless, keyword-relevant.
#   4. Picsum - keyless, always-available backstop so we never ship a gradient.
# This means: earlier (key-based) version is preferred when a key is provided;
# without a key it falls straight through to the current keyless version.
_FALLBACK_SOURCES = (_pexels_image, _wikimedia_image, _openverse_image, _picsum_image)


def _fallback_image(prompt: str, index: int, seed: int | None = None) -> Path | None:
    for source in _FALLBACK_SOURCES:
        try:
            path = source(prompt, index, seed=seed)
        except Exception as exc:  # noqa: BLE001  (never let one source break the chain)
            print(f"[image] scene {index}: {source.__name__} errored: {exc}")
            path = None
        if path:
            return path
    return None


def generate_images(prompts: list[str], on_image=None) -> list[Path]:
    """Generate scene images with LIMITED concurrency (staggered) to dodge 429s.

    Full parallelism trips Pollinations' rate limit, so we use a small worker
    pool and stagger request starts; each worker also backs off on 429.

    on_image: optional callback(done, total, index, ok) fired as each scene
    finishes, so callers can report live progress (e.g. to Telegram).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(prompts)
    workers = min(getattr(config, "MAX_IMAGE_WORKERS", 2), max(1, total))

    def _staggered(prompt: str, i: int) -> Path | None:
        time.sleep(i * 1.5)  # gentle stagger so requests don't all hit at once
        return generate_image(prompt, i)

    results: dict[int, Path] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_staggered, p, i): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            i = futures[future]
            path = future.result()
            done += 1
            if path:
                print(f"[image] saved scene {i}: {path.name}")
                results[i] = path
            if on_image is not None:
                try:
                    on_image(done, total, i, path is not None)
                except Exception:  # noqa: BLE001  (never let progress reporting break generation)
                    pass
    # Preserve scene order for a coherent slideshow.
    return [results[i] for i in sorted(results)]


if __name__ == "__main__":
    imgs = generate_images([
        "A futuristic AI robot working on a laptop, cinematic neon lighting, vertical",
        "Abstract glowing neural network, blue and purple, high tech, vertical",
    ])
    print(f"Generated {len(imgs)} images")
