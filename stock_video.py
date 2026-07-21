"""Fetch free, vertical stock VIDEO clips for scenes (Pexels + Pixabay).

Used by the hybrid visual pipeline: when a scene has a matching real clip we use
it (real motion holds attention better); otherwise the builder falls back to an
AI image with Ken Burns motion. Everything here is best-effort - any failure
simply returns None so the pipeline keeps working with images.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config  # noqa: F401  (imported first to configure SSL trust)
import requests

PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"
PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
HTTP_TIMEOUT = 20
# Reject tiny/broken downloads (a real clip is comfortably larger than this).
MIN_BYTES = 50_000
# Keep clips reasonably short so downloads stay fast; we loop/trim to fit anyway.
MAX_CLIP_SECONDS = 30


def _clean_query(text: str, n: int = 5) -> str:
    """Turn a scene description / AI-art prompt into a short stock-footage query."""
    first = text.split(",")[0]  # drop AI-art quality suffixes
    words = [w for w in re.findall(r"[A-Za-z]+", first) if len(w) > 2]
    # Drop art-direction words that hurt stock search relevance.
    stop = {"cinematic", "vertical", "photorealistic", "detailed", "lighting",
            "resolution", "sharp", "focus", "vibrant", "highly", "ultra", "shot",
            "background", "closeup", "close", "image", "scene", "high"}
    kept = [w for w in words if w.lower() not in stop]
    return " ".join(kept[:n]) or "technology"


def _target_size_ok(w: int, h: int) -> bool:
    """Prefer portrait/vertical clips that can cover a 1080x1920 frame cleanly."""
    return h >= w and h >= 960


def _download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            data = r.content
        if len(data) < MIN_BYTES:
            return False
        dest.write_bytes(data)
        return True
    except requests.RequestException:
        return False


def _pexels_clip(query: str, index: int, dest: Path) -> bool:
    key = getattr(config, "PEXELS_API_KEY", "")
    if not key:
        return False
    try:
        r = requests.get(
            PEXELS_VIDEO_URL,
            params={"query": query, "orientation": "portrait", "size": "medium",
                    "per_page": 15, "page": 1 + (index % 3)},
            headers={"Authorization": key, **HEADERS},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        videos = r.json().get("videos", [])
        # Rotate which result we take so consecutive scenes don't reuse one clip.
        for off in range(len(videos)):
            vid = videos[(index + off) % len(videos)]
            if vid.get("duration", 0) > MAX_CLIP_SECONDS:
                continue
            files = vid.get("video_files", [])
            # Pick the best portrait file at/around 1080 wide, mp4 only.
            portrait = [f for f in files if f.get("file_type") == "video/mp4"
                        and _target_size_ok(f.get("width", 0), f.get("height", 0))]
            portrait.sort(key=lambda f: abs((f.get("width") or 0) - config.VIDEO_WIDTH))
            for f in portrait:
                if f.get("link") and _download(f["link"], dest):
                    print(f"[stockvid] scene {index}: pexels clip ('{query}')")
                    return True
    except requests.RequestException as exc:
        print(f"[stockvid] scene {index}: pexels failed: {exc}")
    return False


def _pixabay_clip(query: str, index: int, dest: Path) -> bool:
    key = getattr(config, "PIXABAY_API_KEY", "")
    if not key:
        return False
    try:
        r = requests.get(
            PIXABAY_VIDEO_URL,
            params={"key": key, "q": query, "per_page": 15, "safesearch": "true"},
            headers=HEADERS, timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        hits = r.json().get("hits", [])
        for off in range(len(hits)):
            hit = hits[(index + off) % len(hits)]
            if hit.get("duration", 0) > MAX_CLIP_SECONDS:
                continue
            streams = hit.get("videos", {})
            # Prefer larger vertical-friendly renditions.
            for quality in ("large", "medium", "small"):
                v = streams.get(quality) or {}
                w, h = v.get("width", 0), v.get("height", 0)
                if v.get("url") and _target_size_ok(w, h) and _download(v["url"], dest):
                    print(f"[stockvid] scene {index}: pixabay clip ('{query}')")
                    return True
    except requests.RequestException as exc:
        print(f"[stockvid] scene {index}: pixabay failed: {exc}")
    return False


def fetch_stock_video(query_text: str, index: int) -> Path | None:
    """Fetch ONE vertical stock clip for a scene, or None if nothing suitable."""
    query = _clean_query(query_text)
    dest = config.OUTPUT_DIR / f"clip_{index}.mp4"
    if _pexels_clip(query, index, dest) or _pixabay_clip(query, index, dest):
        return dest
    return None


def fetch_stock_videos(queries: list[str], on_progress=None) -> dict[int, Path]:
    """Fetch stock clips for scenes in parallel. Returns {scene_index: clip_path}
    for scenes that got a clip; scenes without one fall back to an AI image."""
    if not getattr(config, "STOCK_VIDEO", False):
        return {}
    if not (getattr(config, "PEXELS_API_KEY", "") or getattr(config, "PIXABAY_API_KEY", "")):
        print("[stockvid] no Pexels/Pixabay key set - skipping stock video (images only)")
        return {}

    results: dict[int, Path] = {}
    workers = min(4, max(1, len(queries)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_stock_video, q, i): i for i, q in enumerate(queries)}
        done = 0
        for fut in futures:
            i = futures[fut]
            try:
                path = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[stockvid] scene {i} errored: {exc}")
                path = None
            done += 1
            if path:
                results[i] = path
            if on_progress is not None:
                try:
                    on_progress(done, len(queries), i, path is not None)
                except Exception:  # noqa: BLE001
                    pass
    print(f"[stockvid] got {len(results)}/{len(queries)} stock clips")
    return results


if __name__ == "__main__":
    clips = fetch_stock_videos([
        "person typing on laptop", "smartphone app interface", "excited person phone",
    ])
    print(clips)
