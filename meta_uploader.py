"""Cross-post the finished Short to Facebook (and Instagram) as a Reel.

Uses the Meta Graph API's resumable upload, so no public hosting of the MP4 is
needed - we upload the local file bytes directly. Everything is best-effort:
any failure raises, and the caller (main.py) catches it so YouTube/Telegram
delivery is never blocked by a Meta error.

Setup (see config.py):
  POST_TO_FACEBOOK=true, FB_PAGE_ID, FB_PAGE_TOKEN (permanent Page token)
  POST_TO_INSTAGRAM=true, IG_USER_ID (uses the same FB_PAGE_TOKEN)
"""
import time
from pathlib import Path

import config  # noqa: F401  (imported first to configure SSL trust)
import requests

GRAPH = "https://graph.facebook.com/{ver}"
RUPLOAD = "https://rupload.facebook.com"
HTTP_TIMEOUT = 120
# How long to wait for Meta to finish processing the uploaded reel.
PROCESS_TIMEOUT = 180
POLL_EVERY = 5


def _graph(path: str) -> str:
    return f"{GRAPH.format(ver=config.META_GRAPH_VERSION)}/{path}"


def _build_caption(content: dict) -> str:
    """Compose a Reel caption from the generated content (title + hashtags)."""
    title = (content.get("title") or "").strip()
    desc = (content.get("description") or "").strip()
    hashtags = content.get("hashtags") or []
    tags = " ".join(h if h.startswith("#") else f"#{h}" for h in hashtags)
    parts = [p for p in (title, desc, tags) if p]
    return "\n\n".join(parts)[:2200]  # IG caption hard limit is 2200 chars


# --- Facebook Reels ------------------------------------------------------------
def post_facebook_reel(video_path: Path, caption: str) -> str:
    """Upload + publish a Facebook Reel to the Page. Returns the reel URL."""
    page_id = config.FB_PAGE_ID
    token = config.FB_PAGE_TOKEN
    if not (page_id and token):
        raise ValueError("FB_PAGE_ID / FB_PAGE_TOKEN not configured")

    size = Path(video_path).stat().st_size

    # 1) Start an upload session -> get a video_id + upload_url.
    start = requests.post(
        _graph(f"{page_id}/video_reels"),
        params={"upload_phase": "start", "access_token": token},
        timeout=HTTP_TIMEOUT,
    )
    start.raise_for_status()
    sj = start.json()
    video_id = sj["video_id"]
    upload_url = sj.get("upload_url") or f"{RUPLOAD}/video-reels/{video_id}"

    # 2) Upload the raw bytes to the resumable endpoint.
    with open(video_path, "rb") as f:
        data = f.read()
    up = requests.post(
        upload_url,
        headers={"Authorization": f"OAuth {token}", "offset": "0",
                 "file_size": str(size)},
        data=data,
        timeout=HTTP_TIMEOUT,
    )
    up.raise_for_status()
    if not up.json().get("success", True):
        raise RuntimeError(f"FB reel byte upload failed: {up.text}")

    # 3) Finish + publish.
    finish = requests.post(
        _graph(f"{page_id}/video_reels"),
        params={"upload_phase": "finish", "video_id": video_id,
                "video_state": "PUBLISHED", "description": caption,
                "access_token": token},
        timeout=HTTP_TIMEOUT,
    )
    finish.raise_for_status()

    _wait_fb_ready(video_id, token)
    return f"https://www.facebook.com/reel/{video_id}"


def _wait_fb_ready(video_id: str, token: str) -> None:
    """Poll the reel's processing status until it's published (best-effort)."""
    deadline = time.time() + PROCESS_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(
                _graph(video_id),
                params={"fields": "status", "access_token": token},
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            status = (r.json().get("status") or {})
            phase = status.get("video_status") or status.get("processing_phase", {}).get("status")
            if phase in ("ready", "PUBLISHED", "published", "complete"):
                return
            if phase in ("error", "ERROR"):
                raise RuntimeError(f"FB reel processing error: {status}")
        except requests.RequestException:
            pass
        time.sleep(POLL_EVERY)
    # Timed out waiting - the reel is usually still processing fine, so don't fail.


# --- Instagram Reels (ready for later; needs an IG Business account) -----------
def post_instagram_reel(video_path: Path, caption: str) -> str:
    """Upload + publish an Instagram Reel via resumable upload. Returns media URL."""
    ig_id = config.IG_USER_ID
    token = config.FB_PAGE_TOKEN
    if not (ig_id and token):
        raise ValueError("IG_USER_ID / FB_PAGE_TOKEN not configured")

    size = Path(video_path).stat().st_size

    # 1) Create a REELS container using resumable upload.
    create = requests.post(
        _graph(f"{ig_id}/media"),
        params={"media_type": "REELS", "upload_type": "resumable",
                "caption": caption, "access_token": token},
        timeout=HTTP_TIMEOUT,
    )
    create.raise_for_status()
    cj = create.json()
    container_id = cj["id"]
    upload_uri = cj.get("uri") or f"{RUPLOAD}/ig-api-upload/{config.META_GRAPH_VERSION}/{container_id}"

    # 2) Upload the raw bytes.
    with open(video_path, "rb") as f:
        data = f.read()
    up = requests.post(
        upload_uri,
        headers={"Authorization": f"OAuth {token}", "offset": "0",
                 "file_size": str(size)},
        data=data,
        timeout=HTTP_TIMEOUT,
    )
    up.raise_for_status()

    # 3) Wait for the container to finish processing, then publish.
    _wait_ig_ready(container_id, token)
    publish = requests.post(
        _graph(f"{ig_id}/media_publish"),
        params={"creation_id": container_id, "access_token": token},
        timeout=HTTP_TIMEOUT,
    )
    publish.raise_for_status()
    media_id = publish.json().get("id", "")
    return f"https://www.instagram.com/reel/{media_id}" if media_id else "instagram: published"


def _wait_ig_ready(container_id: str, token: str) -> None:
    """Poll IG container status until FINISHED before publishing."""
    deadline = time.time() + PROCESS_TIMEOUT
    while time.time() < deadline:
        r = requests.get(
            _graph(container_id),
            params={"fields": "status_code,status", "access_token": token},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        code = r.json().get("status_code")
        if code == "FINISHED":
            return
        if code == "ERROR":
            raise RuntimeError(f"IG reel processing error: {r.json()}")
        time.sleep(POLL_EVERY)
    raise TimeoutError("IG reel did not finish processing in time")


def cross_post(video_path: Path, content: dict) -> dict:
    """Post to whichever platforms are enabled. Returns {platform: url_or_error}."""
    caption = _build_caption(content)
    results: dict[str, str] = {}
    if config.POST_TO_FACEBOOK:
        try:
            results["facebook"] = post_facebook_reel(video_path, caption)
            print(f"[meta] facebook reel: {results['facebook']}")
        except Exception as exc:  # noqa: BLE001
            results["facebook"] = f"ERROR: {exc}"
            print(f"[meta] facebook failed: {exc}")
    if config.POST_TO_INSTAGRAM:
        try:
            results["instagram"] = post_instagram_reel(video_path, caption)
            print(f"[meta] instagram reel: {results['instagram']}")
        except Exception as exc:  # noqa: BLE001
            results["instagram"] = f"ERROR: {exc}"
            print(f"[meta] instagram failed: {exc}")
    return results


if __name__ == "__main__":
    print("Run main.py to build then cross-post a video.")
