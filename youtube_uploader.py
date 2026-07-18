"""Upload a video to YouTube via the Data API v3 (OAuth desktop flow)."""
from pathlib import Path

import config  # noqa: F401  (imported first to configure SSL trust)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _get_credentials() -> Credentials:
    creds = None
    if config.TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(config.TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not config.CLIENT_SECRET_FILE.exists():
                raise SystemExit(
                    "Missing client_secret.json. Create an OAuth 2.0 Desktop client in "
                    "Google Cloud Console (YouTube Data API v3) and save it here as "
                    f"{config.CLIENT_SECRET_FILE}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(config.CLIENT_SECRET_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)
        config.TOKEN_FILE.write_text(creds.to_json())
    return creds


def upload_video(video_path: Path, title: str, description: str,
                 tags: list[str], privacy: str = None) -> str:
    """Upload the video and return the YouTube video URL."""
    privacy = privacy or config.YOUTUBE_PRIVACY
    creds = _get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags,
            "categoryId": "28",  # Science & Technology
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=body, media_body=media
    )

    response = None
    while response is None:
        _, response = request.next_chunk()

    video_id = response["id"]
    return f"https://youtube.com/shorts/{video_id}"


if __name__ == "__main__":
    print("Run main.py to build then upload a video.")
