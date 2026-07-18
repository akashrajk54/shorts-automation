"""Central configuration loaded from environment / .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- SSL trust: use the OS keychain (handles corporate proxies like Zscaler) ---
# This must run BEFORE any library (edge-tts/aiohttp/requests) builds its SSL
# context, so Python trusts the same root CAs your macOS system does.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    # Fallback: point at certifi's bundle (works only without SSL inspection).
    try:
        import certifi

        _ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", _ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
    except ImportError:
        pass

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Pexels is optional/legacy; visuals now come from free AI images (Pollinations).
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()

# --- Content ---
NICHE = os.getenv("NICHE", "AI tools and tech tips").strip()
# Prompt/video style: "tips" (single-narrator tip) or "story" (two kids dialogue).
PROMPT_STYLE = os.getenv("PROMPT_STYLE", "tips").strip().lower()

# --- Upload ---
AUTO_UPLOAD = os.getenv("AUTO_UPLOAD", "true").strip().lower() == "true"
YOUTUBE_PRIVACY = os.getenv("YOUTUBE_PRIVACY", "private").strip().lower()

# --- Telegram fallback ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# --- Video settings (YouTube Shorts = vertical 1080x1920) ---
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30

# --- Background music ---
# Drop any royalty-free .mp3/.wav files into assets/music/ and they'll be mixed
# softly under the voiceover. Set BACKGROUND_MUSIC=false to disable.
ASSETS_DIR = BASE_DIR / "assets"
MUSIC_DIR = ASSETS_DIR / "music"
BACKGROUND_MUSIC = os.getenv("BACKGROUND_MUSIC", "true").strip().lower() == "true"
try:
    MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.12"))
except ValueError:
    MUSIC_VOLUME = 0.12

# YouTube OAuth files
CLIENT_SECRET_FILE = BASE_DIR / "client_secret.json"
TOKEN_FILE = BASE_DIR / "token.json"


def validate():
    """Raise a helpful error if required keys are missing."""
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nCopy .env.example to .env and fill in the values."
        )
