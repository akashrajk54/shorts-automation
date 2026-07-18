# AI YouTube Shorts Automation — AI Tools & Tech Tips

Fully automated, **near-zero-cost** daily pipeline:

1. **Generate** a 12–15s script + title/description/tags + scene image prompts with **Google Gemini** (free tier)
2. **Voice** it with **edge-tts** (free, natural TTS — no API key)
3. **Images** generated with **Pollinations.ai** (free AI images, no API key)
4. **Build** a vertical 1080×1920 Short: Ken-Burns image slideshow + burned-in captions + voiceover (`moviepy`)
5. **Upload** to YouTube via **Data API v3** — or, on failure / if disabled, save the MP4 and ping you on **Telegram** to upload manually
6. **Schedule** it daily with `cron`

---

## 1. Prerequisites

- **Python 3.10+**
- **ffmpeg** — *not required to install separately*; moviepy uses the bundled
  `imageio-ffmpeg` binary automatically. (You can still `brew install ffmpeg` if you prefer the system one.)
- **ImageMagick** — *not required*. Captions are rendered with **Pillow**, so there's no ImageMagick dependency.

## 2. Install

```bash
cd /Users/akashbhandari/Desktop/projects/youtube-shorts-automation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure keys

```bash
cp .env.example .env
```

Fill in `.env`:

- `GEMINI_API_KEY` — free at https://aistudio.google.com/app/apikey (only required key)
- `PEXELS_API_KEY` — not needed; visuals use free Pollinations AI images
- `NICHE` — already set to `AI tools and tech tips`
- `AUTO_UPLOAD` — `true` to auto-upload, `false` to only build for manual upload
- `YOUTUBE_PRIVACY` — start with `private` for testing, switch to `public` later
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — optional, for notifications & manual-upload delivery

### Telegram (optional but recommended)
1. Message **@BotFather** → `/newbot` → copy the token
2. Message your new bot once, then visit
   `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your `chat.id`

## 4. YouTube upload setup (one-time)

Only needed if `AUTO_UPLOAD=true`.

1. Go to **Google Cloud Console** → create/select a project
2. Enable **YouTube Data API v3**
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
   - Application type: **Desktop app**
4. Download the JSON and save it here as **`client_secret.json`**
5. First run opens a browser to authorize; a `token.json` is cached for future runs.

> Note: The API allows a limited number of uploads/day per project (quota ~6 uploads). That's plenty for one daily Short.

## 5. Test each piece

```bash
python content_generator.py   # prints a JSON script + image prompts
python voice_generator.py     # writes output/voice.mp3
python image_generator.py     # writes output/img_*.jpg (free AI images)
python video_builder.py       # writes output/output.mp4 (slideshow)
```

## 6. Run the full pipeline

```bash
python main.py
```

- Success → uploaded (or saved locally if `AUTO_UPLOAD=false`)
- Any failure → MP4 saved in `output/` + Telegram alert for manual upload

## 7. Schedule it daily (cron)

Edit your crontab:

```bash
crontab -e
```

Add this line to run **every day at 10:00 AM** (adjust the path to your venv python):

```cron
0 10 * * * cd /Users/akashbhandari/Desktop/projects/youtube-shorts-automation && ./venv/bin/python main.py >> output/run.log 2>&1
```

Cron format: `minute hour day month weekday`. Examples:
- `0 9 * * *` → 9:00 AM daily
- `30 18 * * *` → 6:30 PM daily

> On macOS, ensure your Mac is awake at that time (cron won't wake a sleeping Mac). For always-on runs, use a small cloud VM or GitHub Actions instead.

---

## Cost summary

| Component | Service | Cost |
|-----------|---------|------|
| Script | Gemini 1.5 Flash | Free tier |
| Voice | edge-tts | Free |
| Footage | Pexels | Free |
| Assembly | ffmpeg/moviepy | Free |
| Upload | YouTube Data API | Free |
| Scheduling | cron | Free |

**≈ $0/month.**

## Monetization reality check

YouTube pays ad revenue only after you hit **1,000 subscribers + 10M valid Shorts views in 90 days** (or 4,000 watch hours long-form). Focus on consistency and hooks first — the money follows growth.

## Troubleshooting: SSL certificate errors (corporate networks)

If you see `CERTIFICATE_VERIFY_FAILED` / `unable to get local issuer certificate`,
you're likely behind a corporate HTTPS-inspection proxy (e.g. **Zscaler**). This
project uses the **`truststore`** package (see `config.py`) to trust the same root
CAs as your macOS keychain, which fixes it automatically. `config.py` is imported
first in every module so the SSL trust is configured before any network call.

## Customization tips

- Change the **voice** in `voice_generator.py` (`DEFAULT_VOICE`), e.g. `en-US-GuyNeural`, `en-IN-PrabhatNeural`
- Tune the **caption style** (font, size, position) in `video_builder.py` → `_make_caption`
- Adjust the **script style/length** in `content_generator.py` → `PROMPT_TEMPLATE`
- Post more than once/day by triggering `main.py` at multiple cron times
