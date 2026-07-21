"""Generate a short-form video script + metadata using Google Gemini (free tier)."""
import json
import os
import random
import re
import time

import config  # noqa: F401  (imported first to configure SSL trust)
import google.generativeai as genai
import history
from trend_finder import get_trending

# Models tried in order. The default primary (gemini-2.0-flash) has a far more
# generous free tier than gemini-flash-latest (which now maps to gemini-3.5-flash
# at only 5 requests/min). If one is rate-limited (429), we fall through to the
# next model so a single quota hit no longer kills the whole run.
GEMINI_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_MODELS",
        "gemini-2.0-flash,gemini-2.5-flash,gemini-flash-latest,gemini-1.5-flash",
    ).split(",") if m.strip()
]

TIPS_PROMPT_TEMPLATE = """You are a world-class viral YouTube Shorts scriptwriter and researcher
for the niche: "{niche}".

Today's CURRENTLY TRENDING topics/headlines (use these for inspiration to ride the
wave of what people care about RIGHT NOW - pick or adapt the most relevant, genuinely
useful and viral-worthy angle for the niche; ignore irrelevant ones):
{trends}

DO NOT repeat or closely resemble any of these RECENT videos we already made
(choose a clearly different tool/topic/angle for real variety):
{history}

Create ONE COMPLETE, self-contained short video for today. Target a 20-25 second
narration (about 55-70 spoken words) that hooks the viewer in the first 2 seconds
and delivers a FULL, useful tip from start to finish.

CRITICAL CONTENT RULES:
- Make it something REAL humans actually want: solve a real problem, reveal a
  genuinely useful tool/trick, or share a surprising, accurate insight people love.
- Be SPECIFIC and COMPLETE. NEVER say vague things like "this tool" or "this app"
  without NAMING it. Always name the actual tool/app/website/method.
- Give the viewer everything they need: WHAT it is, WHAT it does, and exactly HOW
  to use it (concrete steps or a clear example). No cliffhangers, no missing info.
- Be genuinely CREATIVE and VARIED. Vary the hook and format across videos
  (e.g. a hidden feature, a tool comparison, a step-by-step workflow, a myth-buster,
  a "you're doing X wrong", a mind-blowing use-case). Do NOT always start the same way.
- End with a clear takeaway or one-line call to action.
- Use natural, conversational, energetic spoken English with contractions.
- Use ONLY real, accurate tools/facts. Never invent fake tools or fake claims.

Return ONLY valid JSON (no markdown, no code fences) with these exact keys:
{{
  "topic": "2-4 word label of the main tool/subject (for de-duplication), e.g. 'Gamma AI slides'",
  "title": "catchy YouTube title under 70 chars, name the tool, include 1 emoji",
  "narration": "the COMPLETE 55-70 word spoken script (names the tool, explains what it does and how to use it, ends with a takeaway)",
  "description": "2-3 sentence YouTube description that names the tool and summarizes the tip",
  "tags": ["8-12", "lowercase", "hashtag-free", "keywords"],
  "hashtags": ["5-8", "#shorts", "#relevant", "#viral", "hashtags with the # sign"],
  "image_prompts": [
    "ONE detailed AI image prompt for EVERY sentence of the narration, IN ORDER",
    "(so if the narration has 6 sentences, return 6 prompts). Each prompt must",
    "visually match that specific sentence's moment. Vivid, modern, high-quality",
    "vertical (9:16), cinematic lighting, photorealistic, NO text/words in image."
  ]
}}

IMPORTANT: 'image_prompts' MUST have exactly ONE entry per sentence in 'narration',
in the same order.
"""


STORY_PROMPT_TEMPLATE = """You are a beloved children's story writer AND a viral YouTube Shorts
creator for the niche: "{niche}".

Write ONE short, delightful STORY in the form of a natural conversation between
TWO cute little kids:
- a smart, sweet, confident GIRL (she is knowledgeable and kindly teaches),
- a curious, funny BOY (he asks questions and learns).

The girl teaches the boy something genuinely USEFUL about being smart with AI -
a specific, real, accurate AI tool or trick. It must feel like a lovable little
story that is a joy to listen to: warm, playful, wholesome, and easy to follow -
while still delivering COMPLETE, correct, useful information the viewer can act on.

FRESHNESS DIRECTIVE FOR TODAY (follow this to make the story feel NEW, not a repeat):
{variety}

Today's CURRENTLY TRENDING topics/headlines (use for inspiration, pick the most
relevant + genuinely useful AI angle; ignore irrelevant ones):
{trends}

DO NOT repeat or closely resemble any of these RECENT videos we already made
(pick a clearly different AI tool/topic for real variety):
{history}

RULES:
- 6 to 9 short dialogue turns, alternating girl/boy, ~70-90 spoken words total
  (keep the whole thing under ~35 seconds when read aloud).
- Kids talk naturally and adorably (short sentences, contractions, a little humor),
  but the ACTUAL tip must be specific and complete: NAME the real AI tool and say
  exactly WHAT it does and HOW to use it. No vague "this tool" without naming it.
- VARIETY IS CRITICAL: do NOT reuse the same setup every time. AVOID the tired
  "the boy is sad / upset / stuck and the girl cheers him up" opening. Instead use
  the mood, setting and hook given in the FRESHNESS DIRECTIVE above. Open with a
  fresh hook (a funny mistake, an exciting discovery, a playful challenge, a "guess
  what I just did!", a mini competition, a curious question) - NOT with sadness.
- Use the two kid NAMES given in the FRESHNESS DIRECTIVE (don't default to the same
  names every time).
- End on a happy, satisfying takeaway from the boy (he "gets it") + a tiny call to action.
- Use ONLY real, accurate tools/facts. Never invent fake tools or claims.

Return ONLY valid JSON (no markdown, no code fences) with these exact keys:
{{
  "topic": "2-4 word label of the main AI tool/subject (for de-duplication)",
  "title": "catchy YouTube title under 70 chars, name the tool, include 1 emoji",
  "dialogue": [
    {{"speaker": "girl", "line": "first line spoken by the girl"}},
    {{"speaker": "boy", "line": "boy's reply"}}
  ],
  "description": "2-3 sentence YouTube description that names the tool and summarizes the lesson",
  "tags": ["8-12", "lowercase", "hashtag-free", "keywords"],
  "hashtags": ["5-8", "#shorts", "#relevant", "#viral", "hashtags with the # sign"],
  "image_prompts": [
    "ONE adorable image prompt for EVERY dialogue turn above, IN THE SAME ORDER",
    "(so the number of image_prompts EQUALS the number of dialogue turns). Each",
    "depicts that line's moment: a smart little girl and a curious little boy,",
    "warm cozy lighting, cute Pixar-like 3D characters, vertical (9:16), NO text."
  ]
}}

IMPORTANT: 'image_prompts' MUST have exactly ONE entry per item in 'dialogue',
in the same order (same count).
"""


# --- Story variety pools: randomly combined so each story feels fresh ---
_STORY_NAMES = [
    ("Pari", "Aarav"), ("Riya", "Kabir"), ("Anaya", "Vivaan"), ("Meera", "Reyansh"),
    ("Saanvi", "Aryan"), ("Diya", "Ishaan"), ("Kiara", "Advik"), ("Myra", "Krish"),
    ("Aadhya", "Dhruv"), ("Navya", "Arjun"), ("Ira", "Veer"), ("Tara", "Kian"),
]
_STORY_MOODS = [
    "cheerful and excited", "playful and giggly", "curious and amazed",
    "competitive and fun", "proud and show-offy (in a cute way)",
    "mischievous and clever", "wide-eyed and wonder-struck", "energetic and bubbly",
]
_STORY_SETTINGS = [
    "on the school playground during recess", "in a cozy bedroom full of toys",
    "on a rooftop flying a kite", "at a birthday party", "in the school computer lab",
    "under a tree in the park", "at a lemonade stand they set up", "in a treehouse",
    "on the way home from school", "in grandma's sunny kitchen", "at a science fair booth",
    "during a rainy afternoon indoors",
]
_STORY_HOOKS = [
    "the girl excitedly shows off something cool she just made with AI",
    "the boy proudly shares a small win and the girl levels it up with a smart AI trick",
    "they turn it into a friendly challenge to see who can do a task faster",
    "the boy asks a funny 'what if' question that leads into the AI tip",
    "the girl surprises the boy with a magic-seeming demo, then reveals the real tool",
    "they're planning a fun project together and need the AI tool to pull it off",
    "the boy makes a silly mistake, they laugh, and the girl shows the smarter way",
    "the girl bets she can do something impressive, then teaches how",
]


def _story_variety() -> str:
    """Pick a fresh name pair, mood, setting and hook so stories don't repeat."""
    girl, boy = random.choice(_STORY_NAMES)
    mood = random.choice(_STORY_MOODS)
    setting = random.choice(_STORY_SETTINGS)
    hook = random.choice(_STORY_HOOKS)
    return (
        f"- Girl's name: {girl}. Boy's name: {boy} (use these names).\n"
        f"- Overall mood/tone: {mood}.\n"
        f"- Setting/scene: {setting}.\n"
        f"- Opening hook to build the story around: {hook}.\n"
        f"- Keep it upbeat from the very first line - NO sadness/crying openings."
    )


def _retry_wait(msg: str, attempt: int) -> float:
    """Seconds to wait before retrying a 429. Honor the server's retry hint if
    present (e.g. 'Please retry in 55.8s' / 'seconds: 55'), else exponential."""
    m = (re.search(r"retry[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s", msg, re.I)
         or re.search(r"seconds:\s*(\d+)", msg))
    if m:
        return min(float(m.group(1)) + 2, 65)
    return min(15 * (attempt + 1), 60)


def _generate_json(prompt: str) -> dict:
    """Call Gemini for JSON content, retrying and falling back across models on
    rate-limit (429) or transient errors, so one quota hit doesn't fail the run."""
    genai.configure(api_key=config.GEMINI_API_KEY)
    last_exc = None
    for model_name in GEMINI_MODELS:
        model = genai.GenerativeModel(model_name)
        for attempt in range(2):  # one retry per model after a backoff wait
            try:
                response = model.generate_content(prompt)
                return _extract_json(response.text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc)
                is_quota = ("429" in msg or "quota" in msg.lower()
                            or "exhaust" in msg.lower() or "rate" in msg.lower())
                if is_quota and attempt == 0:
                    wait = _retry_wait(msg, attempt)
                    print(f"[content] {model_name} rate-limited (429); waiting "
                          f"{wait:.0f}s then retrying once...")
                    time.sleep(wait)
                    continue
                print(f"[content] {model_name} failed ({'429' if is_quota else 'error'}): "
                      f"{msg[:120]}. Trying next model...")
                break
    raise RuntimeError(f"All Gemini models exhausted/failed. Last error: {last_exc}")


def _extract_json(text: str) -> dict:
    """Strip code fences / stray text and parse the first JSON object found."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model response:\n{text}")
    return json.loads(match.group(0))


def generate_content(niche: str = None, style: str = None, language: str = None) -> dict:
    """Return video content. Style 'tips' -> narration; 'story' -> two-kid dialogue."""
    niche = niche or config.NICHE
    style = (style or config.PROMPT_STYLE or "tips").lower()
    language = (language or config.VIDEO_LANGUAGE or "english").strip()

    trends = get_trending(niche)
    trends_block = "\n".join(f"- {t}" for t in trends) or "- (no live trends available)"
    past = history.recent_topics()
    history_block = "\n".join(f"- {p}" for p in past) or "- (none yet)"

    lang_directive = (
        f"CRITICAL LANGUAGE RULE: Write ALL spoken text (the narration and EVERY dialogue "
        f"line) and the title and description in {language}, using natural, native, "
        f"conversational {language} (in its native script). Keep real tool/brand names in "
        f"their original spelling. 'tags' are lowercase keywords ({language} and/or english); "
        f"'hashtags' stay relevant and MUST include #shorts. IMPORTANT: keep 'image_prompts' "
        f"in ENGLISH always (the image model needs English).\n\n"
    )
    if language.lower() == "hindi":
        lang_directive += (
            "HINDI STYLE: Write the way real, everyday North-Indian (Delhi) kids actually "
            "talk \u2014 warm, playful, simple spoken Hindi (Devanagari). It's natural to keep "
            "common English tech words (app, phone, AI, tool names) in English as kids do, "
            "but keep the sentences clean, correct and easy for a child to say. Avoid heavy, "
            "bookish or Sanskritised words.\n\n"
        )
    template = STORY_PROMPT_TEMPLATE if style == "story" else TIPS_PROMPT_TEMPLATE
    prompt = lang_directive + template.format(
        niche=niche, trends=trends_block, history=history_block,
        variety=_story_variety(),
    )
    data = _generate_json(prompt)

    # Shared defaults
    data.setdefault("tags", [])
    data.setdefault("hashtags", ["#shorts", "#ai", "#tech", "#viral"])
    data.setdefault("topic", data.get("title", niche))
    if not data.get("image_prompts"):
        data["image_prompts"] = [f"{niche}, modern cinematic vertical image"]

    if style == "story":
        data["style"] = "story"
        dialogue = data.get("dialogue") or []
        if not dialogue:
            raise ValueError(f"Story response missing 'dialogue': {data}")
        # Normalize speakers and build a combined narration (for captions fallback).
        for turn in dialogue:
            turn["speaker"] = "boy" if str(turn.get("speaker", "")).lower().startswith("b") else "girl"
        data["dialogue"] = dialogue
        data["narration"] = " ".join(t.get("line", "") for t in dialogue).strip()
        for key in ("title", "description"):
            if not data.get(key):
                raise ValueError(f"Story response missing '{key}': {data}")
    else:
        data["style"] = "tips"
        for key in ("title", "narration", "description"):
            if not data.get(key):
                raise ValueError(f"Model response missing '{key}': {data}")

    # Remember this topic so we don't repeat it next time.
    history.add_entry(data["title"], data["topic"])
    return data


if __name__ == "__main__":
    config.validate()
    print(json.dumps(generate_content(), indent=2))
