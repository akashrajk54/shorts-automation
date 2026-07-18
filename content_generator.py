"""Generate a short-form video script + metadata using Google Gemini (free tier)."""
import json
import re

import config  # noqa: F401  (imported first to configure SSL trust)
import google.generativeai as genai
import history
from trend_finder import get_trending

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
    "4 detailed AI image-generation prompts, one per scene, that visually match",
    "the narration. Each: a vivid, modern, high-quality vertical (9:16) image,",
    "tech/futuristic aesthetic, cinematic lighting, photorealistic, NO text or",
    "words rendered in the image."
  ]
}}
"""


STORY_PROMPT_TEMPLATE = """You are a beloved children's story writer AND a viral YouTube Shorts
creator for the niche: "{niche}".

Write ONE short, heart-warming STORY in the form of a natural conversation between
TWO cute little kids:
- a smart, sweet, confident GIRL (she is knowledgeable and kindly teaches),
- a curious, funny, slightly clueless BOY (he asks questions and learns).

The girl teaches the boy something genuinely USEFUL about being smart with AI -
a specific, real, accurate AI tool or trick. It must feel like a lovable little
story that is a joy to listen to: warm, playful, wholesome, and easy to follow -
while still delivering COMPLETE, correct, useful information the viewer can act on.

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
    "4 adorable, high-quality vertical (9:16) storybook / 3D-animation style image",
    "prompts showing a smart little girl and a curious little boy talking, warm",
    "cozy lighting, cute Pixar-like characters, matching the story scenes,",
    "NO text or words rendered in the image."
  ]
}}
"""


def _extract_json(text: str) -> dict:
    """Strip code fences / stray text and parse the first JSON object found."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model response:\n{text}")
    return json.loads(match.group(0))


def generate_content(niche: str = None, style: str = None) -> dict:
    """Return video content. Style 'tips' -> narration; 'story' -> two-kid dialogue."""
    niche = niche or config.NICHE
    style = (style or config.PROMPT_STYLE or "tips").lower()
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-flash-latest")

    trends = get_trending(niche)
    trends_block = "\n".join(f"- {t}" for t in trends) or "- (no live trends available)"
    past = history.recent_topics()
    history_block = "\n".join(f"- {p}" for p in past) or "- (none yet)"

    template = STORY_PROMPT_TEMPLATE if style == "story" else TIPS_PROMPT_TEMPLATE
    prompt = template.format(niche=niche, trends=trends_block, history=history_block)
    response = model.generate_content(prompt)
    data = _extract_json(response.text)

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
