"""Find current trending tech/AI topics from free public sources (no API key)."""
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import config  # noqa: F401  (imported first to configure SSL trust)
import requests

HN_ALGOLIA = "https://hn.algolia.com/api/v1/search"
GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo={geo}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Keep network waits short: many small requests run in parallel, and any single
# slow/unreachable source is skipped rather than stalling the whole run for minutes.
HTTP_TIMEOUT = 8

# Topical queries used to pull FRESH, on-niche headlines (Google News + HN).
NICHE_QUERIES = [
    "AI tools", "new AI app", "ChatGPT tips", "Gemini AI", "AI productivity",
    "best apps", "tech tips", "AI feature",
]

# Keywords that define our lane ("AI & tech tools that make everyday life easier").
# Used to SCORE each trending item so only genuinely relevant, high-demand topics
# reach the script writer - and random Google-Trends noise (sports, celebrities,
# politics) gets filtered out.
NICHE_KEYWORDS = {
    "ai", "a.i", "artificial intelligence", "machine learning", "chatgpt", "gpt",
    "openai", "gemini", "claude", "llm", "copilot", "grok", "deepseek", "llama",
    "midjourney", "sora", "veo", "perplexity", "chatbot", "agent", "prompt",
    "app", "apps", "application", "software", "tool", "tools", "tech", "gadget",
    "gadgets", "phone", "smartphone", "android", "iphone", "ios", "windows",
    "automation", "productivity", "coding", "developer", "programming", "startup",
    "robot", "google", "microsoft", "apple", "meta", "notion", "canva", "excel",
    "whatsapp", "youtube", "instagram", "website", "online", "digital", "feature",
    "update", "launch", "free", "hack", "trick", "tips",
}


def _hn_front_page() -> list[str]:
    try:
        r = requests.get(
            HN_ALGOLIA, params={"tags": "front_page", "hitsPerPage": 20},
            headers=HEADERS, timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return [h["title"] for h in r.json().get("hits", []) if h.get("title")]
    except requests.RequestException as exc:
        print(f"[trends] HN front_page failed: {exc}")
        return []


def _hn_query(q: str) -> list[str]:
    try:
        r = requests.get(
            HN_ALGOLIA, params={"query": q, "tags": "story", "hitsPerPage": 8},
            headers=HEADERS, timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return [h["title"] for h in r.json().get("hits", []) if h.get("title")]
    except requests.RequestException as exc:
        print(f"[trends] HN query '{q}' failed: {exc}")
        return []


def _hacker_news(queries: list[str], limit: int = 12) -> list[str]:
    """Pull trending tech headlines from Hacker News (front page + AI search),
    fetched in parallel so slow queries don't stack up."""
    titles: list[str] = []
    with ThreadPoolExecutor(max_workers=len(queries) + 1) as ex:
        results = list(ex.map(_hn_query, queries))
    titles += _hn_front_page()
    for r in results:
        titles += r
    return titles[:limit]


def _google_trends(geo: str = "IN", limit: int = 12) -> list[str]:
    """Pull daily trending search terms from Google Trends RSS (broad demand signal)."""
    terms: list[str] = []
    try:
        r = requests.get(GOOGLE_TRENDS_RSS.format(geo=geo), headers=HEADERS,
                         timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.iter("item"):
            title = item.findtext("title")
            if title:
                terms.append(title.strip())
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"[trends] Google Trends RSS failed: {exc}")
    return terms[:limit]


def _news_query(q: str, per_query: int = 5) -> list[str]:
    out: list[str] = []
    try:
        url = GOOGLE_NEWS_RSS.format(q=requests.utils.quote(q))
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.iter("item"):
            title = item.findtext("title")
            if title:
                # Google News titles look like "Headline - Publisher"; drop source.
                out.append(title.rsplit(" - ", 1)[0].strip())
                if len(out) >= per_query:
                    break
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"[trends] Google News '{q}' failed: {exc}")
    return out


def _google_news(queries: list[str], per_query: int = 5, limit: int = 18) -> list[str]:
    """Pull FRESH, on-topic headlines from Google News RSS for each niche query,
    fetched in PARALLEL. This is our highest-relevance source: real, current things
    people search for in our lane (new AI tools, app features, tech tips)."""
    titles: list[str] = []
    with ThreadPoolExecutor(max_workers=len(queries)) as ex:
        for r in ex.map(lambda q: _news_query(q, per_query), queries):
            titles += r
    return titles[:limit]


def _relevance_score(text: str) -> int:
    """How well a trending item fits our AI/tech lane (higher = more relevant)."""
    low = f" {text.lower()} "
    return sum(1 for kw in NICHE_KEYWORDS if f" {kw} " in low or f" {kw}s " in low)


def get_trending(niche: str, geo: str = "IN", limit: int = 15) -> list[str]:
    """Return currently trending topics RANKED by relevance to our niche.

    Niche-aware trend-jacking: we gather from multiple free sources, then score
    each item against the lane's keywords so the script writer receives genuinely
    relevant, in-demand topics FIRST (and random noise is filtered out). Falls
    back to the best available items if a slow news day yields few on-niche hits.
    """
    # Run the three sources concurrently so the whole research step is bounded by
    # the single slowest source (~seconds), not the sum of all of them (~minutes).
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_news = ex.submit(_google_news, NICHE_QUERIES)   # freshest, most on-topic
        f_hn = ex.submit(_hacker_news,
                         ["AI", "AI tools", "ChatGPT", "artificial intelligence"])
        f_trends = ex.submit(_google_trends, geo)         # broad demand
        news = f_news.result()
        hn = f_hn.result()
        trends = f_trends.result()

    # Dedupe while preserving source (news/hn are inherently on-niche).
    seen, items = set(), []
    for t in news + hn + trends:
        clean = re.sub(r"\s+", " ", t).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            items.append(clean)

    # Score by relevance; keep only items that clearly fit the lane.
    scored = sorted(items, key=_relevance_score, reverse=True)
    relevant = [t for t in scored if _relevance_score(t) > 0]

    # Fallback: if too few on-niche hits, top up with the freshest news/HN items
    # (still tech-leaning) so the writer always has material.
    if len(relevant) < 6:
        for t in news + hn:
            clean = re.sub(r"\s+", " ", t).strip()
            if clean and clean not in relevant:
                relevant.append(clean)
            if len(relevant) >= 8:
                break

    return relevant[:limit]


if __name__ == "__main__":
    for t in get_trending("AI tools and tech tips"):
        print("-", t)
