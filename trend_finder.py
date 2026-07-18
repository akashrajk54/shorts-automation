"""Find current trending tech/AI topics from free public sources (no API key)."""
import re
import xml.etree.ElementTree as ET

import config  # noqa: F401  (imported first to configure SSL trust)
import requests

HN_ALGOLIA = "https://hn.algolia.com/api/v1/search"
GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo={geo}"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _hacker_news(queries: list[str], limit: int = 12) -> list[str]:
    """Pull trending tech headlines from Hacker News (front page + AI search)."""
    titles: list[str] = []
    try:
        r = requests.get(
            HN_ALGOLIA,
            params={"tags": "front_page", "hitsPerPage": 20},
            headers=HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        titles += [h["title"] for h in r.json().get("hits", []) if h.get("title")]
    except requests.RequestException as exc:
        print(f"[trends] HN front_page failed: {exc}")

    for q in queries:
        try:
            r = requests.get(
                HN_ALGOLIA,
                params={"query": q, "tags": "story", "hitsPerPage": 8},
                headers=HEADERS,
                timeout=20,
            )
            r.raise_for_status()
            titles += [h["title"] for h in r.json().get("hits", []) if h.get("title")]
        except requests.RequestException as exc:
            print(f"[trends] HN query '{q}' failed: {exc}")

    return titles[:limit]


def _google_trends(geo: str = "US", limit: int = 12) -> list[str]:
    """Pull daily trending search terms from Google Trends RSS."""
    terms: list[str] = []
    try:
        r = requests.get(GOOGLE_TRENDS_RSS.format(geo=geo), headers=HEADERS, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.iter("item"):
            title = item.findtext("title")
            if title:
                terms.append(title.strip())
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"[trends] Google Trends RSS failed: {exc}")
    return terms[:limit]


def get_trending(niche: str, geo: str = "US") -> list[str]:
    """Return a deduped list of currently trending topics relevant to the niche."""
    queries = ["AI", "AI tools", "ChatGPT", "artificial intelligence"]
    combined = _hacker_news(queries) + _google_trends(geo)

    seen, result = set(), []
    for t in combined:
        clean = re.sub(r"\s+", " ", t).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


if __name__ == "__main__":
    for t in get_trending("AI tools and tech tips"):
        print("-", t)
