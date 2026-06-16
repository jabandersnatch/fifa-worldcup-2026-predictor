"""Fetch and parse Forebet's 2026 World Cup score predictions, for corroboration.

Forebet is an independent Poisson-style score engine. It rejects default request
headers (HTTP 403), so we send a browser User-Agent. Returns each match's
predicted correct score keyed by canonical team names so it can be lined up with
our own fixtures.
"""
import os
import re
import json
import html
import datetime
import unicodedata

import requests

URL = "https://www.forebet.com/en/predictions-world/world-cup"
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "forebet_cache.json")
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Accept-Language": "en-US,en;q=0.9",
}

# Canonicalised name -> our dataset's canonical name, for the few that differ.
_ALIASES = {
    "bosnia herzegovina": "bosnia and herzegovina",
    "usa": "united states",
    "korea republic": "south korea",
    "korea south": "south korea",
    "czechia": "czech republic",
    "turkiye": "turkey",
    "cote d ivoire": "ivory coast",
    "cabo verde": "cape verde",
    "democratic republic of congo": "dr congo",
    "congo dr": "dr congo",
}


def canon(name):
    """Accent-strip, lowercase, normalise separators, then alias-map."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("-", " ").replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _ALIASES.get(s, s)


def fetch_html(timeout=25):
    resp = requests.get(URL, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse(page):
    """Return {(canon_home, canon_away): 'h-a'} from the predictions page."""
    out = {}
    for block in re.split(r'itemtype="http://schema.org/SportsEvent"', page)[1:]:
        name = re.search(r'itemprop="name" content="([^"]+)"', block)
        sc = re.search(r'class="ex_sc[^"]*"[^>]*>\s*(\d)\s*[-:]\s*(\d)', block)
        if not (name and sc):
            continue
        home, sep, away = html.unescape(name.group(1)).partition(" vs ")
        if not sep:
            continue
        out[(canon(home), canon(away))] = f"{sc.group(1)}-{sc.group(2)}"
    return out


def _save_cache(data):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    payload = {
        "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "matches": [[h, a, s] for (h, a), s in data.items()],
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=0)


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return None, None
    with open(CACHE_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    data = {(h, a): s for h, a, s in payload["matches"]}
    return data, payload.get("fetched_at")


def predictions(timeout=25):
    """Fetch + parse Forebet, caching the result to data/forebet_cache.json.

    Returns (score_dict, meta) where meta = {"source": "live"|"cache",
    "fetched_at": iso-timestamp}. On a live failure, falls back to the cached
    pull if one exists; only raises if there is no cache to fall back to.
    """
    try:
        data = parse(fetch_html(timeout=timeout))
        if not data:
            raise ValueError("no Forebet predictions parsed (page layout may have changed)")
        _save_cache(data)
        return data, {"source": "live",
                      "fetched_at": datetime.datetime.now().isoformat(timespec="seconds")}
    except Exception as live_err:
        cached, ts = _load_cache()
        if cached:
            return cached, {"source": "cache", "fetched_at": ts, "error": str(live_err)}
        raise


def lookup(data, home, away):
    """Forebet score for our (home, away), trying both orientations.
    Returns the score oriented to OUR home/away, or None."""
    h, a = canon(home), canon(away)
    if (h, a) in data:
        return data[(h, a)]
    if (a, h) in data:                      # listed with sides swapped -> flip
        x, y = data[(a, h)].split("-")
        return f"{y}-{x}"
    return None
