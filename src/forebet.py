"""Fetch and parse Forebet's 2026 World Cup score predictions, for corroboration.

Forebet is an independent Poisson-style score engine. It rejects default request
headers (HTTP 403), so we send a browser User-Agent. Returns each match's
predicted correct score keyed by canonical team names so it can be lined up with
our own fixtures.
"""
import re
import html
import unicodedata

import requests

URL = "https://www.forebet.com/en/predictions-world/world-cup"
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


def predictions(timeout=25):
    """Fetch + parse. Returns the score dict, or raises on network/parse failure."""
    data = parse(fetch_html(timeout=timeout))
    if not data:
        raise ValueError("no Forebet predictions parsed (page layout may have changed)")
    return data


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
