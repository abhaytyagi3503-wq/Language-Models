#!/usr/bin/env python3
"""
at3684-lab3 - birdDescriptionDownloader.py

Reads birdNames.txt and produces birdArticles.csv with columns:
    speciesName, sex, sourceURL, extractedTextDescription

Robust features:
- handles UTF-8 / UTF-8-SIG / latin-1 input
- handles hybrid notation Ã— / × / x
- handles binomial, trinomial, and hybrid scientific names
- uses Wikipedia exact title, redirect, and search fallback
- patches known difficult taxonomy/title cases
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


HERE = Path(__file__).resolve().parent
BIRD_NAMES_PATH = HERE / "birdNames.txt"
OUT_CSV_PATH = HERE / "birdArticles.csv"

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "at3684-lab3/1.0 (educational use)"}


def normalize_space(s: str) -> str:
    s = str(s).replace("\ufeff", " ")
    s = s.replace("Ã—", " × ")
    s = s.replace("×", " × ")
    return re.sub(r"\s+", " ", s).strip()


def split_tokens(line: str) -> List[str]:
    parts = re.split(r"[,|\t;]+", line)
    return [normalize_space(p) for p in parts if normalize_space(p)]


def looks_like_scientific_name(text: str) -> bool:
    text = normalize_space(text)
    hybrid_pat = r"^[A-Z][a-zA-Z-]+ [a-z][a-zA-Z-]+ [×x] [a-z][a-zA-Z-]+$"
    regular_pat = r"^[A-Z][a-zA-Z-]+(?: [a-z][a-zA-Z-]+){1,2}$"
    return bool(re.match(hybrid_pat, text) or re.match(regular_pat, text))


def parse_bird_line(line: str) -> Optional[Dict[str, str]]:
    raw = normalize_space(line)
    if not raw or raw.startswith("#"):
        return None

    parts = split_tokens(raw)
    sex_words = {"male", "female", "unknown", "both", "sexes", "m", "f"}

    if len(parts) >= 2:
        if parts[0].lower() in sex_words:
            maybe_species = normalize_space(" ".join(parts[1:]))
            if looks_like_scientific_name(maybe_species):
                return {"sex": parts[0].lower(), "speciesName": maybe_species}

        if parts[-1].lower() in sex_words:
            maybe_species = normalize_space(" ".join(parts[:-1]))
            if looks_like_scientific_name(maybe_species):
                return {"sex": parts[-1].lower(), "speciesName": maybe_species}

    tokens = raw.split()
    if tokens and tokens[0].lower() in sex_words and len(tokens) >= 3:
        maybe_species = normalize_space(" ".join(tokens[1:]))
        if looks_like_scientific_name(maybe_species):
            return {"sex": tokens[0].lower(), "speciesName": maybe_species}

    if looks_like_scientific_name(raw):
        return {"sex": "unknown", "speciesName": raw}

    return None


def load_bird_entries(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    raw_bytes = path.read_bytes()

    text = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        text = raw_bytes.decode("utf-8", errors="ignore")

    entries: List[Dict[str, str]] = []
    bad_lines: List[Tuple[int, str]] = []

    for idx, line in enumerate(text.splitlines(), start=1):
        parsed = parse_bird_line(line)
        if parsed is None:
            if normalize_space(line):
                bad_lines.append((idx, line))
            continue
        entries.append(parsed)

    if not entries:
        raise ValueError("No valid bird entries could be parsed from birdNames.txt")

    if bad_lines:
        print("Warning: skipped lines that did not match expected patterns:")
        for idx, line in bad_lines[:20]:
            print(f"  line {idx}: {line}")

    return entries


def get_extract_for_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "url",
        "redirects": 1,
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=25)
    if r.status_code != 200:
        return None, None

    pages = r.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        txt = normalize_space(page.get("extract", ""))
        url = page.get("fullurl", "")
        if len(txt) > 120 and url:
            return url, txt
    return None, None


def search_titles(query: str) -> List[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
        "format": "json",
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=25)
    if r.status_code != 200:
        return []
    return [x["title"] for x in r.json().get("query", {}).get("search", [])]


KNOWN_TITLE_MAP = {
    "Sialia mexicana": "Western bluebird",
    "Vermivora chrysoptera × cyanoptera": "Brewster's warbler",
    "Driophlox fuscicauda": "Red-throated ant tanager",
}


def candidate_queries(species_name: str) -> List[str]:
    s = normalize_space(species_name)
    toks = s.split()

    out = []

    if s in KNOWN_TITLE_MAP:
        out.append(KNOWN_TITLE_MAP[s])

    out.append(s)

    if len(toks) >= 2:
        out.append(" ".join(toks[:2]))

    if len(toks) == 3 and toks[0][0].isupper():
        out.append(" ".join(toks[:2]))

    if "×" in toks or "x" in toks:
        if len(toks) >= 4:
            out.append(" ".join(toks[:2]))

    seen = set()
    final = []
    for x in out:
        x = normalize_space(x)
        if x and x not in seen:
            seen.add(x)
            final.append(x)
    return final


def article_for_species(species_name: str) -> Tuple[str, str]:
    for q in candidate_queries(species_name):
        url, txt = get_extract_for_title(q)
        if txt:
            return url, txt

    for q in candidate_queries(species_name):
        titles = search_titles(q)
        for t in titles:
            url, txt = get_extract_for_title(t)
            if txt:
                return url, txt

    fallback_url = f"https://en.wikipedia.org/wiki/{species_name.replace(' ', '_')}"
    fallback_txt = f"{species_name} description unavailable. Manual follow-up required."
    return fallback_url, fallback_txt


def main() -> int:
    entries = load_bird_entries(BIRD_NAMES_PATH)
    rows = []
    unresolved = []

    for i, entry in enumerate(entries, start=1):
        sex = normalize_space(entry["sex"])
        species = normalize_space(entry["speciesName"])
        print(f"[{i}/{len(entries)}] {sex} {species}")

        url, txt = article_for_species(species)

        if "manual follow-up required" in txt.lower() or "description unavailable" in txt.lower():
            unresolved.append((sex, species))

        rows.append(
            {
                "speciesName": species,
                "sex": sex,
                "sourceURL": url,
                "extractedTextDescription": txt,
            }
        )
        time.sleep(0.2)

    with OUT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["speciesName", "sex", "sourceURL", "extractedTextDescription"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV_PATH}")
    if unresolved:
        print(f"WARNING: {len(unresolved)} rows need manual follow-up.")
    else:
        print("All rows resolved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())