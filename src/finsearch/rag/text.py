"""Tiny language-aware text helpers shared by the extractive answerer and the evidence gate.

Deliberately simple (no NLTK/spaCy): lowercase, fold umlauts, drop stopwords, strip a few
common suffixes. Good enough to match "Zinsen"/"Zins" or "rates"/"rate" lexically.
"""

from __future__ import annotations

import re

STOPWORDS: dict[str, frozenset[str]] = {
    "de": frozenset(
        "der die das den dem des ein eine einen einem einer eines und oder aber nicht ist sind war "
        "waren wird werden wurde wurden hat haben hatte hatten wie was wer wo wann warum welche "
        "welcher welches welchen wieviel wie viel auf aus bei mit nach von vom zu zum zur im in "
        "am an als auch für fur über uber um es sich sie er wir ihr ich du dass daß so noch nur "
        "kann können konnen soll sollen gibt es bis durch gegen unter seit zwischen dieser diese "
        "dieses ihre ihrer sein seine seiner hoch".split()
    ),
    "en": frozenset(
        "the a an and or but not is are was were be been being has have had do does did what who "
        "where when why which how much many of in on at to for from by with as it its this that "
        "these those there their they he she we you i about into over under between during than "
        "then so can could should would will shall may might".split()
    ),
}
MIN_TERM_LEN = 2
MIN_STEM_LEN = 4  # a stem keeps at least this many letters
_DE_SUFFIXES = (
    "ungen",
    "ung",
    "heit",
    "keit",
    "lich",
    "isch",
    "en",
    "er",
    "es",
    "em",
    "e",
    "n",
    "s",
)
_EN_SUFFIXES = ("ing", "ed", "es", "s")
_FOLD = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "ss"})
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# split after . ! ? when the next sentence starts with an upper-case letter, digit, quote or marker
_LEADING_MARKERS_RE = re.compile(r"((?:\[\d+\]\s*)+)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\[\"„(A-ZÄÖÜ0-9])")


def detect_lang(text: str) -> str:
    """'de' or 'en', by counting stopword hits per language (ties -> 'de', the corpus default)."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    de = sum(t in STOPWORDS["de"] for t in tokens)
    en = sum(t in STOPWORDS["en"] for t in tokens)
    return "en" if en > de else "de"


def stem(token: str, lang: str) -> str:
    """Strip one common suffix (longest first). Not linguistically correct, just consistent."""
    for suffix in _DE_SUFFIXES if lang == "de" else _EN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_LEN:
            return token[: -len(suffix)]
    return token


def content_terms(text: str, lang: str | None = None) -> list[str]:
    """Stemmed, stopword-free terms of `text`. `lang` is detected from the text when omitted."""
    lang = lang or detect_lang(text)
    out = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tok = raw.translate(_FOLD)
        if len(tok) >= MIN_TERM_LEN and raw not in STOPWORDS[lang] and tok not in STOPWORDS[lang]:
            out.append(stem(tok, lang))
    return out


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation. Abbreviations like 'Mio. CHF' are split wrongly (accepted)."""
    merged: list[str] = []
    for part in _SENTENCE_RE.split(re.sub(r"\s+", " ", text).strip()):
        # markers right after the full stop ("... Juni. [1] Der ...") belong to the previous sentence
        lead = _LEADING_MARKERS_RE.match(part)
        if lead and merged:
            merged[-1] += " " + lead[1].strip()
            part = part[lead.end() :]
        if part:
            merged.append(part)
    return merged
