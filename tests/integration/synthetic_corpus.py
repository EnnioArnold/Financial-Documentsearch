"""SYNTHETIC finance snippets for tests. Written for this project, not taken from real documents."""

from __future__ import annotations

from datetime import date

from fakes import make_chunk

from finsearch.contracts import Chunk


def _c(doc: str, ord_: int, text: str, **meta: object) -> Chunk:
    return make_chunk(f"{doc}-{ord_:04d}", text, doc_id=doc, ord=ord_, page=ord_ + 1, **meta)


_SNB_SPEECH = {
    "source": "snb",
    "lang": "de",
    "doc_type": "speech",
    "date": date(2024, 3, 21),
    "title": "SYNTHETIC SNB Rede",
}
_SNB_ANNUAL = {
    "source": "snb",
    "lang": "de",
    "doc_type": "annual_report",
    "date": date(2023, 3, 10),
    "title": "SYNTHETIC SNB Geschaeftsbericht",
}
_FED_SPEECH = {
    "source": "fed",
    "lang": "en",
    "doc_type": "speech",
    "date": date(2024, 6, 12),
    "title": "SYNTHETIC Fed speech",
}
_FED_MINUTES = {
    "source": "fed",
    "lang": "en",
    "doc_type": "market_comment",
    "date": date(2022, 9, 21),
    "title": "SYNTHETIC Fed minutes",
}
_BANK_DE = {
    "source": "other",
    "lang": "de",
    "doc_type": "market_comment",
    "date": date(2024, 1, 15),
    "title": "SYNTHETIC Marktkommentar",
}
_BANK_EN = {
    "source": "other",
    "lang": "en",
    "doc_type": "annual_report",
    "date": None,
    "title": "SYNTHETIC bank annual report",
}

CORPUS: list[Chunk] = [
    _c(
        "snb-speech",
        0,
        "Die Nationalbank senkt den Leitzins auf 1,5 Prozent, weil die Inflation unter Kontrolle ist.",
        **_SNB_SPEECH,
    ),
    _c(
        "snb-speech",
        1,
        "Die Teuerung lag im Februar bei 1,2 Prozent und damit im Zielband der SNB.",
        **_SNB_SPEECH,
    ),
    _c(
        "snb-annual",
        0,
        "Die Bilanz der SNB hat sich verkürzt. Fremdwährungsanlagen und Gold bilden den grössten Teil der Aktiven.",
        **_SNB_ANNUAL,
    ),
    _c(
        "snb-annual",
        1,
        "Der Zinssatz auf Sichtguthaben belastete die Banken, der Jahresgewinn hängt stark vom Wechselkurs ab.",
        **_SNB_ANNUAL,
    ),
    _c(
        "fed-speech",
        0,
        "The Federal Reserve held the policy rate steady. Monetary policy stays restrictive until inflation moves toward two percent.",
        **_FED_SPEECH,
    ),
    _c(
        "fed-speech",
        1,
        "Higher interest rates have slowed consumer spending and cooled the labor market.",
        **_FED_SPEECH,
    ),
    _c(
        "fed-minutes",
        0,
        "The committee raised the federal funds rate by 75 basis points to fight rising prices.",
        **_FED_MINUTES,
    ),
    _c(
        "fed-minutes",
        1,
        "Balance sheet runoff continues: the Fed is reducing its holdings of Treasury securities.",
        **_FED_MINUTES,
    ),
    _c(
        "bank-de",
        0,
        "Die Aktienmärkte reagierten positiv auf Zinssenkungen. Anleger erwarten tiefere Hypothekarzinsen im Sommer.",
        **_BANK_DE,
    ),
    _c(
        "bank-en",
        0,
        "The bank reported higher net interest income and stable capital ratios. Credit losses remained low.",
        **_BANK_EN,
    ),
    _c(
        "bank-en",
        1,
        "Inflation outlook <script>alert(1)</script> and wage growth stay uncertain.",
        **_BANK_EN,
    ),
]

# (query, id of the single relevant chunk) - mixes exact words, inflection, paraphrase, cross-language
JUDGED_QUERIES: list[tuple[str, str]] = [
    ("Leitzins gesenkt", "snb-speech-0000"),
    ("Zinssätze der Banken", "snb-annual-0001"),
    ("balance sheet shrinking Treasury", "fed-minutes-0001"),
    ("central bank cuts its key interest rate", "snb-speech-0000"),
    ("prices rising faster so the Fed hikes the policy rate", "fed-minutes-0000"),
    ("Wie hoch ist die Inflation in der Schweiz", "snb-speech-0001"),
    ("what happens to household consumption when borrowing costs go up", "fed-speech-0001"),
    ("Gewinn der Nationalbank und Wechselkurs", "snb-annual-0001"),
]
