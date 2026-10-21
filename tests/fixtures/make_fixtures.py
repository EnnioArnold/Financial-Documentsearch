"""Regenerate the SYNTHETIC ingest fixtures: `.venv/bin/python tests/fixtures/make_fixtures.py`.

Everything here is invented test data (fictional "Musterbank AG"); no real document is copied.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
HEADER = "SYNTHETISCH - Musterbank AG - Geschaeftsbericht 2099"

PAGES_DE = [
    [
        "Die Musterbank AG ist eine erfundene Bank fuer Testzwecke. Der Zins-",
        "satz stieg im Jahr 2099 auf 2.5% und die Inflation blieb stabil. Die Bank",
        "erwartet fuer das naechste Jahr ein moderates Wachstum der Kredite.",
    ],
    [
        "Im Hypothekargeschaeft sank die Nachfrage leicht. Der Zins- und Kreditmarkt",
        "blieb jedoch robust. Der Verwaltungsrat tagte am 5. Mai 2099 in Bern.",
        "1 234 567 8.9 -0.3",
    ],
    [
        "Die Eigenmittel der Musterbank AG liegen deutlich ueber den regulatorischen",
        "Anforderungen. Die Kernkapitalquote betrug 21.2% (Vorjahr: 20.4%). Das",
        "Risikomanagement wurde im Berichtsjahr weiter gestaerkt.",
    ],
    [
        "Fuer das Jahr 2100 plant die Musterbank AG keine neuen Filialen. Stattdessen",
        "werden digitale Angebote ausgebaut. Dieser Satz dient nur als Testinhalt.",
    ],
]

HTML_EN = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SYNTHETIC speech</title></head>
<body>
<nav>Home | News | Contact</nav>
<article>
<h1>SYNTHETIC speech by the fictional Governor of the Example Bank</h1>
<p>This is a synthetic test document and not a real speech. Inflation in the fictional
economy fell to 2.1 percent in 2099, close to the target of the Example Bank.</p>
<p>The Governing Board therefore decided to keep the policy rate unchanged at 1.5 percent.
Mortgage lending grew moderately and the labour market remained stable throughout the year.</p>
<p>Looking ahead, we will continue to monitor financial stability risks carefully. Thank you
for your attention, and I look forward to your questions about the outlook for the economy.</p>
</article>
<footer>Copyright of nobody. Synthetic.</footer>
</body></html>
"""


def build_pdf() -> bytes:
    doc = pymupdf.open()
    for number, lines in enumerate(PAGES_DE, start=1):
        page = doc.new_page()
        page.insert_text((72, 50), HEADER, fontsize=9)
        page.insert_text((72, 100), "\n".join(lines), fontsize=11)
        page.insert_text((72, 800), f"Seite {number}", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


if __name__ == "__main__":
    (HERE / "synthetic_report_de.pdf").write_bytes(build_pdf())
    (HERE / "synthetic_speech_en.html").write_text(HTML_EN, encoding="utf-8")
