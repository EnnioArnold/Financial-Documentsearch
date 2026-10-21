# Entscheidungen (ADRs)

## ADR-001 – Hybrid-Suche mit RRF im Client
**Kontext:** BM25 ist stark bei exakten Begriffen/Zahlen, kNN bei Paraphrasen und sprachübergreifenden Anfragen. **Entscheid:** beide Ranglisten separat holen und im Python-Code mit Reciprocal Rank Fusion (k=60) mischen. **Warum:** RRF braucht nur Ränge (keine Score-Normalisierung, BM25- und Cosine-Scores sind nicht vergleichbar); nativer ES-RRF-Retriever ist an Lizenzstufen gebunden und schwerer zu testen. Im Client ist die Logik unit-testbar und deterministisch. **Konsequenz:** zwei ES-Roundtrips pro Anfrage.

## ADR-002 – Embedding-Modell `intfloat/multilingual-e5-small`
384 Dim., deutsch/englisch, läuft auf CPU, geringe Index-Grösse. Präfixe `query: `/`passage: ` sind Pflicht, Vektoren werden normalisiert (Cosine = Dot-Product). Alternative `BAAI/bge-m3` per Config, aber 1024 Dim. und deutlich langsamer. Modell-Revision wird beim Laden gepinnt.

## ADR-003 – Ein Index mit Sprach-Subfeldern, Alias + Version
Pro Chunk werden `text_de` (german) und `text_en` (english) aus demselben Text analysiert; die Query geht über `multi_match` auf beide plus `text` (Phrasen). Einfacher als zwei Indizes pro Sprache und robust bei gemischten Dokumenten. Index `finsearch-v1` hinter Alias `finsearch` ermöglicht Re-Index ohne Downtime (neuer Index → Alias umschalten).

## ADR-004 – `LLMClient`-Interface, `extractive` als Standard
Kein Provider ist zwingend. Der `extractive`-Provider wählt die relevantesten Sätze aus den Treffern (kein generierter Text, also keine Halluzination) und liefert Quellen. Echte LLMs (Ollama, Gemini, Groq, Anthropic) sind austauschbar per `.env`. Zitate werden unabhängig vom Provider validiert.

## ADR-005 – Projektstruktur und Git
Das Projekt liegt im Ordner `finsearch/` und hat ein **eigenes, lokales Git-Repository** (damit es später als eigenes GitHub-Repo gepusht werden kann, ohne das umgebende Verzeichnis). Kein Push durch Claude. Python-Pakete werden mit `uv` installiert; Docker-Images sind gepinnt (ES/Kibana 9.4.8).

## ADR-006 – Verzicht auf Plan Mode, Brainstorming-Skill
Der Auftrag ist ein vollständiger Spec; Plan Mode würde auf Nutzerfreigabe warten und der Auftrag verlangt autonome Arbeit. Architektur und Verträge wurden stattdessen direkt in `docs/ARCHITECTURE.md` und `contracts.py` festgehalten.
