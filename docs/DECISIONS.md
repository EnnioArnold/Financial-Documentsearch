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

## ADR-007 – Evidence-Gate vor der Antwort (Review H1)
**Problem:** Die erste Version prüfte, ob Frage-Terme irgendwo in der Vereinigung der Top-6-Chunks vorkommen. Das liess 7 von 8 unbeantwortbaren Golden-Fragen durch (Review `reports/review.md`). **Entscheid:** Eine Anfrage gilt als belegt, wenn (a) ein *einzelner* Chunk mindestens 70 % der Frage-Terme enthält (`rag_min_coverage`, Stemming nach Chunk-Sprache) **oder** (b) der BM25-Top-1-Score ≥ 26 ist (`rag_min_bm25`, eine zusätzliche BM25-Abfrage mit k=1, nur wenn (a) scheitert). Das Gate läuft vor jedem LLM-Aufruf. **Messung** (`eval/rag_gate_eval.py`, `reports/rag_gate.md`): 8/8 unbeantwortbare Fragen abgelehnt, 40/43 beantwortbare beantwortet (Fehlablehnungen: Paraphrasen/Cross-Language). **Grenze:** Schwellen wurden auf demselben Golden Set getunt, das nur 8 negative Fragen hat – die Zahlen sind optimistisch. BM25-Scores sind indexabhängig; bei neuem Korpus neu kalibrieren. `rag_min_score` wurde entfernt (Scores sind methodenabhängig, wirkungslos).

## ADR-008 – Hybrid schlägt BM25 auf unserem Golden Set nicht
Messung (43 Fragen, `reports/eval_results.md`): BM25 Recall@5 0.826 / MRR 0.685, kNN 0.651 / 0.536, Hybrid 0.767 / 0.640, Hybrid+Rerank 0.860 / 0.743. **Ursachen (Analyse, nicht bewiesen):** (1) Das Golden Set wurde aus Chunk-Zitaten geschrieben und ist formulierungsnah, das begünstigt BM25. (2) `multilingual-e5-small` ist deutlich schwächer als BM25 auf diesen Faktenfragen; RRF mit gleichen Gewichten zieht die gute Liste Richtung der schwächeren. (3) Bei Paraphrasen ist kNN besser als BM25 (R@5 0.75 vs 0.62), Hybrid liegt dazwischen. Der Parameter-Sweep (RRF-k 20/60/100, Boosts) ändert Hybrid kaum (R@5 0.767–0.791). Kein Wert wurde zum Verschönern nachgetunt. **Konsequenz:** Hybrid+Rerank ist die beste Methode, kostet aber ca. 4–5 s CPU-Latenz. Nächste Hebel: gewichtetes RRF, ein stärkeres Embedding-Modell (bge-m3), ein unabhängig formuliertes Golden Set. Der Chunk-Grössen-Vergleich (300 vs. 500) wurde **nicht** durchgeführt.
