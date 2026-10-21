# FinSearch – Unabhängiges Code-Review

Stand: Commit `f6a8657` (4 lokale Commits), Review-Datum 2026-10-21. Reviewer: frischer Kontext, Code selbst gelesen, Befunde per kleinem Experiment verifiziert, wo billig möglich. Es wurde keine Datei ausser dieser geändert; der echte Index `finsearch` (1117 Chunks) wurde nur lesend abgefragt (BM25, Aggregationen). Kein Ingest, kein `eval.runner`, kein Commit.

## Zusammenfassung

| Schweregrad | Anzahl |
|---|---|
| CRITICAL | 0 |
| HIGH | 1 |
| MEDIUM | 8 |
| LOW | 12 |

Gesamturteil: Die Basis ist solide (SSRF-Schutz, XSS-Abwehr, Rate-Limit, saubere Fehlerseiten, 398 grüne Unit-Tests, ruff/mypy sauber). Der gewichtigste Mangel liegt nicht in der Sicherheit, sondern in der RAG-Qualität: Das "Evidence Gate" hält unbeantwortbare Fragen praktisch nicht ab (H1), und der Default-Provider `extractive` liefert dann "belegte" Antworten mit irrelevanten Zitaten. Das widerspricht dem Kernversprechen "verifiable citations" und ist im Interview der erste Angriffspunkt.

Testlauf (`pytest --cov`): 398 passed, 95 deselected (integration/e2e/llm), Gesamt-Coverage 96 %.

| Kernlogik | Coverage | | Kernlogik | Coverage |
|---|---|---|---|---|
| `ingest/chunk.py` | 100 % | | `rag/citations.py` | 100 % |
| `search/fusion.py` | 100 % | | `rag/answer.py` | 100 % |
| `rag/prompt.py` | 100 % | | `ingest/fetch.py` | 100 % |
| `search/filters.py`, `vector.py` | 100 % | | `search/bm25.py` | 97 % |
| `search/service.py` | 87 % | | `search/rerank.py` | 85 % |
| `search/embedder.py` | 76 % | | `search/indexer.py` | **54 %** |
| `search/mapping.py` | **49 %** | | `logging.py` | 71 % |

Die Lücken (indexer, mapping, embedder, Teile von service) sind genau der ES-gebundene Code (Idempotenz, Alias-Umschaltung); er wird nur von Integrationstests abgedeckt, die CI nicht ausführt (siehe M8).

---

## Befunde

### HIGH

**H1 – Evidence Gate lässt fast alle unbeantwortbaren Fragen durch; Default-Provider liefert Scheinbelege**
`src/finsearch/rag/answer.py:30-60,103-106`, `config.py:56-57`, `rag/providers/extractive.py:34-79`
- Was: `evidence_coverage` zählt den Anteil der Frage-Terme, die *irgendwo in der Vereinigung* aller k=6 Chunks vorkommen (Schwelle 0.4). Generische Terme ("policy rate", "Zürcher Kantonalbank", "UBS") reichen. `rag_min_score=0.0` ist wirkungslos (Scores sind methodenabhängig: BM25, Cosine, RRF, Reranker). Der Kommentar "tuned in Phase 3" ist nicht erfüllt: der Eval-Runner zählt die 8 unbeantwortbaren Fragen nur, wertet sie nicht aus.
- Verifiziert (echter Index, BM25 top-6, die 8 unbeantwortbaren Golden-Fragen): Gate lässt **7 von 8** passieren (nur q050 blockiert), z. B. "Who won the 2026 FIFA World Cup?" (Coverage 0.60), "What dividend per share does UBS pay for 2025?" (1.0). Mit `ExtractiveClient` (Default!) sind 7/8 davon `found=True` mit Zitaten auf irrelevante Sätze (q049: ein Satz zum ZKB-Zürilauf-Cup). Mit einem Fake-LLM wird das LLM 7-mal aufgerufen. Die 43 beantwortbaren Fragen passieren 43/43, die Schwelle trennt also nichts.
- Szenario: Demo-Frage "Wie hoch ist der Bitcoin-Bestand der SNB?" liefert eine Antwort mit Quelle [n], die das Gegenteil von "In den Quellen nicht gefunden" ist.
- Fix: (1) Gate pro bester Passage statt über die Chunk-Vereinigung messen, Terme idf-gewichten bzw. mindestens die seltensten Frage-Terme verlangen. (2) Im Extractive-Pfad verlangen, dass der beste Satz >= X % der Frage-Terme abdeckt. (3) RAG-Eval ergänzen (Quote "unbeantwortbar korrekt abgelehnt" / "beantwortbar nicht fälschlich abgelehnt" pro Methode), Schwelle daran tunen, `rag_min_score` entfernen oder methodenspezifisch machen. (4) Auf der UI/Doku nicht "belegt" versprechen, solange das nicht gemessen ist.

### MEDIUM

**M1 – Gate und Extractive lehnen englische Stichwort-Fragen fälschlich ab (Sprach-Tie -> "de" + deutsches Stemming)**
`rag/text.py:52-57,68-78`, `rag/answer.py:30-44`, `rag/providers/extractive.py:42`
- Was: `detect_lang` entscheidet bei 0:0 Stoppwort-Treffern für "de". Eine englische Frage ohne Stoppwörter wird deutsch gestemmt ("inflation" -> `inflatio`, "intervention" -> `interventio`), der Chunk aber englisch ("inflation").
- Verifiziert: Chunk "Inflation has eased considerably ..." mit Frage `inflation` oder `Inflation?` -> `has_evidence=False`, `extractive_answer=''`; mit "What about inflation?" -> True. Mit dem Gold-Chunk allein lehnt das Gate 5 von 43 beantwortbaren Fragen ab (q020, q022, q026, q028, q036; Best-Case ohne Retrieval-Rauschen, mit top-6 greifen alle 43).
- Fix: Sprache der Frage nicht per Tie->de bestimmen, sondern beide Stemmer probieren (Terme beider Sprachen vergleichen) oder die Frage mit der Sprache des jeweiligen Chunks stemmen (`content_terms(question, detect_lang(chunk.text))` je Chunk). Mindestens: Tie -> Sprache der Chunks. Test mit 1-Wort-Frage ergänzen.

**M2 – Lazy-Modellladen nicht thread-safe; Laden im Request-Pfad**
`search/embedder.py:31-48`, `search/rerank.py:25-30`, `api/routes.py:58-72`, `Dockerfile`
- Was: `if self._model is None: self._model = ...` ohne Lock. Sync-Routen laufen im Threadpool (40 Threads), also laden parallele Erstanfragen das Modell mehrfach.
- Verifiziert: 8 Threads, ein Embedder -> **8** Ladevorgänge (`loading embedding model` 8x geloggt, Laufzeit 19 s). Beim Reranker (bge-reranker-base, ca. 1.1 GB) vervielfacht sich der RAM-Bedarf entsprechend (OOM-Risiko im Container). Keine Tokenizer-Fehler beobachtet.
- Zusätzlich: Kein Warm-up in `lifespan`, im Docker-Image kein Vorab-Download; der erste Request lädt/downloadet ein 470-MB-Modell in der Request-Zeit. Schlägt das Laden des *Embedders* fehl (offline), gibt es ein generisches 500 (der freundliche 503 existiert nur für `hybrid_rerank`).
- Fix: `threading.Lock` (double-checked) in beiden `_load`; in `lifespan` per `run_in_threadpool` vorwärmen (oder `RUN python -c "SentenceTransformer(...)"` im Dockerfile); Lade-Fehler zentral auf 503 mappen.

**M3 – RuntimeStats und Query-Log wachsen mit der Anzahl verschiedener URL-Pfade unbegrenzt**
`api/main.py:107`, `api/deps.py:162-181`, `api/middleware.py:33-62`, `api/querylog.py:149-154`
- Was: Für nicht gematchte Routen ist der Label-Key der rohe Pfad (`obs.get("endpoint") or path`). `_counters` evictet nie; jeder Key hält eine `deque(maxlen=1000)`. Jeder Request (auch 404) schreibt zusätzlich ein Query-Log-Dokument in ES. `/stats/runtime` ist unauthentisiert, nicht rate-limitiert und sortiert alle Deques unter dem Lock.
- Verifiziert: 2000 Requests auf `/probe-N` -> 2000 Endpoint-Keys im Snapshot.
- Szenario: Ein Scanner (oder ein Typo-Crawler) bläht RAM, `/stats/runtime`-Antwort und den ES-Index `finsearch-queries` auf; die Snapshot-Berechnung blockiert `record()` aller Requests.
- Fix: Label aus `scope["route"].path` (Template) oder konstant `"unmatched"`; im Query-Log `path` bei 404 ebenfalls normalisieren; optional Obergrenze für Keys.

**M4 – Re-Indexierung entfernt veraltete Chunks nicht (Idempotenz nur "Überschreiben")**
`search/indexer.py:70-109`, `ingest/pipeline.py:97-104,130-145`, `search/mapping.py:140-147`
- Was: `_id = chunk_id` überschreibt gleiche IDs, löscht aber nie IDs, die im neuen Lauf fehlen. `make ingest` schreibt immer in `finsearch-v1` (Alias zeigt schon darauf). Ändert sich Chunking (Zielgrösse, Overlap, Cleaning) oder verschwindet ein Dokument aus `sources.yaml`, bleiben alte Chunks `...-0042` neben den neuen stehen -> doppelte/veraltete Treffer, falsche Zitate, Golden-Set-IDs passen nicht mehr.
- Fix: Standard-Ingest in einen *neuen* Index `finsearch-v<content_hash>`/Zeitstempel schreiben und danach `switch_alias` (Infrastruktur existiert schon, wird aber nicht benutzt), alten Index löschen; oder vor dem Bulk `delete_by_query` auf `doc_id` der neu geschriebenen Dokumente. `ensure_index` sollte ausserdem bei vorhandenem Index `dims`/`dynamic` prüfen.

**M5 – Keine Reproduzierbarkeit der Abhängigkeiten**
`pyproject.toml:6-20`, `Dockerfile:5-9`, `.github/workflows/ci.yml:9`, `search/rerank.py:30`
- Was: Alle Python-Abhängigkeiten nur mit Untergrenze (`fastapi>=0.115`, `httpx>=0.27`, `sentence-transformers>=3.0,<6`, `pymupdf>=1.24`, ...), kein `uv.lock`/`requirements.lock`. Torch ungepinnt, installiert per `--extra-index-url` (dort gewinnt bei gleicher Version nur die `+cpu`-Variante; kein Schutz gegen PyPI-Variante, Lieferketten-Angriffsfläche). Basis-Image `python:3.13-slim` ohne Digest. Der Reranker `BAAI/bge-reranker-base` ist ungepinnt (nur das Embedding-Modell hat eine Revision in `MODEL_REVISIONS`), die Eval-Tabelle (Reranker Recall@5 0.86) ist also nicht reproduzierbar. Der Report nennt keine Paket-/Modellversionen.
- Fix: `uv lock` committen und im Dockerfile/CI mit `uv sync --frozen`; Torch CPU per `--index-url` pinnen; Reranker-Revision in `MODEL_REVISIONS` aufnehmen und im Eval-Report ausgeben; Image-Digest pinnen.

**M6 – Lizenzhinweise in `sources.yaml` und tatsächliches Verhalten widersprechen sich**
`sources.yaml:254-258,285-287`, `api/schemas.py:125,202`, `web/static/app.js` (`fullText`)
- Was: FINMA ("all rights reserved ... results show short snippets with citation") und ZKB ("no explicit reuse licence ... used locally for indexing only") sind als nicht redistribuierbar eingestuft. API (`HitOut.text`, `ChunkOut.text`) und UI ("Volltext") liefern aber den vollen Chunk-Text (ca. 1.6 KB). `license_note` wird zwar geladen (`ingest/sources.py`), aber nirgends durchgesetzt oder angezeigt.
- Folge: Lokal vertretbar; sobald die Demo öffentlich gehostet wird, verletzt sie die selbst dokumentierten Auflagen (FINMA/ZKB). Im Repo liegen sonst keine Rohdokumente (nur 43 kurze Zitate, 2.9 KB, in `eval/golden_questions.yaml`; `data/raw` und `data/processed` ignoriert).
- Fix: Pro Quelle ein Flag `redistribute: false` -> `text` auf Snippet kürzen/Volltext ausblenden; Lizenzhinweis im UI neben dem Treffer anzeigen; README-Abschnitt "Nur lokaler Betrieb für FINMA/ZKB".

**M7 – Doku: README fehlt, Fortschrittsdatei veraltet, Code-Doku-Widersprüche**
`PROGRESS.md`, `docs/ARCHITECTURE.md:35-36,31`, `CLAUDE.md:5`, `config.py:56`, `search/bm25.py:17-19`, `docs/DECISIONS.md`
- Was: (a) Es gibt kein `README.md` (nur `eval/README.md`, `observability/README.md`); für ein Portfolio-Projekt ist das der Einstieg. (b) `PROGRESS.md` zeigt Phase 0 "in Arbeit" und alle anderen "offen", obwohl alles implementiert ist. (c) ARCHITECTURE: "BM25-Top-50 und kNN-Top-50 parallel" – Code ruft sequenziell auf (`service.py:_hybrid`); "Tie-Break nach `chunk_id`" – Code: erst bester Rang, dann ID (`fusion.py`); "`title`/`url` (text+keyword)" – `url` ist nur `keyword` (`mapping.py`). (d) CLAUDE.md nennt "Jinja/HTMX", die UI ist Vanilla-JS ohne HTMX. (e) Kommentare in `config.py`/`bm25.py` verweisen auf Tuning-Entscheide in `docs/DECISIONS.md`, die dort nicht stehen (kein ADR zu `best_fields`, RRF-k, Boosts). (f) Das Eval-Ergebnis "Hybrid >= bester Einzelmethode: no" (BM25 schlägt Hybrid bei Recall@5 0.83 vs 0.77) widerspricht der Prämisse von ADR-001 und wird in den ADRs nicht eingeordnet.
- Fix: README (Zweck, Quickstart, Architekturbild, Eval-Ergebnisse, Grenzen, Lizenzen), PROGRESS aktualisieren, ARCHITECTURE-Sätze korrigieren, ADR-007 "Tuning-Ergebnisse und warum Hybrid hier nicht gewinnt".

**M8 – CI führt keine Integrationstests aus; ES-gebundene Kernlogik hat 49-54 % Unit-Coverage**
`.github/workflows/ci.yml`, `tests/integration/*`
- Was: 95 Tests (integration/e2e/llm) sind per `addopts` deselektiert; CI hat keinen ES-Service. Dadurch laufen Index-Mapping, Bulk-Indexierung, Alias-Switch und `stats/list_documents/get_chunks` nie in CI. Die Unit-Gate `--cov-fail-under=80` wird durch die hohe Abdeckung der reinen Logik erfüllt und verdeckt das.
- Fix: CI-Job mit `services: elasticsearch` (9.4.8, `xpack.security.enabled=false`) und `pytest -m integration`; die Integrationstests nutzen schon eindeutige Index-Namen (geprüft: kollidieren nie mit `finsearch`).

### LOW

**L1 – Extractive: Fussnoten der Quelle können falsche Zitate erzeugen**
`rag/providers/extractive.py:21,48`, `rag/citations.py:27`
- `_SOURCE_FOOTNOTE_RE` entfernt nur `[n]`. Ein Quellsatz mit `[1, 3]` oder `[1-2]` bleibt stehen und wird von `validate_citations` als echte Zitation gelesen. Verifiziert: Satz aus Chunk 1 mit `[1, 3]` -> `citations=[1, 3]`, obwohl Quelle 3 irrelevant ist. Im Korpus aktuell 0 Vorkommen (latent).
- Fix: Regex auf `\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]` erweitern bzw. dieselbe `_MARKER_RE` verwenden.

**L2 – Kosten-/Lastbegrenzung: ein Budget für alles, `/ask` mit bis zu k=50**
`api/schemas.py:103` (`K = 6`, `le=50`), `api/deps.py:96-120`, `rag/llm_base.py:28-60`
- `/ask` erlaubt k bis 50 (Prompt bis ca. 90 KB) -> unnötige Token-Kosten bei bezahltem Provider; RAG braucht selten > 10. `post_json` kann im schlechtesten Fall ca. 3 x 30 s + Backoff = ca. 97 s einen Threadpool-Thread belegen. `hybrid_rerank` (Median 4.8 s CPU laut Eval) zählt im Rate-Limit gleich viel wie BM25 (1 von 60). Es gibt kein globales Tageslimit für LLM-Aufrufe.
- Rate-Limiter: Die Schranke `MAX_TRACKED_CLIENTS` hält nur, wenn alte Einträge ablaufen; bei > 10 000 *aktiven* Clients (IPv6-Rotation) wird trotzdem eingefügt und bei jedem Request O(n) gesäubert (`deps.py:113-118`). X-Forwarded-For wird nicht gelesen (gut); uvicorn vertraut Proxy-Headern nur von 127.0.0.1. Im Docker-Betrieb sehen alle Clients dieselbe Gateway-IP, es gilt also ein gemeinsames Budget. `/documents`, `/stats`, `/health` sind nicht limitiert (ES-Aggregationen).
- Fix: `/ask`-k auf <= 10 begrenzen; Rerank mit höherem Gewicht zählen; Obergrenze hart durchsetzen (älteste Einträge verwerfen); Hinweis zu `--forwarded-allow-ips` und Docker in der Doku.

**L3 – Ingest-Fetch: Restrisiken und Compliance-Details**
`ingest/fetch.py:76-77,186-196,202-228`, `config.py:43`
- DNS-Rebinding-Fenster (bekannt, im Code als `shortcut:` markiert; bei fester Allowlist akzeptabel). Redirect-Prüfung nur nach Hostname: Port und die Robots-Regeln des Zielhosts werden nicht geprüft. `/robots.txt` mit 3xx (z. B. `snb.ch` -> `www.snb.ch`) wird als "disallow all" behandelt (fail-closed, aber unnötig strikt). Pythons `RobotFileParser` kennt keine `*`/`$`-Wildcards im Pfad und ignoriert `Crawl-delay` -> Regeln wie `Disallow: /*.pdf$` würden fälschlich erlaubt; hier wurde robots.txt je Host manuell geprüft (`sources.yaml`), daher nur LOW. Der Default-User-Agent `FinSearch-Portfolio/0.1 (research)` enthält keinen Kontakt (nur `.env.example` hat einen Platzhalter). Gesamt-Deadline fehlt (30 s pro Operation, langsamer "Slow-Drip" kann lange laufen). Gzip-Bombe: Limit gilt für dekodierte Bytes, aber ein einzelner Chunk kann vor der Prüfung gross sein.
- Fix: Kontaktangabe in den Default-UA; Redirect auf gleiche Origin (Schema+Host+Port) beschränken; gesamte Download-Dauer deckeln.

**L4 – Query-Log: Index wird nur beim Start angelegt; Hash ohne Salz; unbegrenzte Queue**
`api/querylog.py:56-60,109,149-154`, `api/main.py:202-205`
- War ES beim API-Start down, wird `finsearch-queries` später per Auto-Mapping angelegt (alle Strings `text`+`keyword`, Aggregationen/Dashboard auf `endpoint` etc. brechen). `ThreadPoolExecutor(max_workers=1)` hat eine unbegrenzte Queue; bei hängendem ES (5 s Timeout je Write) wächst sie. `close()` wartet auf alle Queue-Einträge. `query_hash` ist ungesalzen (SHA-256, 16 Hex) und für Wörterbuch-Angriffe auf kurze Anfragen umkehrbar; Rohtext nur mit `LOG_RAW_QUERIES=true` (korrekt, verifiziert in `build_doc`).
- Fix: `ensure_index` bei `NotFoundError`/erstem Fehler im Writer nachholen; Queue begrenzen (`queue.Queue(maxsize)` und verwerfen); Hash mit Salt aus Settings.

**L5 – Chunk-Seite zeigt auf die Overlap-Seite; undatierte Chunks fallen aus Datumsfiltern**
`ingest/chunk.py:186`, `search/filters.py:35-40`
- `page=group[0].page` ist die Seite des ersten *Overlap*-Satzes. Verifiziert (Synthetik, 2 Seiten): Chunk 1 hat 7 Sätze von Seite 1 und 25 von Seite 2, wird aber mit "Seite 1" zitiert. Bei fehlenden Seiten nach dem Cleaning (leere Seiten fallen weg) kann die Abweichung > 1 sein. Die Citation sagt dann "Seite 1" für einen Beleg auf Seite 2.
- 201 von 1117 Chunks haben `date = null` und verschwinden bei jedem Datumsfilter stillschweigend (UI erklärt das nicht).
- Fix: Seite des ersten *neuen* Satzes verwenden (oder `page_start`/`page_end`); Hinweis bei Datumsfilter oder `missing`-Handling.

**L6 – `reranker_guard` maskiert beliebige Fehler**
`api/routes.py:58-72`
- Jede Nicht-ES/ValueError-Exception im Pfad `hybrid_rerank` (auch ein echter Bug in BM25-Parsing) wird zu "Reranker-Modell konnte nicht geladen werden" (503). Das Log hat den Stacktrace, die API-Antwort ist irreführend. Der Embedder-Ladefehler für `knn`/`hybrid` bleibt dagegen ein generisches 500.
- Fix: gezielt `OSError`/`ImportError`/`RuntimeError` um die Modell-Ladestellen fangen und dort eine `ModelUnavailable`-Exception werfen (für Embedder und Reranker).

**L7 – Dockerfile/Compose/CI-Härtung**
`Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.github/workflows/ci.yml`
- Container läuft als root (kein `USER`), `pip install -e .` (editable im Image), kein `HEALTHCHECK` im Image (Compose hat einen), `.dockerignore` schliesst `.env` nicht aus (heute kopiert das Dockerfile nur ausgewählte Ordner, daher nur Vorsorge). `OLLAMA_URL=http://127.0.0.1:11434` aus `.env.example` zeigt im Container auf den Container selbst. `kibana-setup` baut dasselbe schwere Image nur für ein Stdlib-Skript.
- CI: kein `permissions: contents: read`, Actions nur per Tag (`@v4`, `@v5`), läuft doppelt bei Push *und* PR.
- Verifiziert positiv: ES (9.4.8) und Kibana sind nur an `127.0.0.1` gebunden (`docker ps` bestätigt), Images versioniert.

**L8 – Toter Code, Duplikate, Magic Numbers**
- Ungenutzt in `src`: `SearchService.get_chunks` (`search/service.py:224`), `ingest/io.load_chunks`/`load_documents` (`ingest/io.py:44,51`; zusätzlich Duplikat von `search/indexer.load_chunks`, strenger vs. tolerant), `eval/runner.chunk_config_label` (`runner.py:64`), `contracts.Method` (`contracts.py:10`), `wait_for_es` nur in Tests.
- `ingest/pipeline.py:137` lädt den Indexer per `importlib` (Rest paralleler Entwicklung); ein normaler Import wäre einfacher und typgeprüft.
- Doppelte Konstanten mit abweichenden Werten: `MAX_QUERY_CHARS` 500 (`schemas.py`) vs 1000 (`filters.py`), `MAX_K` 50 (`schemas.py`) vs 100 (`service.py`), `SNIPPET_CHARS` 300 vs 200; `HTTP_UNAVAILABLE`/`MS_PER_S` mehrfach definiert (`main.py`, `routes.py`, `deps.py`, `middleware.py`).
- `Services` und `/health` fangen beim LLM-Bau nur `LLMError`/`ValueError`; alles andere (z. B. ein Fehler im Provider-Konstruktor) macht `/health` zum 500, obwohl der Docstring "always 200" sagt (`routes.py:~100`).

**L9 – Prompt/Zitat: Designgrenzen und Kleinigkeiten**
`rag/prompt.py:27-41`, `rag/answer.py:120`, `rag/citations.py`
- `html.escape` auf Chunk-Text und Frage macht `<`, `>`, `&` zu Entities ("<0.5 %", "R&D", "S&P") und kann die LLM-Qualität bei Finanzzahlen leicht stören; ausreichend wäre, nur `</source` bzw. die Tags zu neutralisieren.
- `NOT_FOUND_TEXT.lower() in response.text.lower()` verwirft eine sonst gültig zitierte Teil-Antwort, die den Satz irgendwo enthält.
- `validate_citations` prüft nur, dass die Nummer existiert (und Satz-Level-Abdeckung), nicht ob die Quelle die Aussage trägt. Das ist ein bekanntes Limit von Zitat-Validierung; es sollte in README/ADR als solches benannt werden ("verifiable" = Quelle existiert und ist verlinkt, nicht Entailment).

**L10 – UI: Crafted URL bricht die Seite**
`web/static/app.js:195-198`
- `form.elements[name]` mit `name` aus der URL: `?length=1` liefert eine Zahl, `field.value = value` wirft im Strict-Mode einen `TypeError`, `init()` bricht ab (Seite tot, nur Selbstschaden, kein XSS). Per Code-Lesen/JS-Semantik begründet, nicht im Browser ausgeführt.
- Fix: auf `FILTER_FIELDS + ['q','method','k']` whitelisten.

**L11 – Konfiguration/Secrets-Hygiene**
`config.py:50-54`, `.env.example`
- API-Keys sind einfache `str`-Felder; `repr(settings)` (Debug-Log, Exception-Locals) würde sie ausgeben. `SecretStr` verwenden. `.env.example` fehlen u. a. `RRF_K`, `CANDIDATE_POOL`, `RERANK_POOL`, `BOOST_*`, `CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_RATIO`, `INGEST_DELAY_S`, `LLM_TIMEOUT_S`, `RAG_TOP_K`, `RAG_MIN_SCORE` und die Pfade. Positiv: kein Key im Code, in der Git-Historie (4 Commits gescannt) oder in Logs/Fehlern.

**L12 – Eval-Methodik und Ingest-Nebenwirkung**
`eval/build_golden_set.py:35-45`, `eval/runner.py:181`, `ingest/pipeline.py:102-103`
- Relevanz = "Chunk enthält Zitat" und die Fragen wurden aus genau diesen Chunks formuliert -> lexischer Bias zugunsten BM25; Übersetzungs-Chunks (zwei SNB-Reden in de und en) zählen nicht als relevant, das benachteiligt kNN bei cross-lingualer Suche. Parameter (RRF-k, Boosts, Rerank) werden auf denselben 43 Fragen getunt, die berichtet werden (kein Hold-out); die Konfidenzintervalle überlappen stark. Für einen Portfolio-Report: als Einschränkung benennen.
- `run_experiments` baut je Zeile einen neuen `SearchService` (`get_search_service`) und lädt Embedder/Reranker jeweils neu (9x).
- `pipeline.run` überschreibt `documents.jsonl`/`chunks.jsonl` auch bei `--source`/`--limit` oder wenn alle Downloads scheitern; ein Teil-Lauf ersetzt die vollständigen Daten (Rohdaten bleiben gecached, ein Neu-Lauf stellt sie wieder her).

---

## Verified OK

Sicherheit
- **SSRF** (`fetch.py`): nur https, nur URLs aus `sources.yaml` (`allowed_urls`), jede Hop-Adresse per `ipaddress.is_global` geprüft (inkl. IPv4-mapped IPv6, 100.64/10, link-local, loopback: experimentell bestätigt), Redirects manuell mit Hostname-Allowlist und Limit 5, Redirect auf http abgelehnt, Redirect-Schleife begrenzt.
- **Cache-Pfade**: `raw_dir/<sha256[:16]>.<pdf|html>`; kein Pfadanteil aus Nutzereingabe; atomares Schreiben via `.part` + `replace`.
- **Grössenlimits**: 50 MB per Content-Length und gestreamt; PDFs per `max_pages`; verschlüsselte PDFs abgelehnt. **YAML**: überall `safe_load`; kein pickle/eval/exec/subprocess/`shell=True`/`verify=False`/`trust_remote_code` (grep).
- **robots.txt**: wird vor jedem Netz-Download geholt, 4xx -> erlaubt, 401/403/5xx/unerreichbar -> alles verboten (fail-closed); Crawl-Delay >= 1 s pro Host.
- **Prompt-Injection**: Quellen und Frage in `<source>`/`<question>` mit `html.escape` (inkl. `chunk_id`, `title`), Chunk kann `</source>` oder `<source n="9">` nicht fälschen; Systemprompt deklariert Quellen als Daten. Nummern-Zitate werden positionsbasiert gegen die gelieferten Quellen geprüft; erfundene `[7]`, `[0]`, `[2024]` und Chunk-IDs werden entfernt (verifiziert, auch Bereich `[1-3]` nur als Ganzes gültig); ohne gültige Zitation -> `NOT_FOUND_TEXT`. Ein nicht existierendes Zitat kann nicht überleben.
- **XSS**: `app.js` setzt API-Daten nur als Textknoten (`h()`-Builder), URLs nur `http(s)` mit `rel=noopener`; Highlights serverseitig doppelt abgesichert (ES `encoder: html` + `safe_highlight`) und clientseitig als echte `<em>`-Elemente neu gebaut (`decode` per `textarea`, Eingabe enthält kein `<`); Jinja-Autoescape an, keine `|safe`-Nutzung; LLM-Antwort wird nur als Text mit verlinkten `[n]` gerendert. CSP `default-src 'none'; script-src 'self'; style-src 'self'` ohne inline/eval, plus `nosniff`, `X-Frame-Options`, `Referrer-Policy`; Header auch auf 404/422/429/500 (Test + Experiment).
- **CORS**: Wildcard wird verworfen, nur GET/POST, nur `Content-Type`, keine Credentials. **CSRF**: `text/plain`-POST an `/ask` -> 422 (JSON-Content-Type erzwungen, verifiziert).
- **ES-Injection**: Filter sind `term`/`terms`/`range` mit Slug-Regex bzw. Enum; Query nur als `multi_match`-String (`title:* OR _exists_:text`, unbalancierte Anführungszeichen und `\` getestet, kein Fehler, keine Sonderbehandlung); `/documents` `source` nur als `term`-Wert. Unbekannte Filterfelder -> 422 (`extra=forbid` bei `/ask`).
- **Fehlerseiten**: generisches 500/503, `detail` der Validierung ohne `input`; Test belegt, dass `RuntimeError("secret")` nicht in der Antwort erscheint. **Secrets**: Keys nur in Headern (Gemini per `x-goog-api-key`, nicht in der URL), `LLMError`-Texte enthalten nur Provider + Status; keine Keys in Git (inkl. Historie), kein `.env` im Repo, `data/raw`, `data/processed`, `.env` ignoriert. Keine Claude-Trailer in Commit-Messages.
- **Static-Pfad-Traversal** (`/static/../pyproject.toml`, `%2e%2e`, `..%2f`): alle 404.
- **ES/Kibana-Exposition**: nur `127.0.0.1` (Compose + laufende Container).

Korrektheit
- **RRF** (`fusion.py`): `1/(k+rank)`, 1-basiert, Duplikate nur beste Position, `fsum` für exakte Ties, Tie-Break bester Rang dann ID; Handrechnungs-Tests inkl. exakter Ties.
- **Filter** werden in BM25 (`bool.filter`) und kNN (`knn.filter`, Pre-Filter) aus derselben Funktion gebaut; `num_candidates = clamp(3k, 50, 10000)`; Datumsfilter inklusive beider Grenzen (gegen echten Index geprüft: from=to=2025-10-22 -> nur dieses Datum), `date_from > date_to` -> 422.
- **Chunker**: leere Eingabe -> `[]`; keine Satzverluste (63/63 Sätze über 3 Seiten wiedergefunden); 5000-Zeichen-Wort und Satzlosigkeit werden hart geteilt, harte Obergrenze eingehalten; Overlap vorhanden; deterministische IDs `doc_id-0000`.
- **Evidence-Gate-Reihenfolge**: Gate läuft vor dem LLM-Aufruf (`answer.py:106`) – das Problem ist die Trennschärfe (H1), nicht die Reihenfolge. LLM-Fehler fallen auf Extractive zurück, ohne Exception nach aussen.
- **Middleware**: reine ASGI-Middleware, Exceptions werden geloggt und re-raised (nicht verschluckt), Query-Log-Fehler maskieren das Ergebnis nicht, ES-Write läuft im Hintergrund-Thread (blockiert den Event-Loop nicht), Routen sind `def` (Threadpool) und `lifespan` nutzt `run_in_threadpool`.
- Services-Erzeugung (`Services`) per Lock thread-safe; Rate-Limiter-Zugriff per Lock; Embedding-Dimension wird beim Laden geprüft.
- Query-Log-PII: ohne `LOG_RAW_QUERIES` weder in ES noch in der JSON-Logzeile (`build_doc`), nur Hash + Länge.
- Statische Prüfungen: `ruff check`, `ruff format --check`, `mypy src` ohne Befund.

## Nicht geprüft

- `docker build`/`docker compose up` des `api`-Images (schwerer Torch-Download, CPU belegt); ob `up --wait` mit dem einmaligen `kibana-setup`-Container sauber durchläuft, ist ungeprüft (im Moment laufen nur ES und Kibana).
- Kein Browser-/Playwright-Lauf (e2e), keine visuelle/Accessibility-Prüfung; L10 nur per Code-Lesen.
- Kein Lasttest (Threadpool-Sättigung, Tokenizer-Concurrency über die 8 Threads hinaus, `/ask` mit echtem LLM).
- Echte LLM-Provider (Gemini/Groq/Anthropic/Ollama) nicht aufgerufen; Antwortformate nur per Test-Fakes.
- Semantische Güte der 47 extrahierten Dokumente (Text-/Seitenqualität, Cleaning-Heuristiken auf echten PDFs), PDF-Parser-Robustheit gegen bösartige PDFs (pymupdf/trafilatura sind C-/Parser-Code).
- `eval/report.py`, `eval/results.py` nur überflogen; Metriken (`metrics.py`) auf Definitionen geprüft, nicht gegen eine Referenzimplementierung.
- Kibana-NDJSON, Alert-ES|QL und `scripts/take_screenshots.py` nur oberflächlich gelesen; `observability/seed_demo_queries.py` schreibt in den echten Index `finsearch-queries` (mit `synthetic: true`, `--delete` vorhanden), wurde nicht ausgeführt.
- Bekannte Schwachstellen der gepinnten Paketversionen (kein SCA-Scan).
