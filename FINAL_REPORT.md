# Abschlussbericht FinSearch (v0.1.0)

## 1. Status Definition of Done
| Kriterium | Status |
|---|---|
| `make up` startet ES, Kibana, API+UI, Dashboard-Import | **erfüllt** (ausgeführt, alle Dienste healthy); `make down` nicht separat getestet |
| `make ingest` ≥ 30 Dokumente, ≥ 3 Quellen | **erfüllt**: 47 Dokumente, 5 Quellen, 1117 Chunks (ausgeführt) |
| `make test` grün, Kernlogik ≥ 80 % | **erfüllt**: 401 Unit-Tests, Gesamt 96 %, Chunking/RRF/Prompt/Zitate/Answer je 100 %; Integration 85 grün (lokal); E2E 7 grün (zuvor durch Web-Agent) |
| `make eval`, Golden Set ≥ 40 Fragen | **erfüllt**: 51 Fragen (43 beantwortbar) |
| Hybrid ≥ beste Einzelmethode (R@5, MRR) | **nicht erfüllt** – Hybrid 0.767/0.640 vs. BM25 0.826/0.685; analysiert und offengelegt (ADR-008). Hybrid+Rerank (0.860/0.743) ist am besten |
| RAG mit Zitaten, Zitat-Check, «nicht gefunden» | **erfüllt** (8/8 negativ, 40/43 positiv; auf demselben Set getunt, optimistisch) |
| Kibana-Dashboard als NDJSON, auto-Import | **erfüllt** (Import zweimal und in `make up` ausgeführt) |
| Läuft ohne bezahlten Key | **erfüllt** (extractive). Echte LLM-Provider (Ollama/Gemini/Groq/Anthropic) nur gegen Mock-HTTP getestet, **nie gegen echte APIs** |
| README + ARCHITECTURE + DECISIONS + INTERVIEW_NOTES | **erfüllt** |
| GitHub Actions: ruff, mypy, Tests, Docker-Build | geschrieben, **nie auf GitHub ausgeführt**; Integrationstests laufen dort nicht |
| Keine Secrets / keine Rohdokumente im Repo | erfüllt (`data/` ignoriert, `.env` nicht versioniert) |
| `FINAL_REPORT.md` | dieses Dokument |

## 2. Messwerte
Siehe README und `reports/eval_results.md`. Latenz Median: BM25 11 ms, kNN 15 ms, Hybrid 57 ms, Hybrid+Rerank ca. 5 s (CPU, Mac). Coverage 96 %.

## 3. Was wirklich ausgeführt wurde
`make setup/up/ingest/test/test-int`, `make eval` inkl. `--experiments`, `eval/rag_gate_eval.py`, ruff, mypy, Screenshots, Kibana-Import. **Nur geschrieben bzw. nicht ausgeführt:** CI-Workflow, echte LLM-Provider, Chunk-Grössen-Experiment, Last-/Performancetests.

## 4. Entscheidungen / Abweichungen
Eigenes Git-Repo in `finsearch/` (ADR-005); kein Plan Mode (ADR-006); RAG/Observability-Agenten früher gestartet als im Phasenplan; Evidence-Gate neu (ADR-007); Token-Schätzung `Zeichen/4`; ECB ausgelassen (robots Crawl-delay 5).

## 5. Bekannte Mängel
Offene Review-Punkte (`reports/review.md`): M4 Re-Index entfernt veraltete Chunks nicht; M6 Lizenzfrage FINMA/ZKB (UI zeigt vollen Chunk-Text); M8 CI ohne Integrationstests; Reranker nicht revisionsgepinnt; Docker-Image lädt Modelle erst beim ersten Request (~10 s); Golden Set formulierungsnah (BM25-Bias); Gate lehnt einige Paraphrasen ab; Rate-Limit pro IP/Prozess; Kibana zeigt eigenen «not secure»-Hinweis.

## 6. Skills/Agenten
Subagenten: Ingestion, Search, Eval, RAG, Observability, Web, Reviewer (7). Nicht genutzt/vorhanden: `frontend-design`, `engineering:*`/`data:*`-Skills wurden nicht aufgerufen; gstack nicht verwendet. Kein separater Docs/QA-Agent; Clean-Room-Test von mir selbst gemacht.

## 7. Offene Fragen an dich
1. FINMA-/ZKB-Dokumente im Korpus behalten (Lizenz)? 2. Golden-Set-Fragen kurz gegenlesen. 3. Soll ein unabhängig formuliertes Test-Set folgen?

## Lebenslauf (2 Zeilen)
FinSearch – Hybrid-Suche und RAG für Schweizer Finanzdokumente: Elasticsearch (BM25 de/en + kNN + RRF, Cross-Encoder), FastAPI, Kibana-Observability, reproduzierbare Evaluation (Recall/MRR/nDCG); 96 % Testabdeckung, Docker Compose.
## Motivationsschreiben (3 Sätze)
In meinem Projekt FinSearch habe ich Hybrid-Suche und quellenbelegte Antworten über SNB-, Fed- und Bankdokumente gebaut und ehrlich evaluiert – inklusive der Erkenntnis, dass reine Vektorsuche und naive Fusion BM25 nicht automatisch schlagen. Dabei habe ich Testing, Observability und saubere Dokumentation von Entscheiden gelebt. Genau diese Verbindung aus Search/AI und Engineering möchte ich im Schweizer Finanzsektor einbringen.
## 5-Minuten-Demo
1 min `make up`/Dashboard zeigen; 2 min UI «Vergleich» mit «Teuerung und Zinsen»; 1 min «Frage stellen» (ZKB-Dividende) und FIFA-Frage («nicht gefunden»); 1 min Eval-Tabelle + Ehrlichkeit zu Hybrid vs. BM25.
