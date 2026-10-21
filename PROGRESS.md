# Fortschritt (Stand 2026-10-21, Sitzung wegen Nutzungslimit beendet)

| Phase | Status | Notizen |
|---|---|---|
| 0 Gerüst & Plan | fertig | ADR-001..006, compose, CI-Gerüst |
| 1 Ingestion | fertig | 47 Dokumente, 5 Quellen, 1117 Chunks, idempotent |
| 2 Index & Suche | fertig | BM25/kNN/Hybrid/Rerank, 124 Tests, 98 % Coverage Search-Paket |
| 3 RAG | teilweise | Code + Tests fertig; Evidence-Gate trennt schlecht (Review H1), nicht auf Golden Set getunt |
| 4 API & UI | fertig | Unit/Integration/E2E grün; **Screenshots `docs/screenshots/ui_*.png` fehlen noch** |
| 5 Evaluation | teilweise | Baseline gemessen (reports/eval_results.md); Experiment-Sweep lief im Hintergrund, Ergebnis ungeprüft; Hybrid < BM25 (ehrlich dokumentieren) |
| 6 Observability | fertig | Dashboard-NDJSON + Screenshot (mit synthetischen Demo-Queries) |
| 7 Härtung & Doku | **offen** | Review liegt vor (reports/review.md, 0 critical / 1 high / 8 medium); Fixes, README, INTERVIEW_NOTES, FINAL_REPORT, Clean-Room-Test, Tag v0.1.0 fehlen |

## Nächste Schritte
1. Review H1 beheben (Evidence-Gate auf Golden Set tunen, extractive-Antworten bei Nicht-Evidenz ablehnen), dann M1–M5.
2. `make eval --experiments` Ergebnis prüfen, Hybrid-vs-BM25-Befund als ADR und im README erklären.
3. Screenshots (`scripts/take_screenshots.py`), README.md, docs/INTERVIEW_NOTES.md, FINAL_REPORT.md, Clean-Room-Test, `git tag v0.1.0`.
