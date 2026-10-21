# Evaluation

Misst, wie gut BM25, kNN, Hybrid (RRF) und Hybrid+Rerank die richtigen Chunks finden. Code: `src/finsearch/eval/`.

## Golden Set: Format

`eval/golden_set.jsonl`, eine JSON-Zeile pro Frage:

```json
{"id": "q001", "question": "...", "lang": "de", "type": "fact",
 "relevant_chunk_ids": ["<doc_id>-0007"], "relevant_doc_ids": ["<doc_id>"],
 "answerable": true, "reference_answer": "...", "provenance": "human"}
```

| Feld | Werte |
|---|---|
| `lang` | `de`, `en` |
| `type` | `fact`, `paraphrase`, `number_date`, `comparison`, `unanswerable` |
| `answerable` | `false` genau dann, wenn `type = unanswerable` |
| `provenance` | `human` (selbst geschrieben) oder `llm_suggested_verified` (LLM-Vorschlag, von Hand geprüft) |
| `reference_answer` | kurze Soll-Antwort oder `null` |

**Relevanz:** Ein Treffer ist relevant, wenn seine `chunk_id` in `relevant_chunk_ids` steht. Ist diese Liste leer, zählt
stattdessen die `doc_id` (`relevant_doc_ids`, gröbere Dokument-Ebene). Beantwortbare Fragen brauchen mindestens eine der beiden
Listen. Unbeantwortbare Fragen werden nicht in die Retrieval-Metriken einbezogen, nur gezählt.
Das Format wird beim Laden geprüft (`finsearch.eval.golden.load_golden`); Fehler nennen Datei und Zeilennummer.

`tests/fixtures/eval_golden_sample.jsonl` ist ein **synthetisches** Beispiel nur für Tests, kein echtes Golden Set.

## Golden Set erstellen (Ablauf)

1. Nach der Ingestion die Chunks (`data/processed/`) durchsehen und Fragen formulieren, verteilt über Quellen, beide
   Sprachen und alle Fragetypen (inkl. einiger unbeantwortbarer, die nicht im Korpus stehen).
2. Fragen dürfen von einem LLM vorgeschlagen werden, dann `provenance = llm_suggested_verified`. Selbst geschriebene Fragen: `human`.
3. **Jede** Frage gegen den Text des Quell-Chunks prüfen: Steht die Antwort wirklich dort? `chunk_id` aus dem Chunk-Export
   übernehmen (nicht raten). Weitere Chunks mit derselben Antwort ebenfalls eintragen, sonst wird Recall unterschätzt.
4. Paraphrasen-Fragen bewusst ohne Wortlaut aus dem Chunk formulieren, sonst bevorzugen sie BM25.
5. Bei Re-Chunking ändern sich die `chunk_id`s: dann Golden Set neu mappen (oder auf `relevant_doc_ids` ausweichen).
6. Datei validieren: `python -c "from pathlib import Path; from finsearch.eval.golden import load_golden; print(len(load_golden(Path('eval/golden_set.jsonl'))))"`.

## Ausführen

```bash
python -m finsearch.eval.runner                       # alle Methoden, k=10 -> reports/eval_results.md
python -m finsearch.eval.runner --methods bm25,hybrid --k 10 --out reports/eval_results.md
python -m finsearch.eval.runner --experiments         # zusätzlich Sweeps (rrf_k, Boosts, Rerank an/aus)
python -m finsearch.eval.runner --label chunk200_overlap0.12 --out reports/eval_chunk200.md   # Chunk-Größen-Lauf
```

Voraussetzung: Elasticsearch läuft und der Index ist befüllt (`make up && make ingest`). Der Runner nutzt
`finsearch.search.service.get_search_service`. Ist das Rerank-Modell nicht ladbar, wird `hybrid_rerank` übersprungen
(Hinweis im Report). Der erste Aufruf pro Methode ist ein Warm-up und zählt nicht zur Latenz. Der Chunk-Größen-Vergleich
läuft über Re-Ingest durch den Menschen/Lead; `chunk_config_label(settings)` liefert ein passendes Label. Ausgaben:
`reports/eval_results.md`, `reports/eval_results.json`, `reports/figures/*.png`.

## Report lesen

- **Recall@k**: Anteil der relevanten Chunks, die in den Top k stehen (bei genau einem relevanten Chunk = Trefferquote).
- **MRR**: Mittelwert von 1/Rang des ersten relevanten Treffers (0 bei keinem Treffer).
- **nDCG@10** (binär): `DCG = Σ rel_i / log2(i+1)`, geteilt durch das ideale DCG. 1 = alle relevanten Chunks ganz oben.
- **Latenz**: Median und p95 von `total_ms`; dazu mittlere Zeiten je Stufe (`timings_ms`).
- **[a–b]**: 95-%-Bootstrap-Konfidenzintervall (1000 Resamples, Seed fest) für den Mittelwert über Fragen. Überlappen die
  Intervalle zweier Methoden, ist der Unterschied bei dieser Fragenzahl nicht belastbar.
- **Verdict-Zeile**: `Hybrid >= best single method on Recall@5 and MRR: yes/no`, ehrlich berechnet, ohne Interpretation.
- **Fehleranalyse**: Fragen, bei denen eine Methode keinen relevanten Treffer in den Top k fand, sowie Fragen, bei denen
  Hybrid besser oder schlechter als die beste Einzelmethode rankt. Diese Listen zuerst lesen, bevor man Zahlen verallgemeinert.
- Duplikate: Mehrere Treffer mit demselben Schlüssel zählen nur einmal (relevant bei Dokument-Ebene).
