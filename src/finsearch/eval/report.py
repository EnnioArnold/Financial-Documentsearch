"""Render the evaluation report (Markdown + JSON + matplotlib charts)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from finsearch.eval.results import (
    NDCG_K,
    RECALL_KS,
    EvalRun,
    ExperimentResult,
    MethodSummary,
    breakdown,
)

log = logging.getLogger(__name__)

METHOD_LABELS = {
    "bm25": "BM25",
    "knn": "kNN",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid+Rerank",
}
# Okabe-Ito palette (colour-blind safe)
METHOD_COLORS = {
    "bm25": "#0072B2",
    "knn": "#E69F00",
    "hybrid": "#009E73",
    "hybrid_rerank": "#CC79A7",
}
SINGLE_METHODS = ("bm25", "knn")
FIG_QUALITY = "eval_quality.png"
FIG_LATENCY = "eval_latency.png"
EPS = 1e-9
NDCG_KEY = f"ndcg@{NDCG_K}"
QUESTION_PREVIEW_CHARS = 90


def _ci(summary: MethodSummary, key: str) -> str:
    lo, hi = summary.ci[key]
    return f"{summary.scores[key]:.3f} [{lo:.3f}–{hi:.3f}]"


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _preview(run: EvalRun, qid: str) -> str:
    text = run.questions[qid].question if qid in run.questions else ""
    return text if len(text) <= QUESTION_PREVIEW_CHARS else text[: QUESTION_PREVIEW_CHARS - 1] + "…"


def _rank_key(rank: int | None) -> float:
    return float("inf") if rank is None else float(rank)


def _fmt_rank(rank: int | None) -> str:
    return "–" if rank is None else str(rank)


def _main_table(run: EvalRun) -> list[str]:
    head = ["Methode", *[f"Recall@{k}" for k in RECALL_KS], "MRR", f"nDCG@{NDCG_K}"]
    head += ["Latenz Median (ms)", "Latenz p95 (ms)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for method, res in run.results.items():
        s = res.summary
        cells = [_label(method)]
        for k in RECALL_KS:
            key = f"recall@{k}"
            cells.append(_ci(s, key) if key == "recall@5" else f"{s.scores[key]:.3f}")
        cells += [_ci(s, "mrr"), f"{s.scores[NDCG_KEY]:.3f}"]
        cells += [f"{s.latency_median_ms:.1f}", f"{s.latency_p95_ms:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def verdict(run: EvalRun) -> tuple[str, str]:
    """('yes'|'no'|'n/a', explanation) for: hybrid >= best single method on Recall@5 and MRR."""
    singles = [m for m in SINGLE_METHODS if m in run.results]
    if "hybrid" not in run.results or not singles:
        return "n/a", "Hybrid oder beide Einzelmethoden fehlen im Lauf."
    hyb = run.results["hybrid"].summary
    parts: list[str] = []
    ok = True
    for key, name in (("recall@5", "Recall@5"), ("mrr", "MRR")):
        best_m = max(singles, key=lambda m: run.results[m].summary.scores[key])
        best = run.results[best_m].summary
        good = hyb.scores[key] >= best.scores[key] - EPS
        ok &= good
        overlap = hyb.ci[key][0] <= best.ci[key][1] and best.ci[key][0] <= hyb.ci[key][1]
        parts.append(
            f"{name}: Hybrid {hyb.scores[key]:.3f} vs. {_label(best_m)} {best.scores[key]:.3f}"
            + (" (Konfidenzintervalle überlappen)" if overlap else "")
        )
    return ("yes" if ok else "no"), "; ".join(parts)


def _breakdown_tables(run: EvalRun) -> list[str]:
    lines: list[str] = []
    for attr, title in (("qtype", "Fragetyp"), ("lang", "Sprache")):
        groups = sorted({g for r in run.results.values() for g in breakdown(r.outcomes, attr)})
        head = [title, "n", *[f"{_label(m)} R@5 / MRR" for m in run.results]]
        lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for g in groups:
            per = {m: breakdown(r.outcomes, attr).get(g) for m, r in run.results.items()}
            n = next((v[0] for v in per.values() if v), 0)
            cells = [g, str(n)] + [f"{v[1]:.2f} / {v[2]:.2f}" if v else "–" for v in per.values()]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _failure_section(run: EvalRun) -> list[str]:
    lines = ["## Fehleranalyse", "", f"### Fragen ohne relevanten Treffer in den Top {run.k}"]
    for method, res in run.results.items():
        failed = [o.qid for o in res.outcomes if o.first_rank is None]
        lines += ["", f"**{_label(method)}**: {len(failed)} von {res.summary.n}"]
        lines += [f"- `{qid}` – {_preview(run, qid)}" for qid in failed]
    lines += ["", "### Wo hilft oder schadet Hybrid?", ""]
    singles = [m for m in SINGLE_METHODS if m in run.results]
    if "hybrid" not in run.results or not singles:
        return lines + ["Nicht auswertbar (Hybrid oder Einzelmethoden fehlen)."]
    helps: list[str] = []
    hurts: list[str] = []
    for o in run.results["hybrid"].outcomes:
        ranks = {
            m: next(x for x in run.results[m].outcomes if x.qid == o.qid).first_rank
            for m in singles
        }
        best_m = min(singles, key=lambda m: _rank_key(ranks[m]))
        entry = (
            f"- `{o.qid}`: Hybrid Rang {_fmt_rank(o.first_rank)}, bester Einzelner "
            f"({_label(best_m)}) Rang {_fmt_rank(ranks[best_m])} – {_preview(run, o.qid)}"
        )
        if _rank_key(o.first_rank) < _rank_key(ranks[best_m]):
            helps.append(entry)
        elif _rank_key(o.first_rank) > _rank_key(ranks[best_m]):
            hurts.append(entry)
    lines += [f"**Hybrid besser als die beste Einzelmethode** ({len(helps)}):", *helps]
    lines += ["", f"**Hybrid schlechter als die beste Einzelmethode** ({len(hurts)}):", *hurts]
    return lines


def _experiment_section(exps: list[ExperimentResult]) -> list[str]:
    head = ["Variante", "Methode", "Recall@5", "MRR", f"nDCG@{NDCG_K}", "Latenz Median (ms)"]
    lines = ["## Experimente", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in exps:
        if e.summary is None:
            lines.append(f"| {e.label} | {_label(e.method)} | nicht ausführbar: {e.note} | | | |")
            continue
        s = e.summary
        lines.append(
            f"| {e.label} | {_label(e.method)} | {s.scores['recall@5']:.3f} | {s.scores['mrr']:.3f} "
            f"| {s.scores[NDCG_KEY]:.3f} | {s.latency_median_ms:.1f} |"
        )
    return lines


def _limitations(run: EvalRun) -> list[str]:
    prov = ", ".join(f"{k}: {v}" for k, v in run.provenance_counts.items())
    return [
        "## Ehrliche Einschränkungen",
        "",
        f"- **Kleines Golden Set**: {run.n_answerable} beantwortbare Fragen. Die Konfidenzintervalle "
        "sind breit; kleine Unterschiede zwischen Methoden sind statistisch nicht belastbar.",
        "- **Domänen-Bias**: Das Korpus besteht aus wenigen öffentlichen Quellen (SNB, Fed, Schweizer "
        "Institute); Ergebnisse lassen sich nicht auf andere Domänen übertragen.",
        "- **Ein Embedding-Modell**: Nur `multilingual-e5-small` wurde getestet; kNN- und "
        "Hybrid-Ergebnisse gelten nur für dieses Modell.",
        f"- **Herkunft der Fragen** (provenance): {prov or 'keine Angaben'}. Als `llm_suggested_verified` "
        "markierte Fragen wurden von einem LLM vorgeschlagen und anschließend von Hand gegen den "
        "Quelltext geprüft; sie können dennoch formulierungsnah an den Chunks liegen und BM25 begünstigen.",
        "- **Relevanzurteile** sind binär und pro Frage nur für die markierten Chunks/Dokumente "
        "vollständig; weitere relevante Chunks gelten als nicht relevant (unterschätzt Recall/Precision).",
        "- **Latenzen** stammen von einem einzelnen lokalen Lauf (ein Warm-up-Aufruf ausgeschlossen) "
        "und sind nur relativ zueinander aussagekräftig.",
        f"- Unbeantwortbare Fragen ({run.n_unanswerable}) fließen nicht in die Retrieval-Metriken ein.",
    ]


def render_markdown(run: EvalRun, figures: list[str] | None = None) -> str:
    """Build the Markdown report. ``figures`` are paths relative to the report file."""
    lines = ["# FinSearch – Evaluation", ""]
    lines += [f"- Datum: {run.run_date.isoformat()}"]
    if run.label:
        lines += [f"- Lauf: {run.label}"]
    lines += [
        f"- Golden Set: {run.n_total} Fragen, davon {run.n_answerable} beantwortbar "
        f"(ausgewertet) und {run.n_unanswerable} unbeantwortbar (nur gezählt)",
        f"- Cutoff k: {run.k}",
    ]
    if run.index_stats:
        lines += ["- Index: " + ", ".join(f"{k}={v}" for k, v in sorted(run.index_stats.items()))]
    for method, reason in run.skipped.items():
        lines += [f"- **Übersprungen: {_label(method)}** – nicht verfügbar ({reason})"]
    lines += ["", "## Vergleich der Methoden", ""]
    if run.results:
        lines += _main_table(run)
        lines += ["", "Konfidenzintervalle: Bootstrap, 1000 Resamples, 95 %, fester Seed."]
        stage_rows = [
            f"- {_label(m)}: "
            + ", ".join(f"{s} {v:.1f} ms" for s, v in r.summary.stage_means_ms.items())
            for m, r in run.results.items()
            if r.summary.stage_means_ms
        ]
        if stage_rows:
            lines += ["", "Mittlere Latenz je Stufe:", *stage_rows]
        answer, why = verdict(run)
        lines += ["", f"**Hybrid >= best single method on Recall@5 and MRR: {answer}** – {why}"]
    else:
        lines += ["Keine Methode konnte ausgeführt werden."]
    for fig in figures or []:
        lines += ["", f"![{Path(fig).stem}]({fig})"]
    if run.results:
        lines += ["", "## Aufschlüsselung (Recall@5 / MRR)", *_breakdown_tables(run), ""]
        lines += _failure_section(run)
    if run.experiments:
        lines += ["", *_experiment_section(run.experiments)]
    lines += ["", *_limitations(run), ""]
    return "\n".join(lines)


def to_json(run: EvalRun) -> dict[str, object]:
    methods = {m: {"summary": asdict(r.summary), "per_query": [asdict(o) for o in r.outcomes]}
               for m, r in run.results.items()}  # fmt: skip
    return {
        "date": run.run_date.isoformat(),
        "label": run.label,
        "k": run.k,
        "n_total": run.n_total,
        "n_answerable": run.n_answerable,
        "n_unanswerable": run.n_unanswerable,
        "provenance_counts": run.provenance_counts,
        "index_stats": run.index_stats,
        "verdict_hybrid_ge_best_single": verdict(run)[0],
        "methods": methods,
        "skipped": run.skipped,
        "experiments": [asdict(e) for e in run.experiments],
    }


def write_figures(run: EvalRun, figures_dir: Path) -> list[str]:
    """Write the PNG charts; returns their file names. Empty if nothing ran."""
    if not run.results:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    methods = list(run.results)
    metric_keys = [("recall@5", "Recall@5"), ("mrr", "MRR"), (NDCG_KEY, f"nDCG@{NDCG_K}")]

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs = [j + i * width for j in range(len(metric_keys))]
        vals = [run.results[m].summary.scores[key] for key, _ in metric_keys]
        ax.bar(xs, vals, width, label=_label(m), color=METHOD_COLORS[m])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(metric_keys))])
    ax.set_xticklabels([name for _, name in metric_keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(f"Retrieval quality by method (n={run.n_answerable} questions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / FIG_QUALITY, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[o.total_ms for o in run.results[m].outcomes] for m in methods]
    box = ax.boxplot(data, patch_artist=True, tick_labels=[_label(m) for m in methods])
    for patch, m in zip(box["boxes"], methods, strict=True):
        patch.set_facecolor(METHOD_COLORS[m])
    ax.set_yscale("log")
    ax.set_ylabel("Latency per query [ms, log scale]")
    ax.set_title("Query latency by method")
    fig.tight_layout()
    fig.savefig(figures_dir / FIG_LATENCY, dpi=150)
    plt.close(fig)
    return [FIG_QUALITY, FIG_LATENCY]


def write_report(run: EvalRun, out_md: Path) -> None:
    """Write <out_md>, <out_md stem>.json next to it and charts into <out_md dir>/figures/."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    names = write_figures(run, out_md.parent / "figures")
    out_md.write_text(render_markdown(run, [f"figures/{n}" for n in names]), encoding="utf-8")
    out_md.with_suffix(".json").write_text(
        json.dumps(to_json(run), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    log.info("wrote %s", out_md)
