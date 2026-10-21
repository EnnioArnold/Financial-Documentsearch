# Alerts

Drei einfache Regeln auf dem Query-Log (`finsearch-queries`). Jede Regel ist eine ES|QL-Abfrage, die nur dann eine Zeile liefert, wenn der Schwellenwert überschritten ist. Kibana löst den Alert aus, sobald die Abfrage Zeilen zurückgibt.

| Regel | Bedingung | Zeitfenster | Warum |
|---|---|---|---|
| Latenz | p95 von `total_ms` > 2000 ms | 5 min | Nutzer warten spürbar; meist Reranker oder LLM-Provider |
| Fehlerrate | Anteil `error: true` > 5 % (mind. 20 Anfragen) | 15 min | 5xx-Fehler, z. B. ES oder LLM nicht erreichbar |
| Zero-Hit-Rate | Anteil `zero_hits: true` > 30 % (mind. 20 Anfragen) | 60 min | Index leer/veraltet oder Filter/Sprache falsch |

"Mind. 20 Anfragen" verhindert Fehlalarme bei wenig Verkehr (1 von 2 Anfragen = 50 %). 4xx-Antworten zählen nicht als Fehler (`error` ist nur bei Status >= 500 oder unbehandelter Exception wahr).

## ES|QL-Abfragen

Die Abfragen wurden gegen `finsearch-queries` getestet (mit dem Zeitfenster der Demo-Daten und niedrigeren Schwellen, damit eine Zeile erscheint).

```esql
// Latenz: p95 > 2 s in 5 min
FROM finsearch-queries
| WHERE @timestamp > NOW() - 5 minutes AND endpoint IS NOT NULL
| STATS p95 = PERCENTILE(total_ms, 95)
| WHERE p95 > 2000
```

```esql
// Fehlerrate > 5 % in 15 min
FROM finsearch-queries
| WHERE @timestamp > NOW() - 15 minutes
| STATS total = COUNT(*), errors = COUNT(CASE(error == true, 1, null))
| EVAL rate = errors * 100.0 / total
| WHERE total >= 20 AND rate > 5
```

```esql
// Zero-Hit-Rate > 30 % in 60 min
FROM finsearch-queries
| WHERE @timestamp > NOW() - 60 minutes AND zero_hits IS NOT NULL
| STATS total = COUNT(*), zero = COUNT(CASE(zero_hits == true, 1, null))
| EVAL rate = zero * 100.0 / total
| WHERE total >= 20 AND rate > 30
```

## Regel in Kibana anlegen

1. Kibana öffnen (`http://127.0.0.1:5601`), **Management → Alerts and Insights → Rules → Create rule**.
2. Regeltyp **Elasticsearch query** wählen, Abfragetyp **ES|QL**.
3. Eine der Abfragen oben einfügen. Zeitfeld `@timestamp`, Bedingung "Anzahl der Zeilen **über 0**".
4. Prüfintervall 1 min (Latenz) bzw. 5 min (Fehler, Zero-Hit); Name z. B. `FinSearch: p95 > 2 s`.
5. Unter **Actions** einen Connector wählen. Lokal reicht **Server log** (kostenlos, erscheint im Kibana-Log); produktiv E-Mail oder Slack.
6. Speichern. Test: `python observability/seed_demo_queries.py` schreibt synthetische Daten, deren p95 über 2 s liegt (LLM-Anfragen, `gemini`/`groq`); die Abfrage zeigt dann eine Zeile, wenn man das Zeitfenster testweise auf 8 Stunden erweitert.

Hinweis: Die Regeln sind nicht im Dashboard-NDJSON enthalten (Regeln brauchen einen Connector und sind umgebungsspezifisch), daher werden sie von `setup_kibana.py` nicht angelegt. Das Anlegen per Alerting-API (`POST /api/alerting/rule`) wurde lokal ohne Erfolg versucht (HTTP 500 ohne Detail); die obigen UI-Schritte wurden nicht in der Oberfläche durchgeklickt, nur die Abfragen selbst wurden ausgeführt.
