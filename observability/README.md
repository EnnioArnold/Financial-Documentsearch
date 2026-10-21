# Observability

Jede API-Anfrage (außer `/health` und `/static`) schreibt **ein** Dokument in den Index `finsearch-queries` (Middleware `api/middleware.py`, Schreiben im Hintergrund; ist ES nicht erreichbar, wird das Dokument verworfen und gewarnt). Zusätzlich gibt es pro Anfrage eine JSON-Logzeile (`finsearch.request`).

Datenschutz: Standardmäßig wird nur ein Hash (`query_hash`, sha256 der normalisierten Anfrage, 16 Zeichen) und die Länge gespeichert. Den Klartext (`query`) gibt es nur mit `LOG_RAW_QUERIES=true`. Keine Header, keine Bodies, keine Keys.

## Dashboard "FinSearch – Betrieb"

```bash
make kibana-setup          # = python observability/setup_kibana.py (idempotent, nur stdlib)
```

Legt die Data Views `finsearch-queries*` (Zeitfeld `@timestamp`) und `finsearch-v*` an und importiert `kibana_dashboard.ndjson`. Beide Data Views werden mit `allowNoIndex` angelegt, das Dashboard lässt sich also importieren, bevor der Inhaltsindex existiert; die Indexstatistik-Kacheln zeigen dann "N/A" bis zum ersten Ingest. Umgebung: `KIBANA_URL` (Standard `http://127.0.0.1:5601`), `KIBANA_WAIT_S` (Standard 300).

Dashboard ändern: in Kibana bearbeiten, dann neu exportieren:

```bash
curl -s -XPOST $KIBANA_URL/api/saved_objects/_export -H 'kbn-xsrf: true' -H 'content-type: application/json' \
  -d '{"type":["dashboard"],"includeReferencesDeep":true,"excludeExportDetails":true}' > observability/kibana_dashboard.ndjson
```

## Demo-Daten

```bash
python observability/seed_demo_queries.py            # ~60 synthetische Anfragen der letzten 6 h (synthetic: true)
python observability/seed_demo_queries.py --delete   # entfernt alle synthetischen Dokumente
```

Alerts: siehe `docs/ALERTS.md`. Screenshot: `docs/screenshots/kibana_dashboard.png`.
