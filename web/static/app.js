/* FinSearch UI – vanilla JS, no build step. Talks to the JSON API on the same origin.
 * Security: API data is only ever inserted as text nodes; the single piece of markup we accept is
 * <em> inside search highlights, rebuilt as a real element (see appendHighlight). */
(() => {
  'use strict';

  const METHODS = {
    bm25: { label: 'BM25', hint: 'Stichwortsuche: findet Dokumente, die Ihre Wörter enthalten (auch gebeugt).' },
    knn: { label: 'kNN', hint: 'Bedeutungssuche: vergleicht Embeddings und findet auch umschriebene oder fremdsprachige Passagen.' },
    hybrid: { label: 'Hybrid', hint: 'BM25 und kNN, fusioniert per Reciprocal Rank Fusion – meist die beste Wahl.' },
    hybrid_rerank: { label: 'Hybrid + Rerank', hint: 'Wie Hybrid, danach sortiert ein Cross-Encoder die Top-Treffer neu. Genauer, aber langsamer; der erste Aufruf lädt das Modell.' },
  };
  const COMPARE_METHODS = ['bm25', 'knn', 'hybrid'];
  const COLUMN_TEXT = {
    bm25: 'Stichwörter (Lucene BM25)',
    knn: 'Bedeutung (e5-Embeddings)',
    hybrid: 'BM25 + kNN per Rank-Fusion',
  };
  const FILTER_FIELDS = ['source', 'lang', 'doc_type', 'from', 'to'];
  const SLOW_HINT_MS = 2500;
  const COPY_FEEDBACK_MS = 1500;
  const FLASH_MS = 1600;
  const MAX_HIGHLIGHTS = 3;
  const EXCERPT_CHARS = 280;

  const $ = (sel, root = document) => root.querySelector(sel);
  const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /** Tiny DOM builder; strings become text nodes, never HTML. */
  function h(tag, props = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    node.append(...kids.flat().filter((k) => k != null));
    return node;
  }

  // ---- API ---------------------------------------------------------------------------------

  class ApiFailure extends Error {
    constructor(message, hint) {
      super(message);
      this.hint = hint || null;
    }
  }

  async function api(path, { params, body } = {}) {
    const url = params ? `${path}?${new URLSearchParams(params)}` : path;
    const init = body
      ? { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(body) }
      : { headers: { Accept: 'application/json' } };
    let res;
    try {
      res = await fetch(url, init);
    } catch {
      throw new ApiFailure('Der Server ist nicht erreichbar.', 'Läuft die Anwendung noch?');
    }
    let data = null;
    try {
      data = await res.json();
    } catch { /* non-JSON error body */ }
    if (!res.ok) throw new ApiFailure(data?.message || `Fehler ${res.status}`, data?.hint);
    return data;
  }

  // ---- formatting / safe rendering ---------------------------------------------------------

  const fmtDate = (iso) =>
    iso ? new Date(`${iso}T00:00:00`).toLocaleDateString('de-CH', { day: '2-digit', month: '2-digit', year: 'numeric' }) : 'ohne Datum';
  const fmtMs = (ms) => `${Math.round(ms)} ms`;
  const fmtScore = (s) => (Math.abs(s) >= 1 ? s.toFixed(2) : s.toPrecision(3));
  const safeUrl = (u) => (/^https?:\/\//i.test(u) ? u : null);
  const decoder = document.createElement('textarea'); // RCDATA: decodes &amp; etc., never builds elements
  const decode = (s) => {
    decoder.innerHTML = s;
    return decoder.value;
  };

  /** `raw` is HTML-escaped text whose only markup is <em>…</em>. */
  function appendHighlight(parent, raw) {
    let target = parent;
    for (const part of raw.split(/(<\/?em>)/)) {
      if (part === '<em>') {
        const em = document.createElement('em');
        parent.append(em);
        target = em;
      } else if (part === '</em>') target = parent;
      else target.append(decode(part));
    }
  }

  function snippetNode(hit) {
    const node = h('p', { class: 'snippet' });
    const parts = hit.highlights.slice(0, MAX_HIGHLIGHTS);
    if (!parts.length) node.textContent = hit.snippet;
    parts.forEach((raw, i) => {
      if (i) node.append(' … ');
      appendHighlight(node, raw);
    });
    return node;
  }

  function titleNode(title, url) {
    const href = safeUrl(url);
    return h('h3', { class: 'hit-title' }, href ? h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, title) : title);
  }

  function metaNode(item, extra = []) {
    return h(
      'p',
      { class: 'meta' },
      h('span', { class: 'badge', text: item.source }),
      h('span', { text: fmtDate(item.date) }),
      h('span', { text: `Seite ${item.page}` }),
      ...extra,
    );
  }

  async function copyId(id, btn) {
    try {
      await navigator.clipboard.writeText(id);
      btn.textContent = 'Kopiert';
    } catch {
      btn.textContent = 'Nicht möglich';
    }
    setTimeout(() => { btn.textContent = 'Kopieren'; }, COPY_FEEDBACK_MS);
  }

  function fullText(text) {
    return h('details', { class: 'fulltext' }, h('summary', { text: 'Volltext' }), h('p', { text }));
  }

  function hitCard(hit, also = null) {
    const copy = h('button', { type: 'button', class: 'btn-link', 'aria-live': 'polite', 'aria-label': `Chunk-ID kopieren: ${hit.chunk_id}` }, 'Kopieren');
    copy.addEventListener('click', () => copyId(hit.chunk_id, copy));
    const extra = [h('span', { text: hit.lang.toUpperCase() }), h('span', { text: hit.doc_type })];
    return h(
      'li',
      { class: 'card hit', 'data-chunk': hit.chunk_id },
      h('span', { class: 'rank' }, h('span', { class: 'visually-hidden', text: 'Rang ' }), String(hit.rank)),
      h(
        'div',
        { class: 'card-body' },
        titleNode(hit.title, hit.url),
        metaNode(hit, extra),
        snippetNode(hit),
        also,
        h(
          'div',
          { class: 'hit-foot' },
          h('span', { title: 'Scores sind nur innerhalb einer Methode vergleichbar', text: `Score ${fmtScore(hit.score)}` }),
          h('code', { class: 'chunk-id', text: hit.chunk_id }),
          copy,
          fullText(hit.text),
        ),
      ),
    );
  }

  // ---- shared page plumbing ----------------------------------------------------------------

  const form = $('form');
  const statusBox = $('#status');
  const errorBox = $('#error');

  function setStatus(text, busy = false) {
    statusBox.replaceChildren(...(busy ? [h('span', { class: 'spinner', 'aria-hidden': 'true' })] : []), text);
  }

  function showError(err) {
    const failure = err instanceof ApiFailure ? err : new ApiFailure('Unerwarteter Fehler im Browser.');
    if (!(err instanceof ApiFailure)) console.warn(err);
    errorBox.replaceChildren(h('p', { text: failure.message }), ...(failure.hint ? [h('p', { class: 'hint', text: failure.hint })] : []));
    errorBox.hidden = false;
    setStatus('');
  }

  function readForm() {
    const data = new FormData(form);
    const state = { q: String(data.get('q') || '').trim(), method: String(data.get('method') || 'hybrid'), k: String(data.get('k') || '') };
    for (const name of FILTER_FIELDS) {
      const value = String(data.get(name) || '');
      if (value) state[name] = value;
    }
    return state;
  }

  function writeForm(params) {
    for (const [name, value] of params) {
      const field = form.elements[name];
      if (!field) continue;
      field.value = value; // also selects the matching radio button
    }
    updateMethodHint();
    const filters = $('details.filters');
    if (filters && FILTER_FIELDS.some((name) => params.get(name))) filters.open = true;
  }

  function updateMethodHint() {
    const hint = $('#method-hint');
    if (hint) hint.textContent = METHODS[readForm().method]?.hint ?? '';
  }

  /** Runs `task` with busy/error handling; the slow-model note appears only if it takes a while. */
  async function withLoading(message, task, slowNote = '') {
    const button = $('button[type="submit"]', form);
    errorBox.hidden = true;
    button.setAttribute('aria-busy', 'true');
    button.disabled = true;
    setStatus(message, true);
    const timer = slowNote ? setTimeout(() => setStatus(`${message} ${slowNote}`, true), SLOW_HINT_MS) : null;
    try {
      await task();
    } catch (err) {
      showError(err);
    } finally {
      clearTimeout(timer);
      button.removeAttribute('aria-busy');
      button.disabled = false;
    }
  }

  const SLOW_NOTE = 'Beim ersten Mal wird das Sprachmodell geladen, das dauert einige Sekunden.';

  // ---- page: search ------------------------------------------------------------------------

  async function runSearch(state) {
    const results = $('#results');
    results.replaceChildren();
    const slow = state.method === 'bm25' ? '' : SLOW_NOTE;
    await withLoading('Suche läuft …', async () => {
      const params = { ...state, k: state.k || '10' };
      const data = await api('/search', { params });
      results.replaceChildren(...data.hits.map((hit) => hitCard(hit)));
      if (!data.hits.length) {
        results.append(h('li', { class: 'empty', text: 'Keine Treffer. Versuchen Sie andere Begriffe oder lockern Sie die Filter.' }));
      }
      setStatus(`${data.n_hits} Treffer · ${METHODS[data.method].label} · ${fmtMs(data.total_ms)}`);
    }, slow);
  }

  // ---- page: ask ---------------------------------------------------------------------------

  function answerNodes(data) {
    const known = new Set(data.citations.map((c) => c.n));
    return data.answer.split(/(\[\d+\])/).map((part) => {
      const m = /^\[(\d+)\]$/.exec(part);
      if (!m || !known.has(Number(m[1]))) return part;
      return h('a', { class: 'cite', href: `#src-${m[1]}`, 'aria-label': `Quelle ${m[1]} anzeigen` }, part);
    });
  }

  function sourceCard(chunk, label, id) {
    const href = safeUrl(chunk.url);
    const excerpt = chunk.text.length > EXCERPT_CHARS ? `${chunk.text.slice(0, EXCERPT_CHARS).replace(/\s+\S*$/, '')} …` : chunk.text;
    return h(
      'li',
      { class: 'card source', id, tabindex: '-1' },
      h('span', { class: 'rank', text: label }),
      h(
        'div',
        { class: 'card-body' },
        titleNode(chunk.title, chunk.url),
        metaNode(chunk),
        h('p', { class: 'snippet', text: excerpt }),
        h('div', { class: 'hit-foot' }, href ? h('span', { class: 'chunk-id', text: href }) : null, fullText(chunk.text)),
      ),
    );
  }

  function renderAnswer(data) {
    const panel = $('#answer');
    panel.className = `answer-panel${data.found ? '' : ' not-found'}`;
    const meta = h(
      'p',
      { class: 'answer-meta' },
      h('span', {}, 'Anbieter: ', h('strong', { text: data.provider })),
      h('span', { text: data.confidence_note }),
      data.timings_ms.llm ? h('span', { text: `LLM ${fmtMs(data.timings_ms.llm)}` }) : null,
      data.dropped_citations.length ? h('span', { text: `Ungültige Quellenangaben entfernt: ${data.dropped_citations.join(', ')}` }) : null,
    );
    const notFound = data.found ? [] : [h('p', { text: 'Die gefundenen Passagen belegen diese Frage nicht. Formulieren Sie sie um oder lockern Sie die Filter.' })];
    panel.replaceChildren(h('p', { class: 'answer-text', 'aria-live': 'polite' }, ...answerNodes(data)), ...notFound, meta);
    panel.hidden = false;

    const wrap = $('#sources-wrap');
    const list = $('#sources');
    const byId = new Map(data.used_chunks.map((c) => [c.chunk_id, c]));
    if (data.found) {
      list.replaceChildren(...data.citations.map((c) => sourceCard(byId.get(c.chunk_id) ?? { ...c, source: '', date: null, text: '' }, `[${c.n}]`, `src-${c.n}`)));
      $('h3', wrap).textContent = 'Quellen';
    } else {
      list.replaceChildren(...data.used_chunks.map((c, i) => sourceCard(c, String(i + 1), `chk-${i + 1}`)));
      $('h3', wrap).textContent = data.used_chunks.length ? 'Geprüfte Passagen (nicht belegend)' : '';
    }
    wrap.hidden = !list.children.length;
  }

  async function runAsk(state) {
    $('#answer').hidden = true;
    $('#sources-wrap').hidden = true;
    const filters = {};
    for (const name of FILTER_FIELDS) if (state[name]) filters[name] = state[name];
    await withLoading('Quellen werden gelesen …', async () => {
      const data = await api('/ask', { body: { question: state.q, method: state.method, k: Number(state.k || 6), filters } });
      renderAnswer(data);
      setStatus(data.found ? 'Antwort mit Quellen' : 'In den Quellen nicht gefunden');
    }, SLOW_NOTE);
  }

  document.addEventListener('click', (event) => {
    const link = event.target instanceof Element ? event.target.closest('a.cite') : null;
    if (!link) return;
    event.preventDefault();
    const card = document.getElementById(link.getAttribute('href').slice(1));
    if (!card) return;
    card.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'center' });
    card.focus({ preventScroll: true });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), FLASH_MS);
  });

  // ---- page: compare -----------------------------------------------------------------------

  function renderCompare(data) {
    const where = {}; // chunk_id -> {method: rank}
    for (const m of COMPARE_METHODS) {
      for (const hit of data.results[m].hits) (where[hit.chunk_id] ??= {})[m] = hit.rank;
    }
    const maxMs = Math.max(...COMPARE_METHODS.map((m) => data.results[m].total_ms), 1);

    const columns = COMPARE_METHODS.map((m) => {
      const res = data.results[m];
      const bar = h('span');
      bar.style.setProperty('--w', `${Math.max(4, (res.total_ms / maxMs) * 100)}%`);
      const items = res.hits.map((hit) => {
        const present = Object.keys(where[hit.chunk_id]);
        const others = present.filter((o) => o !== m);
        const also = others.length
          ? h(
            'p',
            { class: 'also' },
            h('span', { class: 'badge multi', text: `in ${present.length} Listen` }),
            ...others.map((o) => h('span', { class: 'badge', 'data-m': o, text: `${METHODS[o].label} #${where[hit.chunk_id][o]}` })),
          )
          : null;
        return hitCard(hit, also);
      });
      return h(
        'section',
        { class: 'col', 'data-method': m, 'aria-labelledby': `col-${m}` },
        h('h3', { id: `col-${m}`, text: METHODS[m].label }),
        h('p', { class: 'col-desc', text: COLUMN_TEXT[m] }),
        h('div', { class: 'latency' }, h('span', { text: `${res.n_hits} Treffer` }), h('strong', { text: fmtMs(res.total_ms) }), h('div', { class: 'bar', 'aria-hidden': 'true' }, bar)),
        items.length ? h('ol', { class: 'results' }, ...items) : h('p', { class: 'empty', text: 'Keine Treffer.' }),
      );
    });
    $('#columns').replaceChildren(...columns);
    $('#columns').hidden = false;

    const hybrid = data.results.hybrid.hits.map((x) => x.chunk_id);
    const missing = (m) => hybrid.filter((id) => !(m in where[id])).length;
    const inAll = Object.values(where).filter((w) => Object.keys(w).length === COMPARE_METHODS.length).length;
    const overlap = $('#overlap');
    overlap.textContent = hybrid.length
      ? `Von ${hybrid.length} Hybrid-Treffern stehen ${missing('bm25')} nicht in der BM25-Liste und ${missing('knn')} nicht in der kNN-Liste. ${inAll} Treffer kommen in allen drei Listen vor.`
      : 'Keine Treffer in den Listen.';
    overlap.hidden = false;
  }

  function linkMatchingCards(event, on) {
    const card = event.target instanceof Element ? event.target.closest('.card[data-chunk]') : null;
    if (!card) return;
    const id = card.dataset.chunk;
    for (const other of $('#columns').querySelectorAll('.card[data-chunk]')) {
      if (other.dataset.chunk === id) other.classList.toggle('linked', on);
    }
  }

  async function runCompare(state) {
    $('#columns').hidden = true;
    $('#overlap').hidden = true;
    await withLoading('Drei Verfahren werden verglichen …', async () => {
      const data = await api('/compare', { params: { ...state, k: state.k || '10' } });
      renderCompare(data);
      setStatus(`Vergleich für «${data.query}»`);
    }, SLOW_NOTE);
  }

  // ---- banner / corpus info ----------------------------------------------------------------

  async function loadSystemInfo() {
    const banner = $('#system-banner');
    try {
      const health = await api('/health');
      const es = health.elasticsearch;
      $('#corpus-info').textContent = `${es.chunks} Abschnitte indexiert · Antworten: ${health.llm.provider}${health.llm.fallback ? ' (Fallback)' : ''}`;
      if (health.status !== 'ok') {
        banner.textContent = health.hint || 'Eingeschränkter Betrieb.';
        banner.hidden = false;
      }
    } catch (err) {
      banner.textContent = err.message;
      banner.hidden = false;
    }
  }

  async function loadSources() {
    const select = $('#f-source');
    if (!select) return;
    try {
      const stats = await api('/stats');
      for (const name of Object.keys(stats.by_source).sort()) {
        select.append(h('option', { value: name, text: name.toUpperCase() }));
      }
    } catch { /* the banner already explains an unavailable backend */ }
  }

  // ---- init --------------------------------------------------------------------------------

  const RUNNERS = { search: runSearch, ask: runAsk, compare: runCompare };

  async function init() {
    loadSystemInfo();
    const run = RUNNERS[document.body.dataset.page];
    if (!form || !run) return;

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const state = readForm();
      if (!state.q) {
        showError(new ApiFailure('Bitte eine Suchanfrage eingeben.'));
        $('#q').focus();
        return;
      }
      history.pushState(null, '', `?${new URLSearchParams(state)}`);
      run(state);
    });
    form.addEventListener('change', (event) => {
      if (event.target.name === 'method') updateMethodHint();
    });
    for (const chip of document.querySelectorAll('[data-example]')) {
      chip.addEventListener('click', () => {
        $('#q').value = chip.dataset.example;
        form.requestSubmit();
      });
    }
    const columns = $('#columns');
    if (columns) {
      for (const [type, on] of [['mouseover', true], ['focusin', true], ['mouseout', false], ['focusout', false]]) {
        columns.addEventListener(type, (event) => linkMatchingCards(event, on));
      }
    }
    const restore = () => {
      const params = new URLSearchParams(location.search);
      writeForm(params);
      if (params.get('q')) run(readForm());
    };
    window.addEventListener('popstate', restore);
    await loadSources();
    restore();
    updateMethodHint();
  }

  init();
})();
