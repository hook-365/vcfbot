// vcfbot — local research console
// Streaming RAG chat client. Plain HTML/CSS/JS, ES modules, no build step.

import { marked } from 'https://cdn.jsdelivr.net/npm/marked@13.0.3/+esm';
import DOMPurify from 'https://cdn.jsdelivr.net/npm/dompurify@3.1.6/+esm';

// ────────────────────────────────────────────────────────────────
// State
// ────────────────────────────────────────────────────────────────
const state = {
  history: [],          // [{role, content, sources?}, ...]
  isStreaming: false,
  controller: null,
  turn: 0,
  status: null,         // last /api/status payload, for export header
};

const $ = (sel, root = document) => root.querySelector(sel);

const els = {
  header:      $('.app-header'),
  main:        $('#app-main'),
  thread:      $('#thread'),
  emptyState:  $('#empty-state'),
  composer:    $('#composer'),
  input:       $('#composer-input'),
  send:        $('#composer-send'),
  counter:     $('#composer-counter'),
  themeToggle: $('#theme-toggle'),
  exportBtn:   $('#export-btn'),
  aboutBtn:    $('#about-btn'),
  aboutDialog: $('#about-dialog'),
  aboutClose:  $('#about-close'),
  clogBtn:     $('#changelog-btn'),
  clogDialog:  $('#changelog-dialog'),
  clogClose:   $('#changelog-close'),
  clogList:    $('#changelog-list'),
  clogMeta:    $('#changelog-meta'),
  statusRail:  $('#status-rail'),
  statusChat:  $('#status-chat'),
  statusEmbed: $('#status-embed'),
  statusChunks:$('#status-chunks'),
  statusTopk:  $('#status-topk'),
  statusMq:    $('#status-mq'),
  statusRerank:$('#status-rerank'),
  statusMqGroup:     $('#status-mq-group'),
  statusRerankGroup: $('#status-rerank-group'),
  statusLink:  $('#status-link-text'),
  tmplUser:    $('#tmpl-user'),
  tmplAsst:    $('#tmpl-assistant'),
  tmplSource:  $('#tmpl-source'),
};

marked.setOptions({ gfm: true, breaks: false, pedantic: false });

// ────────────────────────────────────────────────────────────────
// Theme
// ────────────────────────────────────────────────────────────────
els.themeToggle.addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('vcfbot-theme', next); } catch (_) {}
});

// ────────────────────────────────────────────────────────────────
// Composer
// ────────────────────────────────────────────────────────────────
function autoSize() {
  const t = els.input;
  t.style.height = 'auto';
  const cap = Math.floor(window.innerHeight * 0.32);
  t.style.height = Math.min(t.scrollHeight, cap) + 'px';
  const len = t.value.length;
  els.counter.textContent = len;
  els.send.disabled = state.isStreaming || len === 0;
}

els.input.addEventListener('input', autoSize);
els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    submit();
  }
});

document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    els.input.focus();
    els.input.select();
  }
});

els.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  submit();
});

document.querySelectorAll('.starter').forEach((btn) => {
  btn.addEventListener('click', () => {
    els.input.value = btn.dataset.question || '';
    autoSize();
    submit();
  });
});

function submit() {
  const q = els.input.value.trim();
  if (!q || state.isStreaming) return;
  els.input.value = '';
  autoSize();
  ask(q);
}

// ────────────────────────────────────────────────────────────────
// Status
// ────────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    state.status = d;
    els.statusChat.textContent   = trimModel(d.chat_model);
    els.statusEmbed.textContent  = trimModel(d.embed_model);
    els.statusChunks.textContent = formatNum(d.collection_size);
    // Retrieval knobs
    setStatusValue(els.statusTopk, d.top_k ?? '—', d.top_k != null);
    setStatusValue(els.statusMq, d.multi_query ? 'on' : 'off', !!d.multi_query);
    setStatusValue(els.statusRerank,
      d.rerank_enabled ? trimModel(d.rerank_model) : 'off', !!d.rerank_enabled);
    // Richer detail in tooltips (values stay compact)
    if (els.statusMqGroup) {
      els.statusMqGroup.title = d.multi_query
        ? `multi-query retrieval · up to ${d.multi_query_max} sub-queries`
        : 'multi-query retrieval (off)';
    }
    if (els.statusRerankGroup) {
      els.statusRerankGroup.title = d.rerank_enabled
        ? `cross-encoder rerank · pool ${formatNum(d.rerank_top_n)} → top-${d.top_k}`
        : 'cross-encoder reranking (off)';
    }
    els.statusLink.textContent   = 'online';
    els.statusRail.dataset.state = 'online';
  } catch (e) {
    els.statusLink.textContent   = 'offline';
    els.statusRail.dataset.state = 'error';
    console.warn('status failed', e);
  }
}

function trimModel(name) {
  if (!name) return '—';
  // strip "publisher/" prefix and the cosmetic `.gguf` suffix
  const stripped = name.includes('/') ? name.slice(name.indexOf('/') + 1) : name;
  return stripped.replace(/\.gguf$/i, '');
}
function formatNum(n) {
  return typeof n === 'number' ? n.toLocaleString('en-US') : '—';
}
// Set a status value and dim it when the feature is off/disabled.
function setStatusValue(el, text, active) {
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('is-off', !active);
}

// ────────────────────────────────────────────────────────────────
// Render: messages
// ────────────────────────────────────────────────────────────────
function makeUserMessage(text, num) {
  const n = els.tmplUser.content.firstElementChild.cloneNode(true);
  n.querySelector('.msg__num').textContent = String(num).padStart(2, '0');
  n.querySelector('.msg__content').textContent = text;
  return n;
}

function makeAssistantMessage(num) {
  const n = els.tmplAsst.content.firstElementChild.cloneNode(true);
  n.querySelector('.msg__num').textContent = String(num).padStart(2, '0');
  return n;
}

// ────────────────────────────────────────────────────────────────
// Render: source cards
// ────────────────────────────────────────────────────────────────
function pageKey(start, end) {
  return start === end ? String(start) : `${start}-${end}`;
}

function makeSourceCard(hit, idx) {
  const n = els.tmplSource.content.firstElementChild.cloneNode(true);
  const pages = hit.page_start === hit.page_end
    ? `p.${hit.page_start}`
    : `p.${hit.page_start}–${hit.page_end}`;
  n.dataset.source    = hit.source;
  n.dataset.pages     = pageKey(hit.page_start, hit.page_end);
  n.dataset.pageStart = String(hit.page_start);
  n.dataset.pageEnd   = String(hit.page_end);
  n.dataset.idx       = String(idx);
  n.querySelector('.src__doc').textContent   = hit.source;
  n.querySelector('.src__pages').textContent = pages;
  const sec = n.querySelector('.src__section');
  if (hit.section && hit.section.trim()) {
    sec.textContent = hit.section.trim();
    sec.hidden = false;
  }
  n.querySelector('.src__distance-val').textContent = (hit.distance ?? 0).toFixed(3);
  n.querySelector('.src__text').textContent = hit.text;
  n.querySelector('.src__expand').addEventListener('click', () => {
    n.classList.toggle('is-open');
  });

  const pdfLink = n.querySelector('.src__action--pdf');
  if (hit.pdf_url) {
    pdfLink.href = hit.pdf_url;
    pdfLink.querySelector('.src__action-detail').textContent = pages;
    pdfLink.title = `Open ${hit.source}.pdf at ${pages}`;
    pdfLink.hidden = false;
    pdfLink.dataset.pdfUrl = hit.pdf_url;
  }

  const webLink = n.querySelector('.src__action--web');
  if (hit.web_url) {
    webLink.href = hit.web_url;
    webLink.title = `Open ${hit.source} doc set on broadcom.com`;
    webLink.hidden = false;
  }

  return n;
}

function renderSources(panel, toggleBtn, hits) {
  panel.innerHTML = '';
  hits.forEach((hit, idx) => panel.appendChild(makeSourceCard(hit, idx)));
  toggleBtn.querySelector('.sources-toggle__count').textContent = hits.length;
  toggleBtn.addEventListener('click', () => {
    const expanded = toggleBtn.getAttribute('aria-expanded') === 'true';
    toggleBtn.setAttribute('aria-expanded', expanded ? 'false' : 'true');
    panel.hidden = expanded;
  });
}

// ────────────────────────────────────────────────────────────────
// Markdown + citation decoration
// ────────────────────────────────────────────────────────────────
function renderMarkdown(text) {
  const html = marked.parse(text || '');
  return DOMPurify.sanitize(html, {
    ADD_ATTR: ['class', 'data-source', 'data-pages'],
  });
}

// Match citation markers in answer text: [docname p.42] or [docname p.42-43] or [docname p.42–43]
const CITE_RE = /\[([\w][\w.-]*)\s+p\.(\d+)(?:[–-](\d+))?\]/g;

function decorateCitations(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      let p = node.parentElement;
      while (p && p !== root) {
        const t = p.tagName;
        if (t === 'CODE' || t === 'PRE' || p.classList?.contains('cite')) {
          return NodeFilter.FILTER_REJECT;
        }
        p = p.parentElement;
      }
      CITE_RE.lastIndex = 0;
      return CITE_RE.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });

  const targets = [];
  let n;
  while ((n = walker.nextNode())) targets.push(n);

  for (const node of targets) {
    const text = node.nodeValue;
    const frag = document.createDocumentFragment();
    let lastIdx = 0;
    CITE_RE.lastIndex = 0;
    let m;
    while ((m = CITE_RE.exec(text)) !== null) {
      if (m.index > lastIdx) {
        frag.appendChild(document.createTextNode(text.slice(lastIdx, m.index)));
      }
      const doc = m[1];
      const start = m[2];
      const end = m[3] || start;
      const a = document.createElement('a');
      a.className = 'cite';
      a.href = '#';
      a.dataset.source = doc;
      a.dataset.pages = pageKey(start, end);
      a.title = `${doc} p.${start}${end !== start ? '–' + end : ''}`;
      const pagesSpan = document.createElement('span');
      pagesSpan.className = 'cite__pages';
      pagesSpan.textContent = end === start ? `p.${start}` : `p.${start}–${end}`;
      a.appendChild(pagesSpan);
      a.addEventListener('click', (e) => {
        e.preventDefault();
        // ⌘/⌃-click jumps straight to the PDF at the cited page in a new tab.
        if (e.metaKey || e.ctrlKey) {
          const msg = a.closest('.msg');
          const wantPages = a.dataset.pages;
          let card = null;
          if (msg && wantPages) {
            for (const c of msg.querySelectorAll('.src')) {
              if (c.dataset.pages === wantPages) { card = c; break; }
            }
            if (!card) {
              const wantStart = String(wantPages).split('-')[0];
              for (const c of msg.querySelectorAll('.src')) {
                if (c.dataset.pageStart === wantStart) { card = c; break; }
              }
            }
          }
          const pdfHref = card?.querySelector('.src__action--pdf')?.href;
          if (pdfHref) {
            window.open(pdfHref, '_blank', 'noopener');
            return;
          }
        }
        spotlightSource(a);
      });
      frag.appendChild(a);
      lastIdx = m.index + m[0].length;
    }
    if (lastIdx < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIdx)));
    }
    node.parentNode.replaceChild(frag, node);
  }
}

function spotlightSource(citeEl) {
  const msg = citeEl.closest('.msg');
  if (!msg) return;
  const wrap = msg.querySelector('.msg__sources-wrap');
  const toggle = msg.querySelector('.sources-toggle');
  const panel = msg.querySelector('.sources');

  // ensure sources are visible
  if (wrap?.hidden) wrap.hidden = false;
  if (panel?.hidden) {
    panel.hidden = false;
    toggle?.setAttribute('aria-expanded', 'true');
  }

  const want = citeEl.dataset.pages;
  const wantStart = String(want).split('-')[0];
  const cards = msg.querySelectorAll('.src');
  let match = null;
  for (const c of cards) {
    if (c.dataset.pages === want) { match = c; break; }
  }
  if (!match) {
    for (const c of cards) {
      if (c.dataset.pageStart === wantStart) { match = c; break; }
    }
  }
  if (!match) return;

  match.classList.add('is-open');
  match.classList.remove('is-highlighted');
  // restart animation
  void match.offsetWidth;
  match.classList.add('is-highlighted');
  match.scrollIntoView({ behavior: 'smooth', block: 'center' });
  window.setTimeout(() => match.classList.remove('is-highlighted'), 1600);
}

// ────────────────────────────────────────────────────────────────
// Ask flow
// ────────────────────────────────────────────────────────────────
async function ask(question) {
  // hide empty state on first ask
  if (!els.emptyState.hidden) {
    els.emptyState.hidden = true;
    els.thread.hidden = false;
  }

  state.turn += 1;
  const userNode = makeUserMessage(question, state.turn);
  els.thread.appendChild(userNode);

  state.turn += 1;
  const asstNode = makeAssistantMessage(state.turn);
  els.thread.appendChild(asstNode);

  const retrieving = asstNode.querySelector('.msg__retrieving');
  const contentEl  = asstNode.querySelector('.msg__content');
  const wrap       = asstNode.querySelector('.msg__sources-wrap');
  const toggleBtn  = asstNode.querySelector('.sources-toggle');
  const panel      = asstNode.querySelector('.sources');

  retrieving.hidden = false;
  state.isStreaming = true;
  els.statusRail.dataset.state = 'busy';
  els.send.disabled = true;

  // scroll user message into view, leaving room for assistant + dock
  requestAnimationFrame(() => {
    userNode.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  let answer = '';
  let lastHits = [];

  try {
    state.controller = new AbortController();
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        question,
        // Strip `sources` (UI-only) before sending to the API.
        history: state.history.map(({ role, content }) => ({ role, content })),
      }),
      signal: state.controller.signal,
    });

    if (!res.ok || !res.body) {
      const txt = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${txt.slice(0, 200)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    streamLoop: while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const parsed = parseSseEvent(block);
        if (!parsed) continue;
        const { event, data } = parsed;

        if (event === 'sources') {
          lastHits = (data && data.hits) || [];
          if (lastHits.length) {
            renderSources(panel, toggleBtn, lastHits);
            wrap.hidden = false;
          }
        } else if (event === 'token') {
          if (!retrieving.hidden) retrieving.hidden = true;
          answer += (data && data.text) || '';
          renderStreaming(contentEl, answer);
        } else if (event === 'done') {
          if (data && typeof data.answer === 'string') answer = data.answer;
          renderFinal(contentEl, answer);
          break streamLoop;
        } else if (event === 'error') {
          throw new Error((data && data.message) || 'stream error');
        }
      }
    }

    renderFinal(contentEl, answer);
    state.history.push({ role: 'user',      content: question });
    state.history.push({ role: 'assistant', content: answer, sources: lastHits });
    enableExport();
  } catch (err) {
    retrieving.hidden = true;
    const msg = (err && err.message) || String(err);
    contentEl.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'error';
    p.textContent = `error · ${msg}`;
    p.style.color = 'var(--warn)';
    p.style.fontFamily = 'var(--font-mono)';
    p.style.fontSize = 'var(--step--1)';
    contentEl.appendChild(p);
    els.statusRail.dataset.state = 'error';
  } finally {
    retrieving.hidden = true;
    state.isStreaming = false;
    state.controller = null;
    autoSize();
    if (els.statusRail.dataset.state === 'busy') {
      els.statusRail.dataset.state = 'online';
    }
    els.input.focus();
  }
}

function renderStreaming(contentEl, text) {
  contentEl.innerHTML = '';
  const pre = document.createElement('div');
  pre.className = 'stream-text';
  pre.style.whiteSpace = 'pre-wrap';
  pre.style.wordBreak = 'break-word';
  pre.textContent = text;
  const cursor = document.createElement('span');
  cursor.className = 'stream-cursor';
  pre.appendChild(cursor);
  contentEl.appendChild(pre);
}

function renderFinal(contentEl, text) {
  contentEl.innerHTML = renderMarkdown(text);
  decorateCitations(contentEl);
}

// ────────────────────────────────────────────────────────────────
// SSE parser
// ────────────────────────────────────────────────────────────────
function parseSseEvent(block) {
  if (!block) return null;
  const lines = block.split('\n');
  let event = 'message';
  const dataLines = [];
  for (const line of lines) {
    if (!line || line.startsWith(':')) continue;
    const c = line.indexOf(':');
    const field = c >= 0 ? line.slice(0, c) : line;
    let value = c >= 0 ? line.slice(c + 1) : '';
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') dataLines.push(value);
  }
  const dataStr = dataLines.join('\n');
  let data = {};
  if (dataStr) {
    try { data = JSON.parse(dataStr); }
    catch { data = { raw: dataStr }; }
  }
  return { event, data };
}

// ────────────────────────────────────────────────────────────────
// About dialog
// ────────────────────────────────────────────────────────────────
function openDialog(d) {
  if (typeof d.showModal === 'function') d.showModal();
  else d.setAttribute('open', '');
}

els.aboutBtn.addEventListener('click', () => openDialog(els.aboutDialog));
els.aboutClose.addEventListener('click', () => els.aboutDialog.close());
els.aboutDialog.addEventListener('click', (e) => {
  if (e.target === els.aboutDialog) els.aboutDialog.close();
});

els.clogBtn.addEventListener('click', () => {
  openDialog(els.clogDialog);
  loadChangelog();
});
els.clogClose.addEventListener('click', () => els.clogDialog.close());
els.clogDialog.addEventListener('click', (e) => {
  if (e.target === els.clogDialog) els.clogDialog.close();
});

async function loadChangelog() {
  if (!els.clogList) return;
  els.clogMeta.textContent = 'checking…';
  try {
    const r = await fetch('/api/changelog?limit=20');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const { entries } = await r.json();
    if (!entries || !entries.length) {
      els.clogList.innerHTML = '<li class="about__updates-empty">No corpus updates recorded yet. Upstream checks happen daily; you\'ll see entries here when Broadcom republishes.</li>';
      els.clogMeta.textContent = '0 updates';
      return;
    }
    els.clogList.innerHTML = entries.map(e => {
      const when = relativeTime(e.ts);
      const dur = e.duration_sec ? `${Math.round(e.duration_sec)}s` : '';
      let detail;
      // Prefer the incremental-update breakdown when present (added 2026-05-27).
      // Older entries before that field landed fall back to plain net delta.
      if (e.chunks_added != null && e.chunks_removed != null) {
        const a = e.chunks_added;
        const r = e.chunks_removed;
        if (a === 0 && r === 0) {
          detail = `no diff, ${dur}`;
        } else {
          const parts = [];
          if (a > 0) parts.push(`+${a.toLocaleString()} new`);
          if (r > 0) parts.push(`-${r.toLocaleString()} removed`);
          detail = `${parts.join(', ')}, ${dur}`;
        }
      } else {
        const delta = e.chunks_after - e.chunks_before;
        const sign = delta === 0 ? '±0' : (delta > 0 ? `+${delta}` : `${delta}`);
        detail = `${sign}, ${dur}`;
      }
      const sectionsHtml = renderDiffSections(e.diff_sections);
      return `<li>
        <div class="upd__row">
          <span class="upd__when" title="${escapeAttr(e.ts)}">${when}</span>
          <span class="upd__src">${escapeAttr(e.source)}</span>
          <span class="upd__delta">${e.chunks_after.toLocaleString()} chunks <span style="color:var(--ink-4)">(${detail})</span></span>
        </div>
        ${sectionsHtml}
      </li>`;
    }).join('');
    els.clogMeta.textContent = `${entries.length} update${entries.length === 1 ? '' : 's'} · last check daily @ 04:00`;
  } catch (err) {
    els.clogList.innerHTML = `<li class="about__updates-empty">Couldn't load changelog: ${escapeAttr(err.message || String(err))}</li>`;
    els.clogMeta.textContent = 'error';
  }
}

function renderDiffSections(sections) {
  if (!sections || !sections.length) return '';
  const SHOW = 10;
  const visible = sections.slice(0, SHOW);
  const overflow = sections.length - visible.length;
  const rows = visible.map(s => {
    const a = s.added || 0;
    const r = s.removed || 0;
    const parts = [];
    if (a > 0) parts.push(`<span class="upd__sec-add">+${a}</span>`);
    if (r > 0) parts.push(`<span class="upd__sec-rem">−${r}</span>`);
    const counts = parts.join(' ');
    const pages = (s.pages && s.pages.length === 2)
      ? (s.pages[0] === s.pages[1] ? `p.${s.pages[0]}` : `pp.${s.pages[0]}–${s.pages[1]}`)
      : '';
    return `<li>
      <span class="upd__sec-counts">${counts}</span>
      <span class="upd__sec-name">${escapeAttr(s.section || '(unsectioned)')}</span>
      <span class="upd__sec-pages">${pages}</span>
    </li>`;
  }).join('');
  const more = overflow > 0
    ? `<li class="upd__sec-more">…and ${overflow} more section${overflow === 1 ? '' : 's'}</li>`
    : '';
  return `<details class="upd__sections">
    <summary>${sections.length} section${sections.length === 1 ? '' : 's'} changed</summary>
    <ul>${rows}${more}</ul>
  </details>`;
}

function relativeTime(iso) {
  const then = new Date(iso);
  const now = new Date();
  const sec = Math.max(1, Math.round((now - then) / 1000));
  if (sec < 60)     return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60)     return `${min}m ago`;
  const hr  = Math.round(min / 60);
  if (hr  < 24)     return `${hr}h ago`;
  const day = Math.round(hr / 24);
  if (day < 30)     return `${day}d ago`;
  return then.toISOString().slice(0, 10);
}

function escapeAttr(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
  );
}

// ────────────────────────────────────────────────────────────────
// Export
// ────────────────────────────────────────────────────────────────
function enableExport() {
  els.exportBtn.disabled = false;
  els.header.dataset.empty = 'false';
}

els.exportBtn.addEventListener('click', () => {
  if (!state.history.length) return;
  const md = buildExportMarkdown();
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  triggerDownload(md, `vcfbot-${stamp}.md`, 'text/markdown');
});

function buildExportMarkdown() {
  const now = new Date();
  const ts  = now.toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
  const out = [];

  out.push(`# vcfbot session — ${ts}`);
  if (state.status) {
    const s = state.status;
    out.push('');
    out.push(`**chat:** \`${s.chat_model}\`  **embed:** \`${s.embed_model}\`  **chunks:** ${formatNum(s.collection_size)}`);
    const rerank = s.rerank_enabled ? `${s.rerank_model} (pool ${formatNum(s.rerank_top_n)})` : 'off';
    out.push(`**top-k:** ${s.top_k ?? '—'}  **multi-query:** ${s.multi_query ? `on (≤${s.multi_query_max})` : 'off'}  **rerank:** ${rerank}`);
    out.push(`**endpoint:** \`${s.lm_studio_url}\`  **origin:** ${window.location.origin}`);
  }
  out.push('');

  let qn = 0;
  let an = 0;
  for (const turn of state.history) {
    if (turn.role === 'user') {
      qn += 1;
      out.push('---', '', `## Q${qn}`, '');
      for (const line of turn.content.split('\n')) out.push(`> ${line}`);
      out.push('');
    } else if (turn.role === 'assistant') {
      an += 1;
      out.push(`### A${an}`, '', turn.content.trim(), '');
      const sources = turn.sources || [];
      if (sources.length) {
        out.push('#### Sources', '');
        sources.forEach((s, i) => {
          const pages = s.page_start === s.page_end
            ? `p.${s.page_start}`
            : `p.${s.page_start}–${s.page_end}`;
          const dist = (s.distance ?? 0).toFixed(3);
          out.push(`${i + 1}. **${s.source} ${pages}** — d=${dist}`);
          if (s.section) out.push(`   *${s.section}*`);
          if (s.pdf_url) out.push(`   PDF: ${absoluteUrl(s.pdf_url)}`);
          if (s.web_url) out.push(`   Broadcom: ${s.web_url}`);
          // Excerpt: first ~5 lines of the chunk, truncated.
          const excerpt = (s.text || '')
            .split('\n')
            .filter(l => l.trim())
            .slice(0, 6)
            .join(' ')
            .slice(0, 480);
          if (excerpt) {
            out.push('');
            out.push(`   > ${excerpt}…`);
          }
          out.push('');
        });
      }
    }
  }
  out.push('---', '', '*Generated by vcfbot — local RAG over Broadcom / VMware Cloud Foundation documentation.*');
  return out.join('\n');
}

function absoluteUrl(href) {
  if (/^https?:/.test(href)) return href;
  return new URL(href, window.location.origin).toString();
}

function triggerDownload(content, filename, mime) {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ────────────────────────────────────────────────────────────────
// Init
// ────────────────────────────────────────────────────────────────
loadStatus();
autoSize();
els.input.focus();

// Re-check status periodically (cheap; helps user notice if LM Studio drops)
setInterval(loadStatus, 30_000);

// ────────────────────────────────────────────────────────────────
// View switcher (chat / planner) + Planner tab
// ────────────────────────────────────────────────────────────────
const planner = (() => {
  const form    = $('#planner-form');
  const results = $('#planner-results');
  const statusEl = $('#planner-status');
  let opts = null;
  let built = false;

  // Input groups: [friendly_key, label, kind]. kind drives the control.
  const NUMS = [
    ['host_cpu_cores',          'CPU cores per host'],
    ['host_ram_gb',             'RAM per host (GB)'],
  ];
  const ADV_NUMS = [
    ['cpu_oversubscription',    'CPU oversubscription (X:1)'],
    ['memory_oversubscription', 'Memory oversubscription (X:1)'],
    ['host_ops_reserve_pct',    'Host + operations reserve (%)'],
    ['storage_growth_pct',      'Storage growth reserve (%)'],
  ];

  function selField(key, label, choices, def) {
    const opt = choices.map(c => `<option value="${escapeAttr(c)}"${c === def ? ' selected' : ''}>${escapeAttr(c)}</option>`).join('');
    return `<div class="pl-field"><label class="pl-field__label" for="pl-${key}">${label}</label>
      <select id="pl-${key}" data-key="${key}">${opt}</select></div>`;
  }
  function numField(key, label, def) {
    const v = def != null ? ` value="${escapeAttr(def)}"` : '';
    return `<div class="pl-field"><label class="pl-field__label" for="pl-${key}">${label}</label>
      <input type="number" id="pl-${key}" data-key="${key}" min="0" step="1"${v}></div>`;
  }

  function checkbox(c) {
    const req = !!c.required;
    // Required components are locked on (checked + disabled) — a supported
    // deployment can't omit them. gather() still reads disabled checkboxes, and
    // the server merges DEFAULTS, so the value is sent either way.
    const lock = req ? ' checked disabled' : '';
    const title = req ? ' title="Mandatory for a supported deployment — always deployed, can\'t be removed here."' : '';
    return `<label class="pl-check${req ? ' pl-check--locked' : ''}"${title}><input type="checkbox" data-component="${escapeAttr(c.key)}"${lock}>` +
      `<span>${escapeAttr(c.label)}${req ? '<em class="pl-req">required</em>' : ''}</span></label>`;
  }

  function buildForm() {
    const d = opts.defaults || {};
    const all = opts.components || [];
    const required = all.filter(c => c.required).map(checkbox).join('');
    const optional = all.filter(c => !c.required).map(checkbox).join('');
    form.innerHTML = `
      <div class="pl-group">
        <div class="pl-group__title">deployment</div>
        ${selField('size', 'Management vCenter size', opts.sizes || [], d.size)}
        ${selField('availability_model', 'Availability', opts.availability || [], d.availability_model)}
        ${selField('instance_model', 'Instance model', opts.instance_models || [], d.instance_model)}
      </div>
      <div class="pl-group">
        <div class="pl-group__title">hosts</div>
        ${NUMS.map(([k, l]) => numField(k, l, d[k])).join('')}
      </div>
      ${required ? `<div class="pl-group">
        <div class="pl-group__title">core components</div>
        <div class="pl-checks">${required}</div>
      </div>` : ''}
      <div class="pl-group">
        <div class="pl-group__title">optional components</div>
        <div class="pl-checks">${optional}</div>
      </div>
      <details class="pl-advanced">
        <summary>advanced</summary>
        <div class="pl-group">${ADV_NUMS.map(([k, l]) => numField(k, l, d[k])).join('')}</div>
      </details>
      <button class="pl-compute" type="submit">compute sizing →</button>`;
    form.addEventListener('submit', (e) => { e.preventDefault(); compute(); });
  }

  function gather() {
    const inputs = {};
    form.querySelectorAll('select[data-key]').forEach(s => { inputs[s.dataset.key] = s.value; });
    form.querySelectorAll('input[type="number"][data-key]').forEach(n => {
      if (n.value !== '') inputs[n.dataset.key] = Number(n.value);
    });
    form.querySelectorAll('input[type="checkbox"][data-component]').forEach(cb => {
      inputs[cb.dataset.component] = cb.checked ? 'Include' : 'Exclude';
    });
    return inputs;
  }

  function gnum(x) { return Number(x).toLocaleString('en-US'); }

  function render(d) {
    if (!d.components || !d.components.length) {
      results.innerHTML = '<p class="pl-empty">No components in this configuration.</p>';
      return;
    }
    const rows = d.components.map(c =>
      `<tr><td>${escapeAttr(c.name)}</td><td>${gnum(c.nodes)}</td><td>${gnum(c.vcpu)}</td><td>${gnum(c.ram_gb)}</td><td>${gnum(c.disk_gb)}</td></tr>`
    ).join('');
    const t = d.totals || {};
    const hasRuntime = d.components.some(c => /VCF services runtime/i.test(c.name));
    const hs = d.host_summary || [];
    const hostRows = hs.map(x =>
      `<tr><td>${escapeAttr(x.label)}</td><td>${escapeAttr(String(x.value))}</td></tr>`
    ).join('');
    results.innerHTML = `
      <table class="pl-table">
        <thead><tr><th>component</th><th>nodes</th><th>vCPU</th><th>RAM (GB)</th><th>storage (GB)</th></tr></thead>
        <tbody>${rows}</tbody>
        <tfoot><tr><td>total</td><td>${gnum(t.nodes)}</td><td>${gnum(t.vcpu)}</td><td>${gnum(t.ram_gb)}</td><td>${gnum(t.disk_gb)}</td></tr></tfoot>
      </table>
      ${hasRuntime ? `<p class="pl-note pl-note--gloss"><strong>VCF services runtime</strong> (control + worker nodes) is the Kubernetes-based platform that runs VCF's management services — fleet lifecycle, SDDC Manager, software depot — introduced in VCF 9.x. It deploys with every management domain.</p>` : ''}
      ${hs.length ? `
        <h4 class="pl-subhead">host requirement summary</h4>
        <table class="pl-table pl-table--kv"><tbody>${hostRows}</tbody></table>
        <p class="pl-note">How many physical ESX hosts the configuration needs, per-host utilization (sized to tolerate one host failure, N−1), and the vSAN capacity build-up — this is what the host size, oversubscription, and reserve inputs drive. (The component <em>nodes</em> above are appliance VMs; <em>hosts</em> here are physical servers.)</p>` : ''}
      <p class="pl-note">Computed by the VCF Planning &amp; Preparation Workbook's own formulas (no hand-coded math). Figures are appliance footprint; physical host capacity, vSAN overhead and growth headroom are modeled separately in the workbook.</p>`;
  }

  async function compute() {
    statusEl.hidden = false;
    statusEl.dataset.state = 'busy';
    statusEl.textContent = (opts && opts.ready === false)
      ? 'compiling workbook formulas (first run, ~60s)…'
      : 'computing…';
    try {
      const r = await fetch('/api/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: gather() }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || ('HTTP ' + r.status));
      if (opts) opts.ready = true;
      statusEl.hidden = true;
      render(d);
    } catch (e) {
      statusEl.dataset.state = 'error';
      statusEl.textContent = 'error · ' + (e && e.message || e);
    }
  }

  async function ensure() {
    if (built) return;
    built = true;
    try {
      const r = await fetch('/api/plan/options');
      opts = await r.json();
    } catch (e) {
      statusEl.hidden = false;
      statusEl.dataset.state = 'error';
      statusEl.textContent = 'planner unavailable · ' + (e && e.message || e);
      return;
    }
    buildForm();
    results.innerHTML = '<p class="pl-empty">Smallest supported layout shown below — adjust inputs and recompute.</p>';
    compute(); // auto-show the default smallest config
  }

  return { ensure };
})();

(function initViewTabs() {
  const tabs = document.querySelectorAll('.viewtab');
  function setView(v) {
    if (v !== 'planner' && v !== 'chat') v = 'chat';
    document.body.dataset.view = v;
    tabs.forEach(t => {
      const on = t.dataset.view === v;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    try { history.replaceState(null, '', v === 'planner' ? '#planner' : '#'); } catch (_) {}
    if (v === 'planner') planner.ensure();
  }
  tabs.forEach(t => t.addEventListener('click', () => setView(t.dataset.view)));
  window.addEventListener('hashchange', () => setView(location.hash.replace('#', '')));
  setView(location.hash.replace('#', '') || 'chat');
})();
