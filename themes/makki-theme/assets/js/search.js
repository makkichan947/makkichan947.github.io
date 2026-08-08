/* ==========================================================================
   Makki Theme — local full-text search over the Hugo JSON index.
   Loaded lazily on first open; runs on every keystroke (no debounce —
   the corpus is small, and input latency is a regression).
   ========================================================================== */
(() => {
  'use strict';

  const MAX_RESULTS = 10;

  let index = null;
  let loaded = false;
  let loading = null;
  let current = -1;

  const resultsEl = () => document.getElementById('search-results');
  const inputEl = () => document.getElementById('search-input');

  const escapeHtml = (s) =>
    s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function loadIndex() {
    if (loaded || loading) return loading;
    loading = fetch('/index.json', { credentials: 'same-origin' })
      .then((r) => r.json())
      .then((data) => { index = data; loaded = true; })
      .catch((err) => { console.error('搜索索引加载失败:', err); })
      .finally(() => { loading = null; });
    return loading;
  }

  function score(page, tokens) {
    let s = 0;
    const title = (page.title || '').toLowerCase();
    const content = (page.content || '').toLowerCase();
    const tags = (page.tags || []).map((t) => String(t).toLowerCase());
    for (const tok of tokens) {
      if (title === tok) s += 120;
      else if (title.startsWith(tok)) s += 90;
      else if (title.includes(tok)) s += 60;
      if (tags.some((t) => t.includes(tok))) s += 45;
      if (content.includes(tok)) s += 20;
      if (snippetAt(content, tok)) s += 15;
    }
    return s;
  }

  function snippetAt(content, tok) {
    const i = content.indexOf(tok);
    return i === -1 ? null : i;
  }

  function highlight(text, tokens) {
    let out = escapeHtml(text);
    for (const tok of tokens) {
      out = out.replace(new RegExp(escapeHtml(tok).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'),
        (m) => `<mark>${m}</mark>`);
    }
    return out;
  }

  function snippet(content, tokens, len) {
    if (!content) return '';
    const plain = content.replace(/\s+/g, ' ').trim();
    let i = Infinity;
    for (const tok of tokens) {
      const at = snippetAt(plain.toLowerCase(), tok.toLowerCase());
      if (at !== null && at < i) i = at;
    }
    let start = 0;
    if (i !== Infinity) start = Math.max(0, i - Math.floor(len / 3));
    const text = plain.slice(start, start + len);
    return highlight(text, tokens) + (start + len < plain.length ? '…' : '');
  }

  function render(query) {
    const box = resultsEl();
    if (!box) return;
    current = -1;

    const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!tokens.length) { box.innerHTML = ''; return; }

    const scored = index
      .map((page) => ({ page, s: score(page, tokens) }))
      .filter((r) => r.s > 0)
      .sort((a, b) => b.s - a.s)
      .slice(0, MAX_RESULTS);

    if (!scored.length) {
      box.innerHTML = `<div class="search-empty">没有找到与 “${escapeHtml(query)}” 相关的内容</div>`;
      return;
    }

    box.innerHTML = scored.map(({ page }) => `
      <div class="search-result" data-uri="${escapeHtml(page.uri)}">
        <h3><a href="${escapeHtml(page.uri)}">${highlight(page.title, tokens)}</a></h3>
        <p>${snippet(page.content, tokens, 90)}</p>
        <div class="search-meta">
          ${page.section ? `<span class="search-type">${escapeHtml(page.section)}</span>` : ''}
          ${(page.tags || []).slice(0, 3).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join('')}
        </div>
      </div>`).join('');

    /* Hint in the direction of the gesture: results grow from the input. */
    qa('.search-result').forEach((el, i) => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(6px)';
      setTimeout(() => {
        el.style.transition = 'opacity 200ms ease-out, transform 200ms ease-out';
        el.style.opacity = '1';
        el.style.transform = '';
      }, Math.min(i * 30, 150));
    });
  }

  const qa = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

  function move(delta) {
    const items = qa('.search-result', resultsEl() || document);
    if (!items.length) return;
    current = (current + delta + items.length) % items.length;
    items.forEach((el, i) => el.classList.toggle('is-current', i === current));
    items[current].scrollIntoView({ block: 'nearest' });
  }

  function commit() {
    const items = qa('.search-result', resultsEl() || document);
    const el = items.find((x) => x.classList.contains('is-current')) || items[0];
    if (el) {
      const link = el.querySelector('a');
      if (link) window.location.href = link.getAttribute('href');
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const input = inputEl();
    if (!input) return;

    /* Load the index the first time the sheet opens — never on page load. */
    document.addEventListener('makki:sheet-open', () => {
      loadIndex().then(() => { if (document.activeElement === input) render(input.value); });
    });

    input.addEventListener('input', () => {
      if (!loaded) { loadIndex(); return; }
      render(input.value);
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
      else if (e.key === 'Enter') { e.preventDefault(); commit(); }
    });

    resultsEl()?.addEventListener('click', (e) => {
      const item = e.target.closest('.search-result');
      if (item) {
        const link = item.querySelector('a');
        if (link) window.location.href = link.getAttribute('href');
      }
    });
  });
})();
