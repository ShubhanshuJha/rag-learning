const log = document.getElementById('log');
const input = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const resetBtn = document.getElementById('resetBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

let emptyState = document.getElementById('emptyState');
let requestCount = 0;

/** Fetch and display which embedding/LLM backend is currently active. */
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    statusDot.classList.remove('is-offline');
    statusDot.classList.add('is-online');
    statusText.textContent = `${data.llm_backend} · ${data.llm_model} · ${data.embedding_model}`;
  } catch (err) {
    statusDot.classList.remove('is-online');
    statusDot.classList.add('is-offline');
    statusText.textContent = 'pipeline unreachable';
  }
}

/** Escape user/model text before inserting as HTML. */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str == null ? '' : String(str);
  return div.innerHTML;
}

/** Normalize a similarity score (assumed 0–1, but clamps defensively) to a percent. */
function scoreToPercent(score) {
  const n = Number(score);
  if (Number.isNaN(n)) return 0;
  const pct = n <= 1 ? n * 100 : n;
  return Math.max(0, Math.min(100, pct));
}

function removeEmptyState() {
  if (emptyState) {
    emptyState.remove();
    emptyState = null;
  }
}

/** Append a new request entry with a loading placeholder, return the entry element. */
function createEntry(query) {
  requestCount += 1;
  const idx = String(requestCount).padStart(2, '0');

  const entry = document.createElement('div');
  entry.className = 'entry';
  entry.innerHTML = `
    <div class="entry__request">
      <span class="entry__tag">REQ ${idx}</span>
      <p class="entry__query">${escapeHtml(query)}</p>
    </div>
    <div class="entry__response is-loading">
      <span class="entry__tag entry__tag--response">RESPONSE</span>
      <div class="trace-loader" aria-label="Waiting for response">
        <span></span><span></span><span></span><span></span><span></span>
      </div>
    </div>
  `;
  log.appendChild(entry);
  entry.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return entry;
}

/** Build the sources markup (signal bars + preview) for a response. */
function buildSourcesHtml(sources) {
  if (!sources || !sources.length) return '';

  const rows = sources.map((s, i) => {
    const pct = scoreToPercent(s.score);
    return `
      <div class="source" style="--delay:${i * 70}ms">
        <div class="source__meta">
          <span class="source__file">${escapeHtml(s.source)}</span>
          <span class="source__page">p.${escapeHtml(s.page)}</span>
        </div>
        <div class="source__bar">
          <div class="source__bar-fill" data-target="${pct}"></div>
        </div>
        <span class="source__score">${pct.toFixed(0)}%</span>
        <p class="source__preview">${escapeHtml(s.preview)}</p>
      </div>
    `;
  }).join('');

  return `
    <div class="sources">
      <p class="sources__label">SOURCES TRACED — ${sources.length}</p>
      ${rows}
    </div>
  `;
}

/** Fill in a completed response (answer + sources) into an existing entry. */
function renderResponse(entry, data) {
  const responseEl = entry.querySelector('.entry__response');
  responseEl.classList.remove('is-loading');

  responseEl.innerHTML = `
    <span class="entry__tag entry__tag--response">RESPONSE${data.elapsed_ms != null ? ` · ${data.elapsed_ms}ms` : ''}</span>
    <p class="entry__answer">${escapeHtml(data.answer)}</p>
    <button class="entry__copy" type="button">copy answer</button>
    ${buildSourcesHtml(data.sources)}
  `;

  responseEl.querySelector('.entry__copy').addEventListener('click', (e) => {
    navigator.clipboard.writeText(data.answer || '').then(() => {
      const btn = e.currentTarget;
      const original = btn.textContent;
      btn.textContent = 'copied';
      btn.classList.add('is-copied');
      setTimeout(() => {
        btn.textContent = original;
        btn.classList.remove('is-copied');
      }, 1400);
    });
  });

  // Trigger the bar-fill animation on the next frame so CSS transitions apply.
  requestAnimationFrame(() => {
    responseEl.querySelectorAll('.source__bar-fill').forEach((el) => {
      el.style.width = `${el.dataset.target}%`;
    });
  });

  entry.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

/** Render an error state into an entry's response block. */
function renderError(entry, message) {
  const responseEl = entry.querySelector('.entry__response');
  responseEl.classList.remove('is-loading');
  responseEl.classList.add('is-error');
  responseEl.innerHTML = `
    <span class="entry__tag entry__tag--error">ERROR</span>
    <p class="entry__answer">${escapeHtml(message)}</p>
  `;
}

async function sendQuery() {
  const value = input.value.trim();
  if (!value) return;

  removeEmptyState();

  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;

  const entry = createEntry(value);

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Something went wrong.');
    renderResponse(entry, data);
  } catch (err) {
    renderError(entry, err.message || 'Request failed. Is the Flask server running?');
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

function resetSession() {
  log.innerHTML = `
    <div class="empty-state" id="emptyState">
      <p class="empty-state__eyebrow">NO QUERIES YET</p>
      <p class="empty-state__body">
        Feed it PDFs, documents, spreadsheets, or plain text — then ask anything.
        Every retrieved source traces alongside the answer below.
      </p>
    </div>
  `;
  emptyState = document.getElementById('emptyState');
  requestCount = 0;
}

sendBtn.addEventListener('click', sendQuery);
resetBtn.addEventListener('click', resetSession);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendQuery();
});

fetchStatus();
input.focus();
