// ── State ──────────────────────────────────────────────────────────────────
let geometries = {};       // name → geometry info from /geometries
let activeLogWs = null;    // open WebSocket for live logs
let activeHeatmapJobId = null;

// ── Utilities ──────────────────────────────────────────────────────────────

function showTab(tabId, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');

  if (tabId === 'tab-geometries') loadGeometries();
  if (tabId === 'tab-configure')  loadGeometriesIntoSelect();
  if (tabId === 'tab-queue')      refreshQueue();
  if (tabId === 'tab-results')    refreshResults();
}

function toast(msg, ms = 3000) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), ms);
}

function fmtTime(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleTimeString();
}

function badgeHtml(status) {
  return `<span class="badge badge-${status}">${status}</span>`;
}

// ── Geometries tab ─────────────────────────────────────────────────────────

async function loadGeometries() {
  try {
    const r = await fetch('/geometries');
    geometries = await r.json();
    const grid = document.getElementById('geometry-cards');
    grid.innerHTML = '';
    for (const [name, g] of Object.entries(geometries)) {
      const customTag = g.custom ? '<span class="tag tag-custom">custom</span>' : '';
      const blockTags = g.power_blocks.map(b => `<span class="tag">${b}</span>`).join('');
      grid.innerHTML += `
        <div class="card">
          <h3>${name}</h3>
          <p>${g.stack_type} &middot; ${g.layers} layers &middot; ${g.mesh_points.toLocaleString()} pts</p>
          <p style="margin-top:3px">${g.die_length_um} &times; ${g.die_width_um} &micro;m</p>
          <div style="margin-top:10px">${blockTags} ${customTag}</div>
        </div>`;
    }
  } catch (e) {
    toast('Failed to load geometries: ' + e.message, 5000);
  }
}

function showCustomForm() {
  const f = document.getElementById('custom-form');
  f.style.display = f.style.display === 'none' ? 'block' : 'none';
}

async function fetchSchema() {
  try {
    const r = await fetch('/geometries/schema');
    const data = await r.json();
    document.getElementById('custom-yaml').value = data.schema;
  } catch (e) {
    toast('Failed to fetch template: ' + e.message, 5000);
  }
}

async function uploadCustomGeometry() {
  const yaml = document.getElementById('custom-yaml').value.trim();
  if (!yaml) { toast('Paste a YAML spec first'); return; }
  try {
    const r = await fetch('/geometries', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml }),
    });
    if (r.ok) {
      const d = await r.json();
      toast(`Registered "${d.name}" (${d.layers} layers, ${d.power_blocks} blocks)`);
      document.getElementById('custom-form').style.display = 'none';
      loadGeometries();
    } else {
      const err = await r.json();
      toast(`Error: ${err.detail}`, 6000);
    }
  } catch (e) {
    toast('Request failed: ' + e.message, 5000);
  }
}

// ── Configure tab ──────────────────────────────────────────────────────────

async function loadGeometriesIntoSelect() {
  if (!Object.keys(geometries).length) {
    try {
      const r = await fetch('/geometries');
      geometries = await r.json();
    } catch (e) { return; }
  }
  const sel = document.getElementById('cfg-geometry');
  const prev = sel.value;
  sel.innerHTML = Object.keys(geometries)
    .map(n => `<option value="${n}">${n}</option>`).join('');
  if (prev && geometries[prev]) sel.value = prev;
  onGeometryChange();
}

function onGeometryChange() {
  const name = document.getElementById('cfg-geometry').value;
  const geom = geometries[name];
  if (!geom) return;
  const container = document.getElementById('block-inputs');
  container.innerHTML = geom.power_blocks.map(b => `
    <div class="form-row">
      <label>${b}</label>
      <input type="number" id="block-${b}" value="1.0" min="0" step="0.1" placeholder="W/cm²">
    </div>`).join('');
}

async function submitJob() {
  const geometry = document.getElementById('cfg-geometry').value;
  const geom = geometries[geometry];
  if (!geom) { toast('Select a geometry first'); return; }

  const name = document.getElementById('cfg-name').value.trim()
    || `${geometry}_${Date.now()}`;
  const htc       = parseFloat(document.getElementById('cfg-htc').value);
  const t_ambient = parseFloat(document.getElementById('cfg-tamb').value);
  const pattern   = document.getElementById('cfg-pattern').value;

  const power_blocks = {};
  for (const b of geom.power_blocks) {
    const el = document.getElementById(`block-${b}`);
    power_blocks[b] = parseFloat(el ? el.value : '0');
  }

  try {
    const r = await fetch('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        geometry,
        scenario_name: name,
        scenario_params: { power_blocks, htc, t_ambient, pattern },
      }),
    });
    if (r.ok) {
      const d = await r.json();
      toast(`Job ${d.job_id} queued`);
      document.querySelector('nav button:nth-child(3)').click();
    } else {
      const err = await r.json();
      toast(`Error: ${err.detail}`, 6000);
    }
  } catch (e) {
    toast('Request failed: ' + e.message, 5000);
  }
}

// ── Queue tab ──────────────────────────────────────────────────────────────

async function refreshQueue() {
  try {
    const r = await fetch('/jobs');
    const jobs = await r.json();
    const tbody = document.getElementById('queue-tbody');
    if (!jobs.length) {
      tbody.innerHTML = '<tr><td colspan="6"><div class="empty-state">No jobs yet &mdash; go to Configure to submit one</div></td></tr>';
      return;
    }
    tbody.innerHTML = jobs.map(j => `
      <tr>
        <td style="font-family:monospace;font-size:0.8rem">${j.job_id}</td>
        <td>${j.geometry}</td>
        <td>${j.scenario_name}</td>
        <td>${badgeHtml(j.status)}</td>
        <td>${fmtTime(j.created_at)}</td>
        <td style="display:flex;gap:6px;flex-wrap:wrap">
          ${j.status !== 'pending'
            ? `<button class="btn btn-secondary" style="padding:4px 10px;font-size:0.75rem" onclick="openLogs('${j.job_id}')">Logs</button>` : ''}
          ${j.status === 'pending'
            ? `<button class="btn btn-danger" style="padding:4px 10px;font-size:0.75rem" onclick="cancelJob('${j.job_id}')">Cancel</button>` : ''}
        </td>
      </tr>`).join('');
  } catch (e) {
    toast('Failed to refresh queue: ' + e.message, 5000);
  }
}

function openLogs(jobId) {
  if (activeLogWs) { try { activeLogWs.close(); } catch (_) {} activeLogWs = null; }

  const container = document.getElementById('log-panel-container');
  container.style.display = 'block';
  document.getElementById('log-job-id').textContent = jobId;
  const panel = document.getElementById('log-panel');
  panel.innerHTML = '';

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/jobs/${jobId}/logs`);
  activeLogWs = ws;

  ws.onmessage = (e) => {
    if (e.data.startsWith('__STATUS__')) {
      const status = e.data.split(' ')[1];
      appendLog(panel, `--- Job ${status.toUpperCase()} ---`,
                status === 'done' ? 'log-info' : 'log-error');
      refreshQueue();
      return;
    }
    const lower = e.data.toLowerCase();
    const cls = lower.includes('error') ? 'log-error'
               : lower.includes('warning') ? 'log-warning'
               : lower.startsWith('info') ? 'log-info' : '';
    appendLog(panel, e.data, cls);
  };

  ws.onerror = () => appendLog(panel, 'WebSocket error', 'log-error');
  ws.onclose = () => appendLog(panel, '--- connection closed ---', '');
}

function appendLog(panel, text, cls) {
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = text;
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
}

async function cancelJob(jobId) {
  try {
    const r = await fetch(`/jobs/${jobId}`, { method: 'DELETE' });
    if (r.ok) { toast('Job cancelled'); refreshQueue(); }
    else toast('Could not cancel job');
  } catch (e) {
    toast('Request failed: ' + e.message, 5000);
  }
}

// ── Results tab ────────────────────────────────────────────────────────────

async function refreshResults() {
  try {
    const r = await fetch('/jobs');
    const jobs = (await r.json()).filter(j => j.status === 'done');
    const tbody = document.getElementById('results-tbody');
    if (!jobs.length) {
      tbody.innerHTML = '<tr><td colspan="6"><div class="empty-state">No completed jobs yet</div></td></tr>';
      return;
    }
    tbody.innerHTML = jobs.map(j => {
      const peak = j.stats?.hotspot?.peak_temperature_c != null
        ? j.stats.hotspot.peak_temperature_c.toFixed(1) : '—';
      return `
        <tr>
          <td style="font-family:monospace;font-size:0.8rem">${j.job_id}</td>
          <td>${j.geometry}</td>
          <td>${j.scenario_name}</td>
          <td>${peak}</td>
          <td>${fmtTime(j.finished_at)}</td>
          <td style="display:flex;gap:6px;flex-wrap:wrap">
            <a href="/results/${j.job_id}.npz" download="${j.scenario_name}.npz">
              <button class="btn btn-secondary" style="padding:4px 10px;font-size:0.75rem">NPZ</button>
            </a>
            <button class="btn btn-secondary" style="padding:4px 10px;font-size:0.75rem"
              onclick="showHeatmap('${j.job_id}', '${j.scenario_name}')">Heatmap</button>
          </td>
        </tr>`;
    }).join('');
  } catch (e) {
    toast('Failed to load results: ' + e.message, 5000);
  }
}

function showHeatmap(jobId, scenarioName) {
  activeHeatmapJobId = jobId;
  document.getElementById('heatmap-section').style.display = 'block';
  document.getElementById('heatmap-job-id').textContent = scenarioName;
  document.getElementById('heatmap-layer').value = '0';
  refreshHeatmap();
}

function refreshHeatmap() {
  if (!activeHeatmapJobId) return;
  const layer = document.getElementById('heatmap-layer').value;
  const img = document.getElementById('heatmap-img');
  img.style.display = 'none';
  img.onload = () => { img.style.display = 'block'; };
  img.onerror = () => { toast('Layer not found or heatmap unavailable'); };
  img.src = `/results/${activeHeatmapJobId}/heatmap.png?layer=${layer}&t=${Date.now()}`;
}

// ── Init ───────────────────────────────────────────────────────────────────

loadGeometries();
