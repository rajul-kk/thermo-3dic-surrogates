// Interactive chiplet thermal floorplanner. Tiers: preview (layered solver) -> exact (PCG) -> 3D-ICE sign-off.
'use strict';

const $ = id => document.getElementById(id);
const canvas = $('fp-canvas');
const ctx = canvas.getContext('2d');
const PAD = 18;
const CW = 900;

const S = {
  fp: null,            // /floorplan/{geometry}
  offsets: {},         // movable id -> [dx, dy] µm
  lastGood: {},        // last placement the server accepted
  powers: {},          // block -> W/cm²
  res: null,           // last /solve response
  drag: null,
  invalid: null,       // id of an item whose current position was rejected
  inflight: false,
  queued: null,
  scale: 1,
  previewPeak: null,
  epoch: 0,           // bumped on every geometry load; answers from an older epoch are dropped
  note: null,         // one-off message shown with the next result
};

// ── colour map (inferno-like) ────────────────────────────────────────────────
const STOPS = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function colour(t) {
  t = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i;
  return STOPS[i].map((c, k) => Math.round(c + (STOPS[i + 1][k] - c) * f));
}
(function drawColourbar() {
  const c = $('fp-cbar'), g = c.getContext('2d');
  for (let y = 0; y < c.height; y++) {
    const [r, gg, b] = colour(1 - y / (c.height - 1));
    g.fillStyle = `rgb(${r},${gg},${b})`;
    g.fillRect(0, y, c.width, 1);
  }
})();

// ── status ───────────────────────────────────────────────────────────────────
function status(text, cls) {
  $('fp-status').textContent = text;
  $('fp-led').className = 'led' + (cls ? ' ' + cls : '');
}
function warn(list, info) {
  const box = $('fp-warnings');
  box.innerHTML = '';
  (list || []).forEach(w => {
    const d = document.createElement('div');
    d.className = 'fp-warning' + (info ? ' info' : '');
    d.textContent = w;
    box.appendChild(d);
  });
}

// ── geometry ─────────────────────────────────────────────────────────────────
async function loadGeometries() {
  const names = Object.keys(await (await fetch('/geometries')).json());
  const sel = $('fp-geometry');
  names.forEach(n => sel.add(new Option(n, n)));
  sel.value = names.includes('geometry5') ? 'geometry5' : names[0];
  sel.onchange = () => loadFloorplan(sel.value);
  await loadFloorplan(sel.value);
}

async function loadFloorplan(name) {
  const epoch = ++S.epoch;
  S.queued = null; S.drag = null;
  const r = await fetch('/floorplan/' + name);
  const fp = await r.json();
  if (epoch !== S.epoch) return;
  S.fp = fp;
  S.offsets = {};
  S.fp.movables.forEach(m => { S.offsets[m.id] = [0, 0]; });
  S.lastGood = structuredClone(S.offsets);
  S.powers = {};
  S.fp.blocks.filter(b => !b.tsv).forEach(b => {
    S.powers[b.name] = b.limit_wcm2 <= 10 ? 6 : (/rdl/i.test(b.name) ? 3 : 60);
  });
  S.res = null; S.invalid = null;
  S.scale = (CW - 2 * PAD) / S.fp.width_um;
  canvas.width = CW;
  canvas.height = Math.round(S.fp.length_um * S.scale + 2 * PAD);

  const layer = $('fp-layer');
  layer.innerHTML = '';
  layer.add(new Option('Hottest layer', 'hottest'));
  S.fp.layers.forEach(l => layer.add(new Option(l.name + (l.active ? '  (powered)' : ''), l.name)));
  buildPowerPanel();
  S.previewPeak = null; S.note = null; S.inflight = false;
  ['r-peak', 'r-rise', 'r-layer', 'r-power', 'r-solver', 't-preview', 't-exact', 't-ice'].forEach(id => { $(id).textContent = '–'; });
  $('t-ice-s').textContent = 'not run';
  draw();
  warn([]);
  await solve('exact');
}

function buildPowerPanel() {
  const host = $('fp-power');
  host.innerHTML = '';
  const groups = {};
  S.fp.blocks.filter(b => !b.tsv).forEach(b => { (groups[b.movable || 'blocks'] ||= []).push(b); });
  Object.entries(groups).forEach(([gid, blocks]) => {
    const g = document.createElement('div');
    g.className = 'fp-group';
    g.innerHTML = `<div class="fp-group-title">${gid}</div>`;
    blocks.forEach(b => {
      const row = document.createElement('label');
      row.className = 'fp-block';
      const name = document.createElement('span');
      name.textContent = b.name; name.title = `${b.name} · ${b.layer} · ${b.area_cm2.toFixed(2)} cm²`;
      const range = Object.assign(document.createElement('input'), { type: 'range', min: 0, max: b.limit_wcm2, step: b.limit_wcm2 <= 10 ? 0.5 : 5, value: S.powers[b.name] });
      const num = Object.assign(document.createElement('input'), { type: 'number', min: 0, max: b.limit_wcm2, step: 'any', value: S.powers[b.name] });
      const set = (v, final) => {
        v = Math.min(b.limit_wcm2, Math.max(0, Number(v) || 0));
        S.powers[b.name] = v; range.value = v; num.value = v;
        solve(final && $('fp-auto-exact').checked ? 'exact' : 'preview');
      };
      range.oninput = () => set(range.value, false);
      range.onchange = () => set(range.value, true);
      num.onchange = () => set(num.value, true);
      row.append(name, range, num);
      g.appendChild(row);
    });
    host.appendChild(g);
  });
}

// ── solve ────────────────────────────────────────────────────────────────────
function body(mode) {
  return {
    geometry: S.fp.name, offsets: S.offsets, power_blocks: S.powers,
    htc: Number($('fp-htc').value), t_ambient: Number($('fp-tamb').value),
    mode, layer: $('fp-layer').value, t_limit_c: Number($('fp-limit').value) || null,
  };
}

async function solve(mode) {
  if (S.inflight) { S.queued = mode === 'exact' || S.queued === 'exact' ? 'exact' : 'preview'; return; }
  S.inflight = true;
  const epoch = S.epoch;
  const ok = await run('preview');
  if (ok && mode === 'exact' && !S.drag && epoch === S.epoch) await run('exact');
  if (epoch !== S.epoch) return;          // the geometry changed underneath this request
  S.inflight = false;
  draw();
  if (S.queued) { const q = S.queued; S.queued = null; solve(q); }
}

async function run(mode) {
  let ok = false;
  const epoch = S.epoch;
  status(mode === 'exact' ? 'SOLVING (EXACT)' : 'PREVIEW', 'busy');
  try {
    const sent = structuredClone(S.offsets);
    const r = await fetch('/solve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body(mode)) });
    if (epoch !== S.epoch) return false;
    if (r.status === 422) {
      const d = await r.json();
      S.invalid = S.drag ? S.drag.id : S.invalid;
      warn([typeof d.detail === 'string' ? d.detail : 'That placement is not valid.']);
      status('INVALID PLACEMENT', 'fail');
    } else if (!r.ok) {
      warn(['Solver error: ' + (await r.text())]);
      status('ERROR', 'fail');
    } else {
      const d = await r.json();
      if (epoch !== S.epoch) return false;
      S.res = d; S.invalid = null; S.lastGood = sent;
      if (!S.drag) S.offsets = Object.fromEntries(Object.entries(d.offsets).map(([k, v]) => [k, [v[0], v[1]]]));
      show(d);
      status('READY');
      ok = true;
    }
  } catch (e) {
    warn(['Cannot reach the solver: ' + e.message]);
    status('OFFLINE', 'fail');
  }
  draw();
  return ok;
}

function show(d) {
  const limit = Number($('fp-limit').value);
  $('r-peak').textContent = d.peak_c.toFixed(1) + ' °C';
  $('r-peak').classList.toggle('over', !!limit && d.peak_c > limit);
  $('r-rise').textContent = d.rise_k.toFixed(1) + ' K';
  $('r-layer').textContent = d.hotspot.layer;
  $('r-power').textContent = d.power_w.toFixed(0) + ' W';
  const ms = d.seconds * 1000, took = ms < 1000 ? ms.toFixed(0) + ' ms' : (ms / 1000).toFixed(1) + ' s';
  $('r-solver').textContent = d.mode + ' · ' + took;
  if (d.mode === 'exact') {
    $('t-exact').textContent = d.peak_c.toFixed(2) + ' °C';
    $('t-exact-s').textContent = `preconditioned CG · ${d.iterations} iterations · ${took}`;
    if (S.previewPeak !== null) $('t-preview-s').textContent = `layered solver · ${(S.previewPeak - d.peak_c >= 0 ? '+' : '') + (S.previewPeak - d.peak_c).toFixed(2)} K vs exact`;
  } else {
    S.previewPeak = d.peak_c;
    $('t-preview').textContent = d.peak_c.toFixed(2) + ' °C';
    $('t-preview-s').textContent = 'layered solver · ' + took;
    $('t-exact').textContent = '–'; $('t-exact-s').textContent = 'preconditioned CG';
    $('t-ice').textContent = '–'; $('t-ice-s').textContent = 'not run';
  }
  $('cb-max').textContent = d.field_max_c.toFixed(1) + ' °C';
  $('cb-min').textContent = d.field_min_c.toFixed(1) + ' °C';
  warn(d.warnings);
  if (S.note) {
    const n = document.createElement('div');
    n.className = 'fp-warning info'; n.textContent = S.note;
    $('fp-warnings').appendChild(n);
    if (d.mode === 'exact' || !$('fp-auto-exact').checked) S.note = null;
  }
}

// ── drawing ──────────────────────────────────────────────────────────────────
const X = x => PAD + x * S.scale;
const Y = y => canvas.height - PAD - y * S.scale;
const heat = document.createElement('canvas');

function draw() {
  if (!S.fp) return;
  const { width_um: W, length_um: H } = S.fp;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const d = S.res;
  if (d) {
    heat.width = d.nx; heat.height = d.ny;
    const hc = heat.getContext('2d'), img = hc.createImageData(d.nx, d.ny);
    const lo = d.field_min_c, span = Math.max(1e-6, d.field_max_c - lo);
    for (let i = 0; i < d.nx; i++) {
      for (let j = 0; j < d.ny; j++) {
        const [r, g, b] = colour((d.field[i][j] - lo) / span);
        const p = 4 * ((d.ny - 1 - j) * d.nx + i);
        img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
      }
    }
    hc.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(heat, X(0), Y(H), W * S.scale, H * S.scale);
  }

  ctx.strokeStyle = '#6b7280'; ctx.lineWidth = 1;
  ctx.strokeRect(X(0) + 0.5, Y(H) + 0.5, W * S.scale, H * S.scale);

  const shown = d ? d.layer : null;
  S.fp.movables.forEach(m => {
    const [dx, dy] = S.offsets[m.id];
    ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1;
    S.fp.blocks.filter(b => b.movable === m.id && !b.tsv && (!shown || b.layer === shown)).forEach(b => {
      ctx.strokeRect(X(b.x + dx), Y(b.y + dy + b.height), b.width * S.scale, b.height * S.scale);
    });
    ctx.setLineDash([]);
    const bad = S.invalid === m.id, active = S.drag && S.drag.id === m.id;
    ctx.strokeStyle = bad ? '#f87171' : (active ? '#fcd34d' : '#f59e0b');
    ctx.lineWidth = active || bad ? 2.5 : 1.5;
    ctx.strokeRect(X(m.x + dx), Y(m.y + dy + m.height), m.width * S.scale, m.height * S.scale);
    ctx.font = '11px "Share Tech Mono", monospace';
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    const tw = ctx.measureText(m.id).width + 8;
    ctx.fillRect(X(m.x + dx) + 2, Y(m.y + dy + m.height) + 2, tw, 15);
    ctx.fillStyle = bad ? '#f87171' : '#fcd34d';
    ctx.fillText(m.id, X(m.x + dx) + 6, Y(m.y + dy + m.height) + 13);
  });

  if (d && !S.drag) {
    const hx = X(d.hotspot.x_um), hy = Y(d.hotspot.y_um);
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(hx, hy, 7, 0, 2 * Math.PI);
    ctx.moveTo(hx - 12, hy); ctx.lineTo(hx + 12, hy); ctx.moveTo(hx, hy - 12); ctx.lineTo(hx, hy + 12);
    ctx.stroke();
  }
}

// ── interaction ──────────────────────────────────────────────────────────────
function pointer(e) {
  const r = canvas.getBoundingClientRect(), k = canvas.width / r.width;
  const px = (e.clientX - r.left) * k, py = (e.clientY - r.top) * k;
  return [(px - PAD) / S.scale, (canvas.height - PAD - py) / S.scale];
}
function hit(x, y) {
  for (let i = S.fp.movables.length - 1; i >= 0; i--) {
    const m = S.fp.movables[i], [dx, dy] = S.offsets[m.id];
    if (x >= m.x + dx && x <= m.x + dx + m.width && y >= m.y + dy && y <= m.y + dy + m.height) return m;
  }
  return null;
}

canvas.addEventListener('pointerdown', e => {
  if (!S.fp) return;
  const [x, y] = pointer(e), m = hit(x, y);
  if (!m) return;
  canvas.setPointerCapture(e.pointerId);
  S.drag = { id: m.id, x0: x, y0: y, off0: [...S.offsets[m.id]] };
  canvas.classList.add('dragging');
  draw();
});

canvas.addEventListener('pointermove', e => {
  if (!S.fp) return;
  const [x, y] = pointer(e);
  if (!S.drag) {
    const d = S.res;
    if (d && x >= 0 && y >= 0 && x < S.fp.width_um && y < S.fp.length_um) {
      const i = Math.min(d.nx - 1, Math.floor(x / S.fp.width_um * d.nx)), j = Math.min(d.ny - 1, Math.floor(y / S.fp.length_um * d.ny));
      $('fp-hover').textContent = `${d.field[i][j].toFixed(1)} °C at x ${(x / 1000).toFixed(2)} mm, y ${(y / 1000).toFixed(2)} mm · layer ${d.layer}`;
    }
    return;
  }
  const m = S.fp.movables.find(q => q.id === S.drag.id);
  const [gx, gy] = S.fp.grid_um;
  let nx = Math.round((m.x + S.drag.off0[0] + x - S.drag.x0) / gx) * gx;
  let ny = Math.round((m.y + S.drag.off0[1] + y - S.drag.y0) / gy) * gy;
  nx = Math.min(S.fp.width_um - m.width, Math.max(0, nx));
  ny = Math.min(S.fp.length_um - m.height, Math.max(0, ny));
  const next = [nx - m.x, ny - m.y], cur = S.offsets[m.id];
  if (next[0] === cur[0] && next[1] === cur[1]) return;
  S.offsets[m.id] = next;
  draw();
  solve('preview');
});

function endDrag() {
  if (!S.drag) return;
  const { id, off0 } = S.drag;
  S.drag = null;
  canvas.classList.remove('dragging');
  if (S.invalid) {                       // dropped on an invalid spot: back to where the drag started
    S.offsets[id] = off0;
    S.invalid = null;
    S.note = `${id} was put back: that spot overlaps another item or leaves the package.`;
  }
  solve($('fp-auto-exact').checked ? 'exact' : 'preview');
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

// ── controls ─────────────────────────────────────────────────────────────────
['fp-htc', 'fp-tamb', 'fp-limit', 'fp-layer'].forEach(id => { $(id).addEventListener('change', () => solve(S.res && S.res.mode === 'exact' ? 'exact' : 'preview')); });

async function busy(btn, label, fn) {
  const old = btn.textContent;
  btn.disabled = true; btn.textContent = label;
  try { await fn(); } catch (e) { warn(['Request failed: ' + e.message]); status('ERROR', 'fail'); }
  btn.disabled = false; btn.textContent = old;
}
const post = (url, b) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });

$('btn-exact').onclick = () => solve('exact');

$('btn-reset').onclick = () => {
  S.fp.movables.forEach(m => { S.offsets[m.id] = [0, 0]; });
  solve('exact');
};

$('btn-optimise').onclick = e => busy(e.target, 'Optimising…', async () => {
  status('OPTIMISING', 'busy');
  const r = await post('/optimise', { ...body('preview'), evaluations: 400 });
  if (!r.ok) { warn([(await r.json()).detail || 'Optimisation failed.']); status('ERROR', 'fail'); return; }
  const d = await r.json();
  S.offsets = Object.fromEntries(Object.entries(d.offsets).map(([k, v]) => [k, [v[0], v[1]]]));
  const gain = d.rise_before_k - d.rise_after_k;
  S.note = gain > 0.005
    ? `Optimised placement: peak rise ${d.rise_before_k.toFixed(2)} → ${d.rise_after_k.toFixed(2)} K (−${gain.toFixed(2)} K) after ${d.evaluations} trial placements in ${d.seconds.toFixed(1)} s.`
    : `No better placement found in ${d.evaluations} trial placements; this one is already a local best.`;
  await solve('exact');
});

$('btn-export').onclick = e => busy(e.target, 'Exporting…', async () => {
  const r = await post('/export/3dice', body('preview'));
  if (!r.ok) { warn([(await r.json()).detail || 'Export failed.']); return; }
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(await r.blob()), download: S.fp.name + '_3dice.zip' });
  a.click(); URL.revokeObjectURL(a.href);
});

$('btn-signoff').onclick = e => busy(e.target, 'Running 3D-ICE…', async () => {
  const r = await post('/signoff', body('preview'));
  if (!r.ok) { warn([(await r.json()).detail || 'Could not queue the 3D-ICE run.']); return; }
  const { job_id } = await r.json();
  $('t-ice').textContent = '…'; $('t-ice-s').textContent = 'job ' + job_id + ' running';
  for (;;) {
    await new Promise(res => setTimeout(res, 1500));
    const j = await (await fetch('/jobs/' + job_id)).json();
    if (j.status === 'done') {
      const peak = j.stats && j.stats.temperature ? j.stats.temperature.max_c : null;
      $('t-ice').textContent = peak === null ? 'done' : peak.toFixed(2) + ' °C';
      $('t-ice-s').textContent = 'job ' + job_id + ' · results in the pipeline tab';
      return;
    }
    if (j.status === 'failed') {
      $('t-ice').textContent = 'failed'; $('t-ice-s').textContent = j.error || 'see the pipeline tab';
      return;
    }
  }
});

loadGeometries().catch(e => { warn(['Cannot load geometries: ' + e.message]); status('OFFLINE', 'fail'); });
