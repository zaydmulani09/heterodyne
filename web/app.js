"use strict";
// heterodyne UI. Every number drawn here comes from the experiment JSON produced by the engine.
const TOKEN = document.querySelector('meta[name="hd-token"]').content;
const $ = (s, r = document) => r.querySelector(s);
const main = $("#main");
const S = { state: null, exp: null, file: null, chunk: 0, sel: 0, busy: false, dirty: false };

function h(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) n.append(k instanceof Node ? k : String(k));
  return n;
}
const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...kids) {
  const n = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, v);
  for (const k of kids.flat()) if (k != null) n.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return n;
}

async function api(path, body) {
  const r = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "X-Heterodyne-Token": TOKEN, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

// ---------- formatting ----------
const fmtKL = (x) => (x == null ? "–" : x === 0 ? "0" : x < 1e-3 ? x.toExponential(2) : x.toFixed(4));
const pct = (x) => (x == null ? "–" : (100 * x).toFixed(2) + "%");
const short = (p) => (p || "").split(/[\\/]/).pop();
const vis = (t) => t.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
const tok = (id) => (S.exp.results.vocab[id] ?? `#${id}`);
// How much A preferred its own top token over B's (files before v0.1 lack gap_ab).
const gap = (p) => p.gap_ab ?? p.margin_a;
const dark = () => matchMedia("(prefers-color-scheme: dark)").matches;

// Sequential violet ramp for divergence (never the arm colours).
const RAMP_L = ["#eef1f4", "#ddd2ec", "#b9a2dc", "#8b65c3", "#5d2e92", "#2f1052"];
const RAMP_D = ["#171e26", "#2c2443", "#4b3376", "#7a4fb3", "#ad82e6", "#eadaff"];
const parsed = new Map();
function ramp(t) {
  const key = dark() ? "d" : "l";
  if (!parsed.has(key)) parsed.set(key, (key === "d" ? RAMP_D : RAMP_L).map((hex) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16))));
  const st = parsed.get(key);
  t = Math.min(1, Math.max(0, t)) * (st.length - 1);
  const i = Math.min(st.length - 2, Math.floor(t)), f = t - i, a = st[i], b = st[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}
// Text colour with the higher WCAG contrast against an rgb() background.
function inkOn(rgb) {
  const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [r, g, b] = rgb.match(/\d+/g).map(Number);
  const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return (1.05) / (L + 0.05) > (L + 0.05) / (0.0075 + 0.05) ? "#ffffff" : "#11161c";
}
const KL_LO = -5, KL_HI = 1; // colour scale: 1e-5 .. 10 nats, fixed so experiments compare
const klT = (kl) => (kl <= 0 ? 0 : (Math.log10(kl) - KL_LO) / (KL_HI - KL_LO));
// Residual relative difference: log scale fitted to the data shown (legend states the range).
function errScale(rows) {
  let mn = Infinity, mx = 0;
  for (const r of rows) for (const v of r) if (v > 0) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (!mx) return { t: () => 0, lo: 0, hi: 0 };
  const hi = Math.ceil(Math.log10(mx) * 2) / 2, lo = Math.max(hi - 5, Math.floor(Math.log10(mn) * 2) / 2);
  return { t: (e) => (e <= 0 ? 0 : (Math.log10(e) - lo) / (hi - lo)), lo, hi };
}
const sup = (x) => `1e${x}`;
function rampCss() {
  const st = dark() ? RAMP_D : RAMP_L;
  return `linear-gradient(90deg, ${st.join(",")})`;
}

// ---------- setup form ----------
const ARM_FIELDS = ["ctk", "ctv", "fa", "ub", "threads"];
function readArm(p) {
  return {
    model: $(`#${p}-model`).value.trim(),
    cache_k: $(`#${p}-ctk`).value,
    cache_v: $(`#${p}-ctv`).value,
    flash_attn: $(`#${p}-fa`).value,
    ubatch: +$(`#${p}-ub`).value || 0,
    threads: Math.max(1, +$(`#${p}-threads`).value || 1),
  };
}
function readForm() {
  const a = readArm("a"), b = readArm("b");
  if (!b.model) b.model = a.model;
  const kind = $('input[name="kind"]:checked').value;
  const input = kind === "text"
    ? { kind: "text", text: $("#text").value }
    : { kind: "generate", prompt: $("#text").value, chat: $("#chat").checked, max_new: +$("#max-new").value || 96 };
  return {
    request: {
      a, b, input,
      chunk_tokens: +$("#chunk").value || 256,
      max_chunks: +$("#max-chunks").value || 4,
      floor: $("#floor").checked,
      layers: $("#layers").checked,
      repair: $("#repair").checked || $("#components").checked,
      repair_stride: Math.max(1, +$("#stride").value || 1),
      components: $("#components").checked,
      attribution: $("#attribution").checked,
      top_k: 8,
      skip_first: Math.max(0, +$("#skip").value || 0),
    },
    notes: { title: $("#title").value.trim(), hypothesis: $("#hypothesis").value.trim(), interpretation: "" },
  };
}
function setKind(kind) {
  $(`input[name="kind"][value="${kind}"]`).checked = true;
  $("#gen-opts").hidden = kind !== "generate";
  $("#text-label").textContent = kind === "text" ? "Text both arms will read" : "Prompt both arms will answer";
}
const SAMPLE_TEXT = "Question: A bakery sells muffins for 3 dollars each and cookies for 2 dollars each. On Monday it sold 17 muffins and 23 cookies. How much money did the bakery make?\nAnswer: The muffins brought in 17 * 3 = 51 dollars. The cookies brought in 23 * 2 = 46 dollars. In total, the bakery made 51 + 46 = 97 dollars.\n\nThe capital of Australia is Canberra, not Sydney, although many people assume otherwise.";

function saveForm() {
  try { localStorage.setItem("hd-form", JSON.stringify(readForm())); } catch {}
}
function restoreForm() {
  let f;
  try { f = JSON.parse(localStorage.getItem("hd-form") || "null"); } catch {}
  $("#text").value = SAMPLE_TEXT;
  if (f) fillForm(f.request);
}
// Put a request's settings into the setup rail (so an opened experiment can be re-run as is).
function fillForm(r) {
  for (const p of ["a", "b"]) {
    const arm = r[p];
    // Shared files carry bare file names; map them onto a model this server can see.
    const local = (m) => (S.state && !/[\\/]/.test(m) ? S.state.models.find((x) => x.name === m)?.path : null) ?? m;
    $(`#${p}-model`).value = p === "b" && arm.model === r.a.model ? "" : local(arm.model);
    $(`#${p}-ctk`).value = arm.cache_k; $(`#${p}-ctv`).value = arm.cache_v; $(`#${p}-fa`).value = arm.flash_attn;
    $(`#${p}-ub`).value = arm.ubatch; $(`#${p}-threads`).value = arm.threads;
  }
  setKind(r.input.kind);
  $("#text").value = r.input.text ?? r.input.prompt ?? SAMPLE_TEXT;
  if (r.input.kind === "generate") { $("#chat").checked = r.input.chat; $("#max-new").value = r.input.max_new; }
  $("#chunk").value = r.chunk_tokens; $("#max-chunks").value = r.max_chunks;
  $("#floor").checked = r.floor; $("#layers").checked = r.layers; $("#repair").checked = r.repair;
  $("#stride").value = r.repair_stride; $("#components").checked = !!r.components; $("#attribution").checked = r.attribution;
  $("#skip").value = r.skip_first ?? 0;
}

async function loadState() {
  S.state = await api("/api/state");
  const st = S.state;
  $("#build").textContent = `v${st.version}, llama.cpp ${st.llama_cpp_commit.slice(0, 8)} (${st.build} CPU build)`;
  $("#build").title = st.system_info;
  for (const p of ["a", "b"]) for (const f of ["ctk", "ctv"]) {
    const sel = $(`#${p}-${f}`);
    if (sel.options.length) continue;
    for (const t of st.cache_types) sel.append(h("option", { value: t }, t));
    sel.value = "f16";
  }
  $("#model-list").replaceChildren(...st.models.map((m) => h("option", { value: m.path }, `${m.name} (${(m.bytes / 1e9).toFixed(2)} GB)`)));
  $("#workspace").textContent = `Stored in ${st.workspace}`;
  renderHistory();
}

function renderHistory() {
  const items = S.state.experiments;
  const ul = $("#history");
  if (!items.length) { ul.replaceChildren(h("li", { class: "hint" }, "Nothing saved yet. Run a comparison, then save it here.")); return; }
  ul.replaceChildren(...items.map((x) => h("li", {},
    h("button", { type: "button", "aria-current": S.file === x.file ? "true" : null, onclick: () => openSaved(x.file) },
      x.title || `${x.a} vs ${x.b}`,
      h("span", { class: "meta" }, `${new Date(x.created_unix * 1000).toLocaleString()}, mean KL ${fmtKL(x.mean_kl)}, flips ${pct(x.flip_rate)}`)))));
}

async function openSaved(file) {
  try { showExperiment(await api(`/api/experiment?file=${encodeURIComponent(file)}`), file); }
  catch (e) { renderError(e.message); }
}

// ---------- running ----------
async function startRun(ev) {
  ev.preventDefault();
  if (S.busy) return;
  const body = readForm();
  if (!body.request.a.model) { renderError("Choose a model file for arm A."); return; }
  if (body.request.input.kind === "text" && !body.request.input.text.trim()) { renderError("Paste some text for both arms to read."); return; }
  saveForm();
  try { const { job } = await api("/api/run", body); poll(job); }
  catch (e) { renderError(e.message); }
}

function setBusy(b) { S.busy = b; $("#run").disabled = b; }

async function poll(id) {
  setBusy(true);
  let j;
  try { j = await api(`/api/job?id=${id}`); } catch (e) { setBusy(false); renderError(e.message); return; }
  if (j.status === "running") { renderProgress(j, id); setTimeout(() => poll(id), 600); return; }
  setBusy(false);
  if (j.status === "error") renderError(j.error);
  else if (j.status === "cancelled") renderEmpty("Cancelled. Nothing was saved.");
  else if (j.kind === "run") { showExperiment(j.result.experiment, j.result.file); loadState().catch(() => {}); }
  else showVerify(j.result);
}

function announce(msg) {
  const live = $("#live");
  if (live.textContent !== msg) live.textContent = msg;
}

function renderProgress(j, id) {
  announce(j.stage);
  const frac = j.total ? j.done / j.total : 0;
  const box = $(".progress") || h("div", { class: "progress" });
  box.replaceChildren(
    h("h2", {}, j.kind === "verify" ? "Re-running the saved experiment" : "Running both arms"),
    h("div", { class: "meter", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": j.total || 1, "aria-valuenow": j.done }, h("div", { style: `width:${(100 * frac).toFixed(1)}%` })),
    h("p", { class: "status" }, j.total ? `${j.stage}: pass ${j.done} of ${j.total}` : j.stage),
    h("button", { class: "quiet", type: "button", onclick: () => api(`/api/cancel?id=${id}`, {}) }, "Cancel"));
  if (!box.isConnected) main.replaceChildren(box);
}

function renderError(msg) {
  main.replaceChildren(h("h2", {}, "The run stopped"), h("div", { class: "error" }, msg),
    h("p", { class: "hint", style: "margin-top:.6rem" }, "Check the file paths and settings in the left panel, then run again."));
}

function renderEmpty(note) {
  const recipe = (title, desc, apply) => h("button", { class: "recipe", type: "button", onclick: () => { apply(); $("#a-model").focus(); } }, h("strong", {}, title), h("span", {}, desc));
  const same = () => { $("#b-model").value = ""; for (const f of ARM_FIELDS) $(`#b-${f}`).value = $(`#a-${f}`).value; };
  main.replaceChildren(h("div", { class: "intro" },
    note && h("p", { class: "status" }, note),
    h("h1", {}, "Find where two runs of a model stop agreeing"),
    h("p", {}, "Arm A and arm B read the same tokens in llama.cpp. heterodyne compares their next-token distributions at every position, maps where the difference appears inside the network, and can move activations from one arm into the other to test which layers carry it."),
    h("p", {}, "Pick a starting point, choose your model files on the left, and run."),
    h("div", { class: "recipes" },
      recipe("Is my quant faithful?", "Put the full-precision GGUF in A and the quantized one in B.", () => { same(); setKind("text"); $("#repair").checked = true; $("#b-model").focus(); }),
      recipe("Does a quantized KV cache matter?", "Same file twice; B stores K and V as q8_0. Attribution splits the two.", () => { same(); $("#b-ctk").value = "q8_0"; $("#b-ctv").value = "q8_0"; $("#attribution").checked = true; }),
      recipe("Is the batch size changing outputs?", "Same file twice; B splits each chunk into ubatches of 64.", () => { same(); $("#b-ub").value = 64; }),
      recipe("Where do the answers fork?", "Both arms answer a prompt greedily; see the first token where they disagree.", () => { setKind("generate"); $("#text").value = "A train leaves at 14:35 and the trip takes 2 hours and 50 minutes. When does it arrive? Answer briefly."; $("#repair").checked = true; })),
    h("p", { class: "hint" }, "Everything runs on this machine on the CPU. Nothing is uploaded.")));
}

// ---------- experiment view ----------
function showExperiment(e, file) {
  if (!e || e.schema !== "heterodyne/experiment@1") { renderError("That file is not a heterodyne experiment (schema heterodyne/experiment@1)."); return; }
  S.exp = e; S.file = file; S.chunk = 0; S.dirty = false;
  fillForm(e.request);
  history.replaceState(null, "", file ? `#file=${encodeURIComponent(file)}` : location.pathname);
  S.sel = argmaxKL(e.results.chunks[0]);
  if (S.state) renderHistory();
  render();
}
// Largest divergence among scored positions (the first skip_first are shown but not counted).
function argmaxKL(c) {
  const k = Math.min(skipN(), c.pos.length - 1);
  let b = k;
  c.pos.forEach((p, i) => { if (i >= k && p.kl > c.pos[b].kl) b = i; });
  return b;
}
const skipN = () => (S.exp && S.exp.request.skip_first) || 0;

function render() {
  const e = S.exp, r = e.results, sm = r.summary, req = e.request, pv = e.provenance;
  const ran = (k) => (r.flash_attn_used && req[k].flash_attn === "auto" ? ` (ran ${r.flash_attn_used[k === "a" ? 0 : 1] ? "on" : "off"})` : "");
  const settings = (k) => { const a = req[k]; return `K ${a.cache_k}, V ${a.cache_v}, flash attn ${a.flash_attn}${ran(k)}, ubatch ${a.ubatch || "whole chunk"}, ${a.threads} threads`; };
  const head = h("div", {},
    h("div", { class: "head" }, h("h1", {}, e.notes.title || `${pv.a.file_name} vs ${pv.b.file_name}`)),
    h("div", { class: "arms" },
      h("span", { class: "who" }, h("span", { class: "pen a" }), "A"), h("span", { class: "what" }, h("b", {}, pv.a.file_name), ` ${pv.a.desc}; ${settings("a")}`),
      h("span", { class: "who" }, h("span", { class: "pen b" }), "B"), h("span", { class: "what" }, h("b", {}, pv.b.file_name), ` ${pv.b.desc}; ${settings("b")}`)),
    r.warnings.map((w) => h("div", { class: "warning" }, w)));

  const fl = r.floor;
  const splits = [...new Set(r.chunks.map((c) => c.floor_ubatch).filter(Boolean))].join(", ");
  const nch = sm.n_chunks || r.chunks.length;
  const ciNote = !sm.mean_kl_ci95 ? "one chunk: no interval" : `95% t-interval over ${nch} chunks: ${fmtKL(sm.mean_kl_ci95[0])} to ${fmtKL(sm.mean_kl_ci95[1])}${nch < 5 ? " (few chunks, approximate)" : ""}`;
  const readout = h("dl", { class: "readout" },
    cell("Mean KL(A‖B), nats", fmtKL(sm.mean_kl), ciNote),
    cell("Top-1 changed", pct(sm.flip_rate), `${sm.flips} of ${sm.n_positions}; ${sm.flips_near_tie} near-tie, ${sm.flips_confident} confident`),
    cell("Perplexity A / B", sm.ppl_a != null ? `${sm.ppl_a.toFixed(3)} / ${sm.ppl_b.toFixed(3)}` : "–", "on the scored text"),
    fl && cell("Numerical floor", fmtKL(fl.mean_kl), `A re-run${splits ? ` with ubatch ${splits}` : " with another ubatch split"}; ${pct(fl.flip_rate)} top-1 changed`),
    cell("Max KL", fmtKL(sm.max_kl), `p99 ${fmtKL(sm.p99_kl)}, median ${fmtKL(sm.median_kl)}`),
    cell("Scored", `${sm.n_positions} positions`, `${r.chunks.length} chunk${r.chunks.length > 1 ? "s" : ""}${req.skip_first ? `; first ${req.skip_first} of each not counted` : ""}; ${(e.elapsed_ms / 1000).toFixed(1)} s`));

  const tabs = r.chunks.length > 1 && h("div", { class: "tabs", role: "tablist" }, r.chunks.map((c, i) =>
    h("button", { type: "button", role: "tab", "aria-selected": String(i === S.chunk), onclick: () => { S.chunk = i; S.sel = argmaxKL(c); render(); } }, `Chunk ${i + 1}`)));

  const c = r.chunks[S.chunk];
  const left = h("div", {},
    panel("Divergence per token", `KL(A‖B) at each position, log scale. ${c.floor_kl ? `Grey line: the numerical floor, arm A against itself re-run${c.floor_ubatch ? ` with ubatch ${c.floor_ubatch}` : " with a different ubatch"}. ` : ""}Triangles mark positions where B's top token differs from A's; hollow ones were near-ties for A.`, stripChart(c)),
    panel("Text, coloured by divergence", "Each token is coloured by how differently the arms predicted it. Click a token, or use the arrow keys, to inspect it.", tokenView(c), legendKL()),
    c.layer_err && (() => {
      const es = errScale(c.layer_err);
      return panel("Where it appears inside the network", "Relative difference of the residual stream ‖h_B − h_A‖ / ‖h_A‖ after each layer (rows, first layer at the bottom) at each position (columns). This is correlation, not cause: depth amplifies small differences.", heat(c.layer_err, es.t, (l, i, v) => `layer ${l}, pos ${i}\nrel. diff ${v.toExponential(2)}`), legendErr(es));
    })(),
    c.repair_denoise && panel("Restoring A's activations, layer by layer", "Intervention. Row l: B runs with A's residual stream written in after layer l; colour is the divergence from A that remains. A row that goes light means everything that differs was already in the residual stream by that depth; what is left comes from later layers.", heat(c.repair_denoise, klT, (l, i, v) => `B + A's residual after layer ${l}, pos ${i}\nKL left ${fmtKL(v)}`, c.repair_layers), legendKL()),
    c.repair_noise && panel("Injecting B's activations, layer by layer", "Intervention. Row l: A runs with B's residual stream written in after layer l; colour is how far that pushes A away from itself. Dark rows mark depths where B's difference, carried forward through A's own later layers, already reaches the output.", heat(c.repair_noise, klT, (l, i, v) => `A + B's residual after layer ${l}, pos ${i}\nKL ${fmtKL(v)}`, c.repair_layers), legendKL()),
    r.attribution && attributionPanel(r.attribution),
    r.generation && generationPanel(r.generation, c));
  const work = h("div", { class: "work" }, left, inspector(c));
  const keep = $(".tokens")?.scrollTop;
  main.replaceChildren(head, readout, tabs || "", work, notesPanel(), provenancePanel());
  if (keep != null) $(".tokens").scrollTop = keep;
}

function cell(dt, dd, small) {
  return h("div", {}, h("dt", {}, dt), h("dd", {}, dd, small && h("small", {}, small)));
}
function panel(title, explain, ...body) {
  return h("section", { class: "panel" }, h("header", {}, h("h2", {}, title)), h("p", { class: "explain" }, explain), ...body);
}
function legendKL() {
  return h("div", { class: "legend" }, h("span", { class: "key" }, "KL", h("span", { class: "ramp", style: `background:${rampCss()}` }), "10⁻⁵ to 10 nats"));
}
function legendErr(es) {
  const range = es.hi === es.lo ? "all zero" : `${sup(es.lo)} to ${sup(es.hi)} (fitted to this chunk)`;
  return h("div", { class: "legend" }, h("span", { class: "key" }, "relative difference", h("span", { class: "ramp", style: `background:${rampCss()}` }), range));
}

const M = { l: 46, r: 10, t: 14, b: 22 }; // shared plot margins so all position axes line up

// Width of the left results column, so SVG and canvas plots draw at 1:1 scale.
function plotWidth() {
  const cs = getComputedStyle(main);
  const inner = main.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  return Math.max(300, (innerWidth > 1100 ? inner - 320 - 24 : inner) - 2);
}

function select(i) {
  const c = S.exp.results.chunks[S.chunk];
  S.sel = Math.max(0, Math.min(c.pos.length - 1, i));
  if (S.file) history.replaceState(null, "", `#file=${encodeURIComponent(S.file)}&chunk=${S.chunk}&pos=${S.sel}`);
  render();
  const p = c.pos[S.sel];
  announce(`Position ${S.sel}, KL ${fmtKL(p.kl)} nats${p.top_a !== p.top_b ? ", top token changed" : ""}`);
  const t = $(".tokens span.sel");
  if (t) t.scrollIntoView({ block: "nearest" });
}

function tipAt(ev, text) {
  const t = $("#tip");
  if (!text) { t.hidden = true; return; }
  t.textContent = text; t.hidden = false;
  t.style.left = Math.min(innerWidth - t.offsetWidth - 8, ev.clientX + 12) + "px";
  t.style.top = ev.clientY + 14 + "px";
}

function stripChart(c) {
  const W = plotWidth(), H = 170;
  const n = c.pos.length, pw = W - M.l - M.r, step = pw / n;
  const maxKL = Math.max(1e-6, ...c.pos.map((p) => p.kl));
  const lo = -6, hi = Math.max(0, Math.ceil(Math.log10(maxKL)));
  const y = (kl) => M.t + (H - M.t - M.b) * (1 - (Math.log10(Math.max(kl, 10 ** lo)) - lo) / (hi - lo));
  const x = (i) => M.l + (i + 0.5) * step;
  const svg = s("svg", { class: "plot", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "KL divergence per position" });
  const grid = s("g", { class: "grid axis" });
  for (let d = lo; d <= hi; d++) {
    grid.append(s("line", { x1: M.l, x2: W - M.r, y1: y(10 ** d), y2: y(10 ** d) }), s("text", { x: M.l - 6, y: y(10 ** d) + 3.5, "text-anchor": "end" }, `1e${d}`));
  }
  for (let i = 0; i < n; i += Math.max(1, Math.round(n / 8))) grid.append(s("text", { x: x(i), y: H - 6, "text-anchor": "middle" }, i));
  svg.append(grid);
  const k = Math.min(skipN(), n);
  if (k) svg.append(s("rect", { x: M.l, y: M.t - 10, width: k * step, height: H - M.b - M.t + 10, fill: "var(--grid)", opacity: 0.45 }),
    s("text", { x: M.l + 3, y: H - M.b - 4, class: "axis-label" }, "not counted"));
  const bars = s("g");
  c.pos.forEach((p, i) => {
    if (p.kl <= 0) return;
    bars.append(s("rect", { x: x(i) - Math.max(0.5, step * 0.4), width: Math.max(1, step * 0.8), y: y(p.kl), height: Math.max(0.5, y(10 ** lo) - y(p.kl)), fill: ramp(0.25 + 0.75 * klT(p.kl)), opacity: i < k ? 0.35 : null }));
  });
  svg.append(bars);
  if (c.floor_kl) {
    svg.append(s("polyline", { points: c.floor_kl.map((k, i) => `${x(i)},${y(k)}`).join(" "), fill: "none", stroke: "var(--floor)", "stroke-width": 1.2 }));
  }
  c.pos.forEach((p, i) => {
    if (p.top_a === p.top_b) return;
    const tie = gap(p) < 0.25;
    svg.append(s("path", { d: `M${x(i) - 3.5},${M.t - 9} L${x(i) + 3.5},${M.t - 9} L${x(i)},${M.t - 3} Z`, fill: tie ? "none" : "var(--b)", stroke: "var(--b)", "stroke-width": 1 }));
  });
  svg.append(s("line", { x1: x(S.sel), x2: x(S.sel), y1: M.t - 10, y2: H - M.b, stroke: "var(--ink)", "stroke-width": 1.2 }));
  const hit = (ev) => {
    const r = svg.getBoundingClientRect();
    return Math.floor(((ev.clientX - r.left) * (W / r.width) - M.l) / step);
  };
  svg.addEventListener("mousemove", (ev) => {
    const i = hit(ev);
    if (i < 0 || i >= n) return tipAt(ev, null);
    const p = c.pos[i], next = c.tokens[i + 1];
    tipAt(ev, `pos ${i} → ${next != null ? vis(tok(next)) : "(after the text)"}\nKL ${fmtKL(p.kl)}${p.top_a !== p.top_b ? `\nA: ${vis(tok(p.top_a))}  B: ${vis(tok(p.top_b))}` : ""}`);
  });
  svg.addEventListener("mouseleave", (ev) => tipAt(ev, null));
  svg.addEventListener("click", (ev) => { const i = hit(ev); if (i >= 0 && i < n) select(i); });
  return svg;
}

function tokenView(c) {
  // Token i+1 is what position i predicted; the first token has no prediction.
  const box = h("div", { class: "tokens", tabindex: 0, role: "listbox", "aria-label": "Tokens; arrow keys move the selection" });
  box.append(h("span", { class: "first" }, tok(c.tokens[0])));
  for (let i = 0; i < c.pos.length; i++) {
    const p = c.pos[i];
    const has = i + 1 < c.tokens.length;
    const txt = has ? tok(c.tokens[i + 1]) : " ⟂";
    const cls = [p.top_a !== p.top_b && "flip", p.top_a !== p.top_b && gap(p) < 0.25 && "tie", i === S.sel && "sel", i < skipN() && "unscored"].filter(Boolean).join(" ");
    const bg = ramp(klT(p.kl));
    const span = h("span", { class: cls, "data-i": i, style: `background:${bg};color:${inkOn(bg)}`, title: has ? null : "prediction after the last token", role: "option", "aria-selected": String(i === S.sel) });
    const parts = txt.split("\n");
    parts.forEach((part, k) => { if (k) span.append(h("span", { class: "nl" }, "↵"), "\n"); span.append(part); });
    box.append(span);
  }
  box.addEventListener("click", (ev) => { const t = ev.target.closest("[data-i]"); if (t) select(+t.dataset.i); });
  box.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowRight") { ev.preventDefault(); select(S.sel + 1); $(".tokens").focus(); }
    if (ev.key === "ArrowLeft") { ev.preventDefault(); select(S.sel - 1); $(".tokens").focus(); }
  });
  return box;
}

function heat(rows, toT, label, rowNames) {
  // rows[r][pos]; row 0 drawn at the bottom.
  const W = plotWidth();
  const nr = rows.length, n = rows[0].length;
  const rh = Math.max(4, Math.min(9, Math.floor(240 / nr)));
  const H = M.t + nr * rh + M.b;
  const dpr = devicePixelRatio || 1;
  const cv = h("canvas", { class: "plot", width: Math.round(W * dpr), height: Math.round(H * dpr), style: `height:${H}px`, role: "img", "aria-label": "heatmap" });
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const pw = W - M.l - M.r, step = pw / n;
  for (let r = 0; r < nr; r++) {
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = ramp(toT(rows[r][i]));
      ctx.fillRect(M.l + i * step, M.t + (nr - 1 - r) * rh, Math.ceil(step), rh);
    }
  }
  const cs = getComputedStyle(document.body);
  ctx.fillStyle = cs.getPropertyValue("--muted");
  ctx.font = "10.5px Archivo, sans-serif";
  ctx.textAlign = "right";
  const names = rowNames || rows.map((_, r) => r);
  for (let r = 0; r < nr; r += Math.max(1, Math.round(nr / 6))) ctx.fillText(`L${names[r]}`, M.l - 6, M.t + (nr - 1 - r) * rh + rh - 1);
  ctx.textAlign = "center";
  for (let i = 0; i < n; i += Math.max(1, Math.round(n / 8))) ctx.fillText(i, M.l + (i + 0.5) * step, H - 6);
  ctx.strokeStyle = cs.getPropertyValue("--ink");
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(M.l + (S.sel + 0.5) * step, M.t - 4); ctx.lineTo(M.l + (S.sel + 0.5) * step, M.t + nr * rh + 2); ctx.stroke();
  const at = (ev) => {
    const b = cv.getBoundingClientRect();
    const i = Math.floor(((ev.clientX - b.left) * (W / b.width) - M.l) / step);
    const r = nr - 1 - Math.floor(((ev.clientY - b.top) * (H / b.height) - M.t) / rh);
    return i >= 0 && i < n && r >= 0 && r < nr ? [r, i] : null;
  };
  cv.addEventListener("mousemove", (ev) => { const p = at(ev); tipAt(ev, p && label(names[p[0]], p[1], rows[p[0]][p[1]])); });
  cv.addEventListener("mouseleave", (ev) => tipAt(ev, null));
  cv.addEventListener("click", (ev) => { const p = at(ev); if (p) select(p[1]); });
  return cv;
}

function distList(list, color, target) {
  return list.map(([id, lp]) => {
    const p = Math.exp(lp);
    return h("div", { class: "bar-row" + (id === target ? " target" : "") },
      h("span", { class: "tok", title: JSON.stringify(tok(id)) }, h("i", { style: `width:${(100 * p).toFixed(1)}%;background:${color}` }), h("span", {}, vis(tok(id)))),
      h("span", { class: "p" }, p >= 0.001 ? (100 * p).toFixed(1) + "%" : "<0.1%"));
  });
}

function repairCurve(c, i) {
  if (!c.repair_denoise) return null;
  const L = c.repair_layers, W = 288, H = 150, m = { l: 40, r: 8, t: 10, b: 22 };
  const den = L.map((_, k) => c.repair_denoise[k][i]), noi = L.map((_, k) => c.repair_noise[k][i]);
  const base = c.pos[i].kl, floor = c.floor_kl ? c.floor_kl[i] : null;
  const all = [...den, ...noi, base, floor ?? 0].filter((v) => v > 0);
  const hi = Math.ceil(Math.log10(Math.max(1e-6, ...all))), lo = Math.min(hi - 2, Math.floor(Math.log10(Math.min(...all, 1))) );
  const lastL = L[L.length - 1] || 1;
  const x = (l) => m.l + (W - m.l - m.r) * (l / lastL);
  const y = (v) => m.t + (H - m.t - m.b) * (1 - (Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo));
  const svg = s("svg", { class: "plot", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "transplant curve for this token" });
  const g = s("g", { class: "grid axis" });
  for (let d = lo; d <= hi; d++) g.append(s("line", { x1: m.l, x2: W - m.r, y1: y(10 ** d), y2: y(10 ** d) }), s("text", { x: m.l - 5, y: y(10 ** d) + 3.5, "text-anchor": "end" }, `1e${d}`));
  for (const l of [0, Math.round(lastL / 2), lastL]) g.append(s("text", { x: x(l), y: H - 6, "text-anchor": "middle" }, `L${l}`));
  svg.append(g);
  svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y(base), y2: y(base), stroke: "var(--ink)", "stroke-dasharray": "4 3" }));
  if (floor != null) svg.append(s("line", { x1: m.l, x2: W - m.r, y1: y(floor), y2: y(floor), stroke: "var(--floor)" }));
  const line = (vals, col) => s("polyline", { points: vals.map((v, k) => `${x(L[k])},${y(v)}`).join(" "), fill: "none", stroke: col, "stroke-width": 1.8 });
  svg.append(line(den, "var(--a)"), line(noi, "var(--b)"));
  return h("div", {}, svg, h("div", { class: "legend" },
    h("span", { class: "key" }, h("span", { class: "sw", style: "background:var(--a)" }), "B with A's residual after layer"),
    h("span", { class: "key" }, h("span", { class: "sw", style: "background:var(--b)" }), "A with B's residual after layer"),
    h("span", { class: "key" }, h("span", { class: "sw", style: "background:var(--ink)" }), "untouched KL"),
    floor != null && h("span", { class: "key" }, h("span", { class: "sw", style: "background:var(--floor)" }), "floor")));
}

function inspector(c) {
  const i = S.sel, p = c.pos[i], next = c.tokens[i + 1];
  const ctx = c.tokens.slice(Math.max(0, i - 14), i + 1).map(tok).join("");
  const flip = p.top_a !== p.top_b;
  const g = gap(p), tie = flip && g < 0.25, confident = flip && g >= 1;
  const floor = c.floor_kl ? c.floor_kl[i] : null;
  let verdict;
  if (p.kl === 0) verdict = "Both arms produced exactly the same distribution here.";
  else if (tie) verdict = `B picks a different top token, one that A rated within ${g.toFixed(2)} nats of its own choice (a near-tie by heterodyne's 0.25-nat convention). One flip like this says little on its own.`;
  else if (confident) verdict = `B's top token is one A rated ${g.toFixed(2)} nats below its own choice (confident by the 1-nat convention). Worth checking on other inputs.`;
  else if (flip) verdict = `B picks a different top token; A preferred its own by ${g.toFixed(2)} nats.`;
  else if (floor != null && floor > 0 && p.kl <= 2 * floor) verdict = "Same top token, and the divergence is within twice the numerical floor at this position.";
  else verdict = "Same top token; the arms differ in how the remaining probability is spread.";
  const lastErr = c.layer_err ? c.layer_err[c.layer_err.length - 1][i] : null;
  return h("aside", { class: "inspector", "aria-label": "Selected position" },
    h("h3", {}, `Position ${i}`),
    h("div", { class: "ctx" }, vis(ctx.slice(-120)), next != null ? h("mark", {}, vis(tok(next))) : h("mark", {}, "after the text")),
    h("div", { class: "dist" },
      h("div", {}, h("h4", {}, h("span", { class: "pen a" }), "A predicts"), distList(p.topk_a, "var(--a-soft)", next)),
      h("div", {}, h("h4", {}, h("span", { class: "pen b" }), "B predicts"), distList(p.topk_b, "var(--b-soft)", next))),
    h("dl", { class: "facts" },
      h("dt", {}, "KL(A‖B)"), h("dd", {}, `${fmtKL(p.kl)} nats`),
      floor != null && [h("dt", {}, "floor here"), h("dd", {}, `${fmtKL(floor)} nats`)],
      h("dt", {}, "A's top-2 margin"), h("dd", {}, `${p.margin_a.toFixed(3)} nats`),
      flip && [h("dt", {}, "A: own vs B's"), h("dd", {}, `${g.toFixed(3)} nats${tie ? " (near-tie)" : confident ? " (confident)" : ""}`)],
      h("dt", {}, "A's entropy"), h("dd", {}, `${p.entropy_a.toFixed(3)} nats`),
      p.lp_next_a != null && [h("dt", {}, "p(actual next)"), h("dd", {}, `A ${(100 * Math.exp(p.lp_next_a)).toFixed(2)}%, B ${(100 * Math.exp(p.lp_next_b)).toFixed(2)}%`)],
      lastErr != null && [h("dt", {}, "last-layer diff"), h("dd", {}, lastErr.toExponential(2))]),
    h("p", { class: "verdict" }, verdict),
    c.repair_denoise && h("div", {}, h("h3", { style: "margin-top:.8rem" }, "Transplants at this position"), repairCurve(c, i),
      h("p", { class: "hint" }, "Each point is a separate run on this text. A curve that falls early means the divergence was already in the residual stream at that depth. It holds for this input only.")));
}

function attributionPanel(at) {
  const full = at.cells[at.cells.length - 1];
  const max = Math.max(...at.cells.map((c) => c.mean_kl), 1e-12);
  const minimal = at.cells.filter((c) => !c.error && c.mean_kl >= 0.9 * full.mean_kl && c.switched.length).sort((a, b) => a.switched.length - b.switched.length || b.mean_kl - a.mean_kl)[0];
  const table = h("table", { class: "attr" },
    h("thead", {}, h("tr", {}, at.factors.map((f) => h("th", {}, f.replace("_", " "))), h("th", {}, "mean KL"), h("th", {}, ""), h("th", {}, "top-1 changed"))),
    h("tbody", {}, at.cells.map((c) => h("tr", { class: c === full ? "full" : null },
      at.factors.map((f) => h("td", { class: "c" }, c.switched.includes(f) ? "B" : "A")),
      c.error
        ? h("td", { colspan: 3, class: "hint" }, `llama.cpp rejected this combination: ${c.error}`)
        : [h("td", {}, fmtKL(c.mean_kl)), h("td", { class: "barcell" }, h("i", { style: `width:${(100 * c.mean_kl / max).toFixed(1)}%` })), h("td", {}, pct(c.flip_rate))]))));
  return panel("Which setting causes it", "Every combination of the settings that differ, each run on the same tokens and compared with A. A column shows whose value that setting took.",
    table,
    minimal && h("p", { class: "status", style: "margin-top:.5rem" }, minimal.mean_kl > full.mean_kl
      ? `Switching only ${minimal.switched.join(" and ")} already diverges as much as switching everything (${(minimal.mean_kl / full.mean_kl).toFixed(2)}× the full divergence): the settings do not add.`
      : `Switching only ${minimal.switched.join(" and ")} accounts for ${(100 * minimal.mean_kl / full.mean_kl).toFixed(0)}% of the full divergence on this text.`));
}

function generationPanel(g, c) {
  const col = (cls, label, toks, other) => h("div", { class: cls }, h("h3", {}, label),
    h("pre", {}, toks.map((t, k) => (k === g.fork ? h("mark", { title: other[k] != null ? `other arm: ${JSON.stringify(tok(other[k]))}` : "other arm stopped" }, tok(t)) : tok(t)))));
  const forkAt = g.fork == null ? null : g.prompt_tokens.length + g.fork - 1;
  return panel("Greedy answers",
    g.fork == null ? `Both arms generated the same ${g.a.length} tokens.` : `The arms first disagree at generated token ${g.fork}. The scored text is A's answer, re-read by both arms.${g.fork_gap_a != null ? ` At the fork A preferred its own token over B's by ${g.fork_gap_a.toFixed(3)} nats.` : ""}${g.fork_reproduced === false ? " Re-scoring the whole sequence in one batch does not reproduce both choices: incremental decoding and batched scoring compute slightly different numbers." : ""}`,
    h("div", { class: "gen" }, col("ga", "A answered", g.a, g.b), col("gb", "B answered", g.b, g.a)),
    forkAt != null && forkAt < c.pos.length && h("div", { class: "actions" }, h("button", { class: "quiet", type: "button", onclick: () => select(forkAt) }, "Inspect the fork")));
}

function notesPanel() {
  const e = S.exp;
  const status = h("span", { class: "status" }, S.file ? `Saved as ${S.file}` : "Not saved yet");
  const title = h("input", { type: "text", value: e.notes.title, oninput: (ev) => { e.notes.title = ev.target.value; S.dirty = true; } });
  const hyp = h("textarea", { oninput: (ev) => { e.notes.hypothesis = ev.target.value; S.dirty = true; } }, e.notes.hypothesis);
  const interp = h("textarea", { placeholder: "What do you conclude, and what would change your mind?", oninput: (ev) => { e.notes.interpretation = ev.target.value; S.dirty = true; } }, e.notes.interpretation);
  const save = async () => {
    try {
      const r = await api("/api/save", e);
      S.file = r.file; S.dirty = false;
      await loadState();
      status.textContent = `Saved as ${r.file}`; status.className = "status ok";
    } catch (err) { status.textContent = err.message; status.className = "status bad"; }
  };
  const download = () => {
    // Share file names, not your folder layout. Paths are outside the integrity hash; sha256 pins the files.
    const out = structuredClone(e);
    for (const k of ["a", "b"]) { out.request[k].model = short(out.request[k].model); out.provenance[k].path = short(out.provenance[k].path); }
    const blob = new Blob([JSON.stringify(out)], { type: "application/json" });
    const a = h("a", { href: URL.createObjectURL(blob), download: (S.file || `heterodyne-${e.created_unix}.json`) });
    a.click(); URL.revokeObjectURL(a.href);
  };
  const verify = async () => {
    if (!S.file || S.dirty) await save();
    if (!S.file) return;
    try { const { job } = await api("/api/verify", { file: S.file }); poll(job); }
    catch (err) { status.textContent = err.message; status.className = "status bad"; }
  };
  return h("section", { class: "panel notes" }, h("header", {}, h("h2", {}, "Notes")),
    h("p", { class: "explain" }, "Your words are stored next to the measurements but kept out of the integrity hash: interpretation is not evidence."),
    h("label", {}, "Title"), title,
    h("label", {}, "Hypothesis"), hyp,
    h("label", {}, "Interpretation"), interp,
    h("div", { class: "actions" },
      h("button", { type: "button", onclick: save }, "Save to workspace"),
      h("button", { class: "quiet", type: "button", onclick: download }, "Download JSON"),
      h("button", { class: "quiet", type: "button", onclick: verify, title: "Re-run the saved request and compare every number" }, "Verify by re-running"),
      status));
}

function provenancePanel() {
  const e = S.exp, p = e.provenance;
  const m = (id) => [h("dt", {}, id === p.a ? "A file" : "B file"), h("dd", {}, `${id.path}`), h("dt", {}, "sha256"), h("dd", {}, id.sha256), h("dt", {}, "model"), h("dd", {}, `${id.desc}${id.general_name ? `, ${id.general_name}` : ""}; ${id.n_layer} layers, width ${id.n_embd}, vocab ${id.n_vocab}`)];
  return h("details", { class: "prov" }, h("summary", {}, "Provenance"),
    h("dl", {}, m(p.a), m(p.b),
      h("dt", {}, "llama.cpp"), h("dd", {}, `${p.llama_cpp_commit} (${p.build} build)`),
      h("dt", {}, "heterodyne"), h("dd", {}, p.heterodyne),
      h("dt", {}, "system"), h("dd", {}, `${p.os}/${p.arch}; ${p.system_info}`),
      h("dt", {}, "created"), h("dd", {}, new Date(e.created_unix * 1000).toString()),
      h("dt", {}, "integrity"), h("dd", {}, e.integrity),
      h("dt", {}, "replay"), h("dd", {}, "Re-running on the same machine and build reproduced every number exactly in heterodyne's tests. Other CPUs or llama.cpp builds can take different kernel paths; use Verify to see by how much.")));
}

function showVerify(r) {
  if (!S.exp) return renderEmpty();
  render();
  const ok = r.outcome === "exact" && r.integrity_ok;
  const box = h("div", { class: ok ? "warning" : "error", style: ok ? "border-left-color:var(--ok)" : null },
    h("strong", {}, ok ? "Verified: re-running reproduced every number exactly." : `Verification: ${r.outcome}${r.integrity_ok ? "" : ", integrity check failed"}`),
    !ok && h("p", {}, `${r.top1_changes} positions changed their top token. Largest absolute difference: ${Object.entries(r.max_abs_diff || { kl: r.max_abs_kl_diff }).map(([k, v]) => `${k} ${fmtKL(v)}`).join(", ")}.`),
    r.notes.map((n) => h("p", {}, n)));
  main.prepend(box);
}

// ---------- wire up ----------
$("#setup").addEventListener("submit", startRun);
for (const r of document.querySelectorAll('input[name="kind"]')) r.addEventListener("change", () => setKind(r.value));
$("#import").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  try { showExperiment(JSON.parse(await f.text()), null); }
  catch (e) { renderError(`Could not read ${f.name}: ${e.message}`); }
});
let rt;
addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => S.exp && render(), 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => S.exp && render());

(async () => {
  try { await loadState(); } catch (e) { renderError(`Cannot reach the heterodyne server: ${e.message}`); return; }
  restoreForm();
  // Reattach to a run still going on the server (page reloaded or reopened mid-run).
  if (S.state.job != null) {
    const j = await api(`/api/job?id=${S.state.job}`).catch(() => null);
    if (j && j.status === "running") { poll(S.state.job); return; }
  }
  const q = new URLSearchParams(location.hash.slice(1));
  const int = (k) => (Number.isInteger(+q.get(k)) && q.get(k) !== "" ? +q.get(k) : null);
  if (q.get("file")) {
    await openSaved(q.get("file"));
    if (S.exp && int("chunk") != null) S.chunk = Math.max(0, Math.min(int("chunk"), S.exp.results.chunks.length - 1));
    if (S.exp && int("pos") != null) select(int("pos"));
  } else renderEmpty();
})();
