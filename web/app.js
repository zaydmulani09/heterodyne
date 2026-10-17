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
      repair: $("#repair").checked,
      repair_stride: Math.max(1, +$("#stride").value || 1),
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
  $("#stride").value = r.repair_stride; $("#attribution").checked = r.attribution;
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
