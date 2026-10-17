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
