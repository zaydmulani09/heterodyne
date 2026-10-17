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
