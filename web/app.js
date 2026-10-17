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
