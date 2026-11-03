"""HTML report of scripts/pitch_signature_test.py: one self-contained page (data, charts and code inline).

    python scripts/pitch_signature_view.py artifacts/pitch_signature/results.json      # rewrite report.html

``write_report(data, path)`` writes a full HTML document; ``standalone=False`` leaves out the document skeleton
(doctype, html, head, body) for hosts that add their own.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

KEEP_META = ("hole", "size_nm", "pitch_nm", "layout", "pitch_px", "size_px", "blur_nm", "lens_nm", "photons_per_level",
             "read_noise", "cd_sigma_nm", "placement_sigma_nm", "brightness_sigma", "row_shift_nm", "row_shift_every",
             "holes_inside", "seed")
KEEP_METHOD = ("name", "group", "found", "failed", "center_sd_y", "center_sd_x", "phase_error", "phase_curve", "spike",
               "need_seen", "need_method", "noise_seen", "noise_method", "seconds", "note", "needs_blur")


def _clean(value, digits=4):
    """JSON-safe copy: floats rounded, NaN / inf -> None."""
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    return value


def compact(data):
    """The numbers the page needs: per image the methods in a list, curves at 3 decimals, the true centers' curve
    as the reference."""
    images = []
    for image in data["images"]:
        methods = image["methods"]
        truth = methods.get("truth", {})
        keys = list(truth.get("noise_seen", {}).keys())
        entries = []
        for key, method in methods.items():
            entry = {k: method.get(k) for k in KEEP_METHOD}
            entry["key"] = key
            entry["profiles"] = _clean(method.get("profiles", {}), 3)
            entries.append(entry)
        images.append(dict(
            key=image["key"], label=image["label"], meta={k: image["meta"].get(k) for k in KEEP_META},
            limit_nm=image.get("limit_nm"), crop_png=image.get("crop_png"), evaluated_holes=image["evaluated_holes"],
            pairs=image["pairs"], seconds=image.get("seconds"), best_gpu=image.get("best_gpu"),
            best_cd=image.get("best_cd"), best_new=image.get("best_new"), keys=keys,
            columns_total=truth.get("columns_total"), row_intervals=truth.get("row_intervals"),
            planted=truth.get("planted", []), methods=entries))
    return _clean(dict(settings=data["settings"], images=images))


def write_report(data, path, *, standalone=True):
    """Write the report page to ``path``; returns the path."""
    payload = json.dumps(compact(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    body = PAGE.replace("__DATA__", payload)
    if standalone:
        body = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + body.replace("<!--/head-->", "</head>\n<body>") + "\n</body>\n</html>\n")
    else:
        body = body.replace("<!--/head-->", "")
    path = Path(path)
    path.write_text(body, encoding="utf-8")
    return path


PAGE = r"""<title>Pitch Signature Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  /* Layout: one reading column (72rem) of stacked sections; the explorer is image tabs over stacked panels;
     wide tables scroll inside their own frames. Palette: cool instrument greys, blue = the method looked at,
     orange = the planted step, grey = true centers. */
  --bg: #f3f5f7;
  --surface: #fcfdfd;
  --ink: #0f1419;
  --ink-2: #46505b;
  --muted: #76808a;
  --line: #e1e5e9;
  --axis: #c3c9d0;
  --accent: #2a78d6;
  --ref: #eb6834;
  --truth: #7f8892;
  --ctx: #d8dde2;
  --chip: #e7ebef;
  --chip-on: #0f1419;
  --chip-on-ink: #fcfdfd;
  --focus: #2a78d6;
  --s1: #cde2fb; --s1-ink: #0b0b0b;
  --s2: #9ec5f4; --s2-ink: #0b0b0b;
  --s3: #6da7ec; --s3-ink: #0b0b0b;
  --s4: #3987e5; --s4-ink: #ffffff;
  --s5: #256abf; --s5-ink: #ffffff;
  --s6: #184f95; --s6-ink: #ffffff;
  --never: #e4e7ea; --never-ink: #46505b;
  --font-ui: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
  --step--1: 0.8125rem;
  --step-0: 0.9375rem;
  --step-1: 1.125rem;
  --step-2: 1.5rem;
  --step-3: 2.25rem;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1113; --surface: #161a1d; --ink: #eef1f4; --ink-2: #c1c8cf; --muted: #8c959e; --line: #2a3035;
    --axis: #3c434a; --accent: #3987e5; --ref: #d95926; --truth: #a0a8b0; --ctx: #2f353b; --chip: #232930;
    --chip-on: #eef1f4; --chip-on-ink: #0e1113; --focus: #6da7ec;
    --s1: #104281; --s1-ink: #ffffff; --s2: #184f95; --s2-ink: #ffffff; --s3: #256abf; --s3-ink: #ffffff;
    --s4: #3987e5; --s4-ink: #ffffff; --s5: #6da7ec; --s5-ink: #0b0b0b; --s6: #9ec5f4; --s6-ink: #0b0b0b;
    --never: #262c31; --never-ink: #c1c8cf;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1113; --surface: #161a1d; --ink: #eef1f4; --ink-2: #c1c8cf; --muted: #8c959e; --line: #2a3035;
  --axis: #3c434a; --accent: #3987e5; --ref: #d95926; --truth: #a0a8b0; --ctx: #2f353b; --chip: #232930;
  --chip-on: #eef1f4; --chip-on-ink: #0e1113; --focus: #6da7ec;
  --s1: #104281; --s1-ink: #ffffff; --s2: #184f95; --s2-ink: #ffffff; --s3: #256abf; --s3-ink: #ffffff;
  --s4: #3987e5; --s4-ink: #ffffff; --s5: #6da7ec; --s5-ink: #0b0b0b; --s6: #9ec5f4; --s6-ink: #0b0b0b;
  --never: #262c31; --never-ink: #c1c8cf;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink); font: 400 var(--step-0)/1.55 var(--font-ui);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
}
.page { max-width: 72rem; margin: 0 auto; padding-block: 2.5rem 4rem; display: grid; gap: 3rem; }
h1, h2, h3 { font-family: var(--font-head); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.2; }
h1 { font-size: var(--step-3); letter-spacing: -0.01em; }
h2 { font-size: var(--step-2); }
h3 { font-size: var(--step-1); }
p { margin: 0; max-width: 68ch; }
.eyebrow { font: 500 var(--step--1)/1.3 var(--font-mono); color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
.masthead { display: grid; gap: 0.75rem; }
.lede { font-size: var(--step-1); color: var(--ink-2); max-width: 60ch; }
.facts { display: flex; flex-wrap: wrap; gap: 0.5rem 1.75rem; margin: 0.5rem 0 0; font-size: var(--step--1); }
.facts div { display: grid; gap: 0.1rem; }
.facts dt { color: var(--muted); }
.facts dd { margin: 0; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
section { display: grid; gap: 1rem; min-width: 0; }
.section-head { display: grid; gap: 0.35rem; }
.section-head p { color: var(--ink-2); }
.answer { display: grid; gap: 0.6rem; }
.answer p { font-size: var(--step-1); }
.answer strong { font-weight: 600; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; min-width: 0; }
figure { margin: 0; display: grid; gap: 0.6rem; min-width: 0; }
figcaption { color: var(--ink-2); font-size: var(--step--1); max-width: 80ch; }
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; font-size: var(--step--1); color: var(--ink-2); align-items: center; }
.key { display: inline-flex; align-items: center; gap: 0.4rem; }
.key i { display: inline-block; width: 18px; height: 0; border-top: 2px solid; }
.key i.method { border-color: var(--accent); }
.key i.truth { border-color: var(--truth); }
.key i.ref { border-color: var(--ref); border-top-width: 1.5px; }
.key i.ctx { border-color: var(--ctx); border-top-width: 1px; }
.key i.limit { border-color: var(--muted); border-top-width: 1px; }
.key b { display: inline-block; width: 14px; height: 14px; border-radius: 3px; }
.featured-grid { display: grid; gap: 0.25rem; }
.featured-row { display: grid; grid-template-columns: minmax(8rem, 13rem) 1fr; gap: 0.75rem; align-items: center; }
.featured-row .who { display: grid; gap: 0.1rem; font-size: var(--step--1); }
.featured-row .who strong { font-size: var(--step-0); font-weight: 600; }
.featured-row .who span { color: var(--ink-2); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; min-width: 0; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: var(--step--1); }
th, td { padding: 0.45rem 0.6rem; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { font-weight: 500; color: var(--ink-2); background: var(--surface); position: sticky; top: 0; }
th.num { text-align: right; }
td.num { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
#table td:first-child, #best td:first-child { min-width: 15rem; }
tbody tr:last-child td { border-bottom: 0; }
tr.pick { cursor: pointer; }
tr.pick:hover td { background: color-mix(in srgb, var(--accent) 6%, transparent); }
tr.pick.on td { background: color-mix(in srgb, var(--accent) 13%, transparent); }
tr.pick:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; }
tr.weak td { color: var(--muted); }
.tag { display: inline-block; font: 500 0.6875rem/1 var(--font-mono); letter-spacing: 0.04em; text-transform: uppercase;
  padding: 0.2rem 0.35rem; border-radius: 4px; background: var(--chip); color: var(--ink-2); margin-left: 0.35rem; vertical-align: 1px; }
.best-cell { display: grid; gap: 0.1rem; }
.best-cell span { color: var(--ink-2); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.matrix td.cell { text-align: center; font-family: var(--font-mono); font-variant-numeric: tabular-nums; min-width: 4.2rem;
  border-left: 2px solid var(--surface); white-space: nowrap; }
.matrix th.img { text-align: center; font-size: 0.75rem; line-height: 1.3; min-width: 5.5rem; }
.matrix td.who { white-space: nowrap; }
.c1 { background: var(--s1); color: var(--s1-ink); } .c2 { background: var(--s2); color: var(--s2-ink); }
.c3 { background: var(--s3); color: var(--s3-ink); } .c4 { background: var(--s4); color: var(--s4-ink); }
.c5 { background: var(--s5); color: var(--s5-ink); } .c6 { background: var(--s6); color: var(--s6-ink); }
.cn { background: var(--never); color: var(--never-ink); }
.cw { background: transparent; color: var(--muted); font-style: italic; }
.controls { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center; }
.controls .label { font-size: var(--step--1); color: var(--muted); }
.seg { display: inline-flex; flex-wrap: wrap; gap: 0.25rem; }
button.chip { font: 500 var(--step--1)/1 var(--font-ui); border: 0; border-radius: 999px; padding: 0.5rem 0.8rem;
  background: var(--chip); color: var(--ink); cursor: pointer; }
button.chip[aria-pressed="true"] { background: var(--chip-on); color: var(--chip-on-ink); }
button.chip:focus-visible, .mcard:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.tabs { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.image-head { display: grid; grid-template-columns: auto 1fr; gap: 1.25rem; align-items: start; }
.image-head img { width: 216px; max-width: 100%; height: auto; image-rendering: pixelated; border-radius: 6px; border: 1px solid var(--line); display: block; }
.image-head figcaption { max-width: 216px; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1rem; margin: 0; font-size: var(--step--1); align-content: start; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.multiples { display: grid; grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); gap: 0.6rem; }
.mcard { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 0.55rem 0.6rem 0.4rem; display: grid;
  gap: 0.2rem; cursor: pointer; min-width: 0; }
.mcard.on { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.mcard .t { display: flex; justify-content: space-between; gap: 0.5rem; font-size: var(--step--1); font-weight: 500; }
.mcard .t span:last-child { font-family: var(--font-mono); font-weight: 400; color: var(--ink-2); font-variant-numeric: tabular-nums; white-space: nowrap; }
.mcard .none { color: var(--muted); font-size: var(--step--1); padding: 1.2rem 0; }
.two { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 1rem; }
svg.plot { width: 100%; height: auto; display: block; overflow: visible; }
svg .grid { stroke: var(--line); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .tick { fill: var(--muted); font: 400 10px var(--font-mono); }
svg .axis-label { fill: var(--ink-2); font: 400 11px var(--font-ui); }
svg .ln-ref { fill: none; stroke: var(--ref); stroke-width: 1.25; stroke-linejoin: round; }
svg .ln-truth { fill: none; stroke: var(--truth); stroke-width: 1.25; stroke-linejoin: round; stroke-linecap: round; }
svg .ln-method { fill: none; stroke: var(--accent); stroke-width: 1.75; stroke-linejoin: round; stroke-linecap: round; }
svg .ln-ctx { fill: none; stroke: var(--ctx); stroke-width: 1; }
svg .ln-limit { stroke: var(--muted); stroke-width: 1; }
svg .dot { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
svg .dot-truth { fill: var(--truth); stroke: var(--surface); stroke-width: 2; }
svg .cross { stroke: var(--muted); stroke-width: 1; }
svg .note { fill: var(--muted); font: 400 10.5px var(--font-ui); }
.tip { position: fixed; z-index: 20; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--line);
  border-radius: 8px; padding: 0.45rem 0.6rem; font-size: var(--step--1); box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12); display: grid; gap: 0.2rem; min-width: 11rem; }
.tip .h { color: var(--muted); font-size: 0.75rem; }
.tip .r { display: grid; grid-template-columns: 14px auto 1fr; gap: 0.4rem; align-items: center; }
.tip .r i { width: 14px; border-top: 2px solid; }
.tip .r b { font: 500 var(--step--1) var(--font-mono); font-variant-numeric: tabular-nums; }
.tip .r span { color: var(--ink-2); }
.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr)); gap: 1.25rem 2rem; }
.notes > div { display: grid; gap: 0.5rem; align-content: start; min-width: 0; }
.notes ul { margin: 0; padding-left: 1.1rem; display: grid; gap: 0.35rem; color: var(--ink-2); }
.notes p { color: var(--ink-2); }
code { font-family: var(--font-mono); font-size: 0.9em; }
@media (max-width: 720px) {
  .two { grid-template-columns: 1fr; }
  .image-head { grid-template-columns: 1fr; }
  .featured-row { grid-template-columns: 1fr; gap: 0.2rem; }
  h1 { font-size: 1.9rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
<!--/head-->
<main class="page">
  <header class="masthead">
    <p class="eyebrow" id="eyebrow">subpx · synthetic hole arrays</p>
    <h1>Pitch signature test</h1>
    <p class="lede">Every row about 1 µm apart is moved 1 nm. Which center methods show that step in the y pitch when only a few columns are averaged?</p>
    <dl class="facts" id="run-facts"></dl>
  </header>

  <section aria-labelledby="answer-h">
    <div class="section-head"><h2 id="answer-h">Answer</h2></div>
    <div class="answer" id="answer"></div>
    <figure class="card" id="featured"></figure>
  </section>

  <section aria-labelledby="best-h">
    <div class="section-head">
      <h2 id="best-h">Best method per image</h2>
      <p>Columns that must be averaged before the 1 nm step stands out (curve noise at most a third of it), and each method's own center error per hole. Perfect centers set the floor: the mask's placement noise needs that many columns even with no measurement error.</p>
    </div>
    <div class="table-wrap"><table id="best"></table></div>
  </section>

  <section aria-labelledby="matrix-h">
    <div class="section-head">
      <h2 id="matrix-h">Columns needed, every method</h2>
      <p>Lighter is better. With the mask's placement noise, perfect centers need the same number of columns as the best methods; switch to "method alone" to compare the methods as if the mask were perfect.</p>
    </div>
    <div class="controls">
      <span class="label">Count</span>
      <div class="seg" id="mode"></div>
    </div>
    <div class="table-wrap"><table class="matrix" id="matrix"></table></div>
    <div class="legend" id="matrix-legend"></div>
  </section>

  <section aria-labelledby="explore-h">
    <div class="section-head">
      <h2 id="explore-h">One image at a time</h2>
      <p>Pick an image, the number of columns averaged and a method (click a plot or a table row).</p>
    </div>
    <div class="tabs" id="tabs" role="group" aria-label="Test image"></div>
    <div class="image-head" id="image-head"></div>
    <div class="controls"><span class="label">Columns averaged</span><div class="seg" id="ncols"></div>
      <span class="label">Rows shown</span><div class="seg" id="rows"></div></div>
    <figure class="card" id="focus"></figure>
    <div class="two">
      <figure class="card" id="noise"></figure>
      <figure class="card" id="phase"></figure>
    </div>
    <div class="legend">
      <span class="key"><i class="method"></i>method's y pitch</span>
      <span class="key"><i class="truth"></i>true centers (mask placement noise only)</span>
      <span class="key"><i class="ref"></i>planted step</span>
    </div>
    <div class="multiples" id="multiples"></div>
    <div class="table-wrap"><table id="table"></table></div>
  </section>

  <section aria-labelledby="notes-h">
    <div class="section-head"><h2 id="notes-h">How the numbers are made</h2></div>
    <div class="notes" id="notes"></div>
  </section>
</main>
<div class="tip" id="tip" role="tooltip" hidden></div>
<script type="application/json" id="report-data">__DATA__</script>
<script>
(() => {
"use strict";
const D = JSON.parse(document.getElementById("report-data").textContent);
const S = D.settings, IMAGES = D.images;
const NS = "http://www.w3.org/2000/svg";
const state = { img: 0, n: "8", method: null, mode: "seen", rows: 50 };
let clipCount = 0;
const $ = (id) => document.getElementById(id);
const shift = S.row_shift;
const visible = Math.abs(shift) / 3;

function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function sv(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function stext(parent, x, y, text, cls, anchor) {
  const t = sv("text", { x, y, class: cls, "text-anchor": anchor || "start" }, parent);
  t.textContent = text;
  return t;
}
const ok = (v) => v != null && Number.isFinite(v);
const fmt = (v, d = 2) => (ok(v) ? v.toFixed(d) : "–");
const signed = (v, d = 2) => (ok(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d) : "–");
const pct = (v) => (ok(v) ? (v >= 0.995 ? "100 %" : (100 * v).toFixed(v >= 0.9 ? 1 : 0) + " %") : "–");
const byKey = (im) => Object.fromEntries(im.methods.map((m) => [m.key, m]));
const usable = (m) => m && !m.failed && m.found >= 0.99;
function needText(v, im) {
  if (v == null) return "never";
  return v === im.columns_total ? "all " + v : String(v);
}
function colsText(v, im) { return v == null ? "never shows it" : needText(v, im) + (v === 1 ? " column" : " columns"); }
const cut = (a) => (a && state.rows ? a.slice(0, state.rows) : a);
function needOf(m, mode) { return mode === "method" ? m.need_method : m.need_seen; }
function noiseOf(m, mode) { return mode === "method" ? m.noise_method : m.noise_seen; }
function groupTag(m) {
  return { truth: "reference", cd: "CD", gpu: "GPU", new: "prototype" }[m.group] || m.group;
}

/* ---------- tooltip ---------- */
const tip = $("tip");
function showTip(evt, title, rows) {
  tip.replaceChildren();
  tip.appendChild(h("div", "h", title));
  for (const [value, label, cls] of rows) {
    if (value === undefined) continue;
    const r = h("div", "r");
    const i = h("i");
    i.style.borderColor = `var(--${cls})`;
    r.append(i, h("b", null, typeof value === "string" ? value : signed(value, 2) + " nm"), h("span", null, label));
    tip.appendChild(r);
  }
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, hh = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + hh > window.innerHeight - 8) y = evt.clientY - hh - pad;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}
function hideTip() { tip.hidden = true; }

/* ---------- y-pitch plot ---------- */
function linePath(values, X, Y) {
  let d = "", pen = false;
  values.forEach((v, i) => {
    if (!ok(v)) { pen = false; return; }
    d += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
    pen = true;
  });
  return d;
}
function yRange(im, n) {
  const t = byKey(im).truth;
  const noise = t && t.noise_seen ? t.noise_seen[n] : 0.5;
  return Math.max(1.5, Math.ceil((1.25 * Math.abs(shift) + 3 * (ok(noise) ? noise : 0.5)) * 2) / 2);
}
function pitchPlot(o) {
  const { w, hgt, compact } = o;
  const m = { l: compact ? 28 : 40, r: 6, t: compact ? 6 : 22, b: compact ? 4 : 28 };
  const n = o.planted.length;
  const X = (i) => m.l + (w - m.l - m.r) * (n > 1 ? i / (n - 1) : 0.5);
  const Y = (v) => m.t + (hgt - m.t - m.b) * (1 - (Math.max(-o.yMax * 1.04, Math.min(o.yMax * 1.04, v)) + o.yMax) / (2 * o.yMax));
  const svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "plot", role: "img", "aria-label": o.label });
  const id = "clip" + ++clipCount;
  const cp = sv("clipPath", { id }, sv("defs", {}, svg));
  sv("rect", { x: m.l, y: m.t - 1, width: w - m.l - m.r, height: hgt - m.t - m.b + 2 }, cp);
  const step = o.yMax > 3 ? 2 : 1;
  for (let v = -Math.floor(o.yMax / step) * step; v <= o.yMax + 1e-9; v += step) {
    sv("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: v === 0 ? "axis" : "grid" }, svg);
    stext(svg, m.l - 5, Y(v) + 3.5, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v), "tick", "end");
  }
  if (!compact) {
    stext(svg, m.l, hgt - 8, "row pair, from the top of the page →", "axis-label");
    stext(svg, w - m.r, hgt - 8, `${n} row pairs`, "tick", "end");
  }
  if (!compact) stext(svg, 4, 11, "y pitch − design, nm", "axis-label");
  const g = sv("g", { "clip-path": `url(#${id})` }, svg);
  sv("path", { d: linePath(o.planted, X, Y), class: "ln-ref" }, g);
  if (o.truth) sv("path", { d: linePath(o.truth, X, Y), class: "ln-truth" }, g);
  if (o.curve) sv("path", { d: linePath(o.curve, X, Y), class: "ln-method" }, g);
  const cross = sv("line", { y1: m.t, y2: hgt - m.b, class: "cross", visibility: "hidden" }, svg);
  const hit = sv("rect", { x: m.l, y: 0, width: w - m.l - m.r, height: hgt, fill: "transparent" }, svg);
  const move = (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) * w) / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (w - m.l - m.r)) * (n - 1))));
    cross.setAttribute("x1", X(i));
    cross.setAttribute("x2", X(i));
    cross.setAttribute("visibility", "visible");
    showTip(e, `rows ${i} → ${i + 1}`, [
      [o.curve ? o.curve[i] : undefined, o.name, "accent"],
      [o.truth ? o.truth[i] : undefined, "true centers", "truth"],
      [o.planted[i], "planted", "ref"],
    ]);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  return svg;
}

/* ---------- run facts and answer ---------- */
function runFacts() {
  const dl = $("run-facts");
  const im0 = IMAGES[0];
  const items = [
    ["Images", `${IMAGES.length} · ${S.shape[1]} × ${S.shape[0]} px`],
    ["Pixel", `${S.pixel_nm} nm`],
    ["Blur", `${S.blur} nm wide (lens ${S.lens} nm)`],
    ["Noise per hole", `CD ${S.cd_sigma} nm · placement ${S.placement_sigma} nm · brightness ${Math.round(S.brightness_sigma * 100)} %`],
    ["Planted", `+${S.row_shift} nm every ~${S.row_period} nm of rows`],
    ["Run", `${Math.round(S.total_seconds)} s on the ${S.backend.toUpperCase()}`],
  ];
  for (const [k, v] of items) {
    const d = h("div");
    d.append(h("dt", null, k), h("dd", null, v));
    dl.appendChild(d);
  }
  $("eyebrow").textContent = `subpx · synthetic hole arrays · seed ${S.seed}`;
}

function bestPerImage() {
  return IMAGES.map((im) => {
    const M = byKey(im);
    return { im, M, truth: M.truth, cd: M[im.best_cd], gpu: M[im.best_gpu], neu: M[im.best_new] };
  });
}

function answer() {
  const box = $("answer");
  const rows = bestPerImage();
  const tn = rows.map((r) => r.truth && r.truth.need_seen).filter((v) => v != null);
  const lo = Math.min(...tn), hi = Math.max(...tn);
  const floor = lo === hi ? `${lo} columns` : `${lo} to ${hi} columns`;
  const same = (key) => rows.filter((r) => r[key] && r.truth && r[key].need_seen != null && r[key].need_seen <= r.truth.need_seen).length;
  const p1 = h("p");
  p1.innerHTML = "";
  p1.append("Even perfect centers need ", h("strong", null, floor),
    ` averaged to show the ${shift} nm step: every hole also sits ${S.placement_sigma} nm (1 sd) off its design place, and that noise has to average out first.`);
  const p2 = h("p");
  p2.append("The best CD method gets there with no more columns on ", h("strong", null, `${same("cd")} of ${rows.length}`),
    " images, the best GPU finder on ", h("strong", null, `${same("gpu")} of ${rows.length}`));
  if (rows.some((r) => r.neu)) p2.append(", the template-fit prototype on ", h("strong", null, `${same("neu")} of ${rows.length}`));
  p2.append(".");
  const p3 = h("p");
  p3.append("A method needs more columns when its own error is large next to the mask's noise, or when its error depends on where the hole sits inside its pixel: rows run parallel to the pixel rows, so a whole row shares that error and it never averages away. The plot below shows such a case.");
  box.append(p1, p2, p3);
}

function featured() {
  const fig = $("featured");
  let pick = null;
  for (const im of IMAGES) {
    const M = byKey(im), t = M.truth;
    if (!t || t.failed) continue;
    for (const m of im.methods) {
      if (!usable(m) || m.group === "truth" || !m.noise_method || !ok(m.noise_method["8"]) || !ok(m.noise_method["1"])) continue;
      const excess = m.noise_method["8"] / (m.noise_method["1"] / Math.sqrt(8));  // 1 = pure averaging
      const score = (im.meta.row_shift_every >= 4 ? 100 : 0) + (m.need_seen == null ? 10 : 0)
        + excess * (m.center_sd_y < 0.8 ? 1 : 0);
      if (!pick || score > pick.score) pick = { im, m, score };
    }
  }
  if (!pick) { fig.hidden = true; return; }
  const im = pick.im, M = byKey(im);
  const better = im.methods.filter((m) => usable(m) && (m.group === "cd" || m.group === "gpu") && !m.needs_blur
      && m.key !== pick.m.key && m.noise_method)
    .sort((a, b) => (a.need_seen ?? 1e9) - (b.need_seen ?? 1e9) || a.noise_method["8"] - b.noise_method["8"])[0];
  const n = im.keys.includes("8") ? "8" : im.keys[0];
  const yMax = yRange(im, n);
  const grid = h("div", "featured-grid");
  const rows = [[M.truth, "True centers"], [pick.m, pick.m.name], better ? [better, better.name] : null].filter(Boolean);
  for (const [m, name] of rows) {
    const row = h("div", "featured-row");
    const who = h("div", "who");
    who.append(h("strong", null, name),
      h("span", null, m.group === "truth" ? `needs ${colsText(m.need_seen, im)}` :
        `${fmt(m.center_sd_y)} nm per hole · ${m.need_seen == null ? "never shows it" : "needs " + colsText(m.need_seen, im)}`));
    row.appendChild(who);
    const rows50 = (a) => (a ? a.slice(0, 50) : a);
    row.appendChild(pitchPlot({ w: 760, hgt: 92, compact: true, planted: rows50(im.planted), yMax, name,
      curve: m.group === "truth" ? null : rows50(m.profiles[n]), truth: rows50(M.truth.profiles[n]),
      label: `${name}: y pitch with ${n} columns averaged` }));
    grid.appendChild(row);
  }
  fig.appendChild(grid);
  const cap = h("figcaption");
  cap.textContent = `${im.label}, ${n} columns averaged, middle band of the page, first 50 row pairs. Orange: the planted step (+${shift} nm into a moved row, −${shift} nm out of it). Grey: the true centers. Blue: the method. ${pick.m.name} keeps a repeating pattern of its own (pixel-position error ${fmt(pick.m.phase_error)} nm, curve noise ${fmt(pick.m.noise_method.all)} nm even with all ${im.columns_total} columns).`;
  fig.appendChild(cap);
}

/* ---------- best per image ---------- */
function bestTable() {
  const t = $("best");
  const head = h("thead"), tr = h("tr");
  const hasNew = IMAGES.some((im) => im.best_new);
  const cols = ["Image", "Perfect centers", "Best CD method", "Best GPU finder"].concat(hasNew ? ["Template fit (prototype)"] : []).concat(["Best possible per hole"]);
  cols.forEach((c, i) => tr.appendChild(h("th", i === cols.length - 1 ? "num" : null, c)));
  head.appendChild(tr);
  t.appendChild(head);
  const body = h("tbody");
  for (const r of bestPerImage()) {
    const row = h("tr");
    row.appendChild(h("td", null, r.im.label));
    const cell = (m, isTruth, hideName) => {
      const td = h("td");
      if (!m) { td.textContent = "–"; return td; }
      const d = h("div", "best-cell");
      if (!isTruth && !hideName) d.appendChild(h("div", null, m.name));
      d.appendChild(h("span", null, colsText(m.need_seen, r.im) + (isTruth ? "" : ` · ${fmt(m.center_sd_y)} nm`)));
      td.appendChild(d);
      return td;
    };
    row.append(cell(r.truth, true), cell(r.cd), cell(r.gpu));
    if (hasNew) row.appendChild(cell(r.neu, false, true));
    row.appendChild(h("td", "num", ok(r.im.limit_nm) ? `${fmt(r.im.limit_nm)} nm` : "–"));
    body.appendChild(row);
  }
  t.appendChild(body);
}

/* ---------- matrix ---------- */
const GROUP_ORDER = { truth: 0, new: 1, cd: 2, gpu: 3 };
function needClass(v) {
  if (v == null) return "cn";
  if (v <= 2) return "c1";
  if (v <= 4) return "c2";
  if (v <= 8) return "c3";
  if (v <= 16) return "c4";
  if (v <= 32) return "c5";
  return "c6";
}
function matrix() {
  const t = $("matrix");
  t.replaceChildren();
  const keys = [];
  const info = {};
  for (const im of IMAGES) for (const m of im.methods) if (!info[m.key]) { info[m.key] = m; keys.push(m.key); }
  const atFloor = (key) => IMAGES.filter((im) => {
    const M = byKey(im), m = M[key], t = M.truth;
    return m && usable(m) && t && needOf(m, state.mode) != null && needOf(m, state.mode) <= needOf(t, state.mode);
  }).length;
  const score = (key) => IMAGES.reduce((s, im) => {
    const m = byKey(im)[key];
    const v = m && usable(m) ? needOf(m, state.mode) : null;
    return s + Math.log2(v == null ? 256 : v);
  }, 0);
  keys.sort((a, b) => (GROUP_ORDER[info[a].group] ?? 9) - (GROUP_ORDER[info[b].group] ?? 9)
    || atFloor(b) - atFloor(a) || score(a) - score(b));
  const head = h("thead"), tr = h("tr");
  tr.appendChild(h("th", null, "Method"));
  for (const im of IMAGES) tr.appendChild(h("th", "img", im.label));
  head.appendChild(tr);
  t.appendChild(head);
  const body = h("tbody");
  for (const key of keys) {
    const row = h("tr", "pick");
    row.tabIndex = 0;
    const who = h("td", "who", info[key].name);
    who.appendChild(h("span", "tag", groupTag(info[key])));
    if (info[key].needs_blur) who.appendChild(h("span", "tag", "true blur"));
    row.appendChild(who);
    IMAGES.forEach((im) => {
      const m = byKey(im)[key];
      let td;
      if (!m || m.failed) td = h("td", "cell cw", "no centers");
      else if (m.found < 0.99) td = h("td", "cell cw", Math.floor(100 * m.found) + " % found");
      else { const v = needOf(m, state.mode); td = h("td", "cell " + needClass(v), needText(v, im)); }
      if (m && !m.failed) td.title = `${m.name} on ${im.label}: found ${pct(m.found)}, ${fmt(m.center_sd_y)} nm per hole`;
      row.appendChild(td);
    });
    const select = () => { state.method = key; renderImage(); $("explore-h").scrollIntoView({ behavior: "smooth", block: "start" }); };
    row.addEventListener("click", select);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } });
    body.appendChild(row);
  }
  t.appendChild(body);
  const lg = $("matrix-legend");
  lg.replaceChildren();
  [["c1", "1–2"], ["c2", "4"], ["c3", "8"], ["c4", "16"], ["c5", "32"], ["c6", "64 or more"], ["cn", "never (curve noise stays above a third of the step)"]].forEach(([c, label]) => {
    const k = h("span", "key"), b = h("b", c);
    k.append(b, label);
    lg.appendChild(k);
  });
  lg.appendChild(h("span", "key", "italic: method found under 99 % of the holes"));
}
function modeButtons() {
  const box = $("mode");
  [["seen", "With the mask's placement noise"], ["method", "Method alone"]].forEach(([k, label]) => {
    const b = h("button", "chip", label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(state.mode === k));
    b.addEventListener("click", () => {
      state.mode = k;
      box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      matrix();
      renderImage();
    });
    box.appendChild(b);
  });
}

/* ---------- explorer ---------- */
function tabs() {
  const box = $("tabs");
  IMAGES.forEach((im, i) => {
    const b = h("button", "chip", im.label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(i === state.img));
    b.addEventListener("click", () => {
      state.img = i;
      state.method = null;
      box.querySelectorAll("button").forEach((x, j) => x.setAttribute("aria-pressed", String(j === i)));
      renderImage();
    });
    box.appendChild(b);
  });
}
function nButtons(im) {
  const box = $("ncols");
  box.replaceChildren();
  if (!im.keys.includes(state.n)) state.n = im.keys.includes("8") ? "8" : im.keys[0];
  for (const k of im.keys) {
    const b = h("button", "chip", k === "all" ? `all ${im.columns_total}` : k);
    b.type = "button";
    b.setAttribute("aria-pressed", String(k === state.n));
    b.addEventListener("click", () => { state.n = k; renderImage(); });
    box.appendChild(b);
  }
}
function defaultMethod(im) {
  return im.best_cd || im.best_gpu || im.best_new || im.methods.find((m) => m.group !== "truth")?.key;
}
function rowButtons(im) {
  const box = $("rows");
  box.replaceChildren();
  [[50, "first 50 row pairs"], [0, `whole page (${im.row_intervals})`]].forEach(([v, label]) => {
    const b = h("button", "chip", label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(state.rows === v));
    b.addEventListener("click", () => { state.rows = v; renderImage(); });
    box.appendChild(b);
  });
}
function imageHead(im) {
  const box = $("image-head");
  box.replaceChildren();
  const fig = h("figure");
  if (im.crop_png) {
    const img = h("img");
    img.src = "data:image/png;base64," + im.crop_png;
    img.alt = `Center of the ${im.label} test page`;
    fig.appendChild(img);
    fig.appendChild(h("figcaption", null, "Page center, 72 × 72 px, contrast stretched."));
  }
  box.appendChild(fig);
  const kv = h("dl", "kv");
  const meta = im.meta;
  const add = (k, v) => { kv.append(h("dt", null, k), h("dd", null, v)); };
  add("Holes", `${meta.hole}, ${meta.size_nm.map((v) => Math.round(v)).join(" × ")} nm (${meta.size_px.map((v) => v.toFixed(1)).join(" × ")} px)`);
  add("Layout", `${meta.layout}, pitch ${meta.pitch_nm.map((v) => Math.round(v)).join(" / ")} nm (x / row)`);
  add("Rows moved", `every ${meta.row_shift_every} rows (${Math.round(meta.row_shift_every * meta.pitch_nm[1])} nm), +${meta.row_shift_nm} nm`);
  add("Holes measured", `${im.evaluated_holes.toLocaleString("en-US")} (${im.pairs.toLocaleString("en-US")} pitch pairs, ${im.columns_total} columns × ${im.row_intervals} row pairs)`);
  add("Best possible", ok(im.limit_nm) ? `${fmt(im.limit_nm)} nm per hole (one hole alone, its own noise only)` : "–");
  add("Seconds", `${fmt(im.seconds, 1)} for this image, all methods`);
  box.appendChild(kv);
}
function focusPlot(im, m) {
  const fig = $("focus");
  fig.replaceChildren();
  const M = byKey(im);
  const title = h("h3", null, `${m.name}: y pitch with ${state.n === "all" ? "all " + im.columns_total : state.n} column${state.n === "1" ? "" : "s"} averaged`);
  fig.appendChild(title);
  if (m.failed || !m.profiles || !m.profiles[state.n]) { fig.appendChild(h("p", null, "This method gave no centers on this image.")); return; }
  fig.appendChild(pitchPlot({ w: 1000, hgt: 260, compact: false, planted: cut(im.planted), yMax: yRange(im, state.n), name: m.name,
    curve: m.group === "truth" ? null : cut(m.profiles[state.n]), truth: cut(M.truth.profiles[state.n]), label: `${m.name} y pitch` }));
  const ns = m.noise_seen || {}, nm = m.noise_method || {};
  fig.appendChild(h("figcaption", null,
    `Middle band of columns. Curve noise ${fmt(ns[state.n])} nm with the mask's placement noise, ${fmt(nm[state.n])} nm from the method alone; step seen with all columns ${fmt(m.spike)} nm (true centers ${fmt(M.truth.spike)} nm).`));
}
function noiseChart(im, sel) {
  const fig = $("noise");
  fig.replaceChildren();
  fig.appendChild(h("h3", null, state.mode === "method" ? "Curve noise from the method alone" : "Curve noise you would see"));
  const w = 560, hgt = 300, m = { l: 46, r: 12, t: 10, b: 38 };
  const Ns = im.keys.map((k) => (k === "all" ? im.columns_total : +k));
  const nMax = Math.max(...Ns);
  const X = (N) => m.l + ((w - m.l - m.r) * Math.log2(N)) / Math.log2(nMax);
  const lo = 0.02, hi = 10;
  const Y = (v) => m.t + (hgt - m.t - m.b) * (1 - (Math.log10(Math.max(lo, Math.min(hi, v))) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)));
  const svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "plot", role: "img", "aria-label": "Curve noise against columns averaged" });
  [0.03, 0.1, 0.3, 1, 3, 10].forEach((v) => {
    sv("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg);
    stext(svg, m.l - 5, Y(v) + 3.5, String(v), "tick", "end");
  });
  im.keys.forEach((k, i) => stext(svg, X(Ns[i]), hgt - m.b + 14, k === "all" ? String(im.columns_total) : k, "tick", "middle"));
  stext(svg, (m.l + w - m.r) / 2, hgt - 6, "columns averaged", "axis-label", "middle");
  sv("line", { x1: m.l, x2: w - m.r, y1: Y(visible), y2: Y(visible), class: "ln-limit" }, svg);
  stext(svg, w - m.r, Y(visible) - 5, `step visible below ${fmt(visible)} nm`, "note", "end");
  const line = (method, cls) => {
    const nz = noiseOf(method, state.mode);
    if (!nz) return;
    const vals = im.keys.map((k) => nz[k]);
    let d = "", pen = false;
    vals.forEach((v, i) => {
      if (!ok(v) || v <= 0) { pen = false; return; }
      d += (pen ? "L" : "M") + X(Ns[i]).toFixed(1) + " " + Y(v).toFixed(1);
      pen = true;
    });
    sv("path", { d, class: cls }, svg);
    return vals;
  };
  for (const other of im.methods) if (usable(other) && other.key !== sel.key && other.group !== "truth") line(other, "ln-ctx");
  const M = byKey(im);
  const tv = state.mode === "method" ? null : line(M.truth, "ln-truth");
  const mv = sel.group === "truth" || !usable(sel) && sel.failed ? null : line(sel, "ln-method");
  if (mv) mv.forEach((v, i) => { if (ok(v) && v > 0) sv("circle", { cx: X(Ns[i]), cy: Y(v), r: 4, class: "dot" }, svg); });
  const cross = sv("line", { y1: m.t, y2: hgt - m.b, class: "cross", visibility: "hidden" }, svg);
  const hit = sv("rect", { x: m.l, y: 0, width: w - m.l - m.r, height: hgt, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) * w) / r.width;
    let best = 0;
    Ns.forEach((N, i) => { if (Math.abs(X(N) - px) < Math.abs(X(Ns[best]) - px)) best = i; });
    cross.setAttribute("x1", X(Ns[best]));
    cross.setAttribute("x2", X(Ns[best]));
    cross.setAttribute("visibility", "visible");
    const rows = [];
    if (mv) rows.push([fmt(mv[best]) + " nm", sel.name, "accent"]);
    if (tv) rows.push([fmt(tv[best]) + " nm", "true centers", "truth"]);
    showTip(e, `${Ns[best]} column${Ns[best] === 1 ? "" : "s"} averaged`, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  fig.appendChild(svg);
  const lg = h("div", "legend");
  const key = (cls, text) => { const k = h("span", "key"); k.append(h("i", cls), text); return k; };
  lg.append(key("method", sel.name));
  if (tv) lg.append(key("truth", "true centers"));
  lg.append(key("ctx", "other methods"));
  fig.appendChild(lg);
  fig.appendChild(h("figcaption", null, "Root-mean-square error of the y-pitch curve over every band of that many columns. Averaging alone makes it fall along a straight line; a method whose error repeats row after row levels off."));
}
function phaseChart(im, sel) {
  const fig = $("phase");
  fig.replaceChildren();
  fig.appendChild(h("h3", null, "Error against place inside the pixel"));
  const curve = sel.phase_curve;
  if (!curve || sel.group === "truth" || sel.failed) {
    fig.appendChild(h("p", null, "Pick a method to see its mean y error against where the hole's true center sits inside its pixel."));
    return;
  }
  const w = 420, hgt = 300, m = { l: 46, r: 10, t: 10, b: 38 };
  const vals = curve.filter(ok);
  const span = Math.max(0.1, Math.ceil(Math.max(...vals.map(Math.abs), 0.05) * 1.25 * 20) / 20);
  const nb = curve.length;
  const xc = (i) => -0.5 + (i + 0.5) / nb;
  const X = (u) => m.l + (w - m.l - m.r) * (u + 0.5);
  const Y = (v) => m.t + (hgt - m.t - m.b) * (1 - (v + span) / (2 * span));
  const svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "plot", role: "img", "aria-label": "Mean y error against place inside the pixel" });
  const tickStep = span > 0.6 ? 0.5 : span > 0.25 ? 0.1 : 0.05;
  for (let v = -Math.floor(span / tickStep) * tickStep; v <= span + 1e-9; v += tickStep) {
    const vv = Math.round(v * 1000) / 1000;
    sv("line", { x1: m.l, x2: w - m.r, y1: Y(vv), y2: Y(vv), class: Math.abs(vv) < 1e-9 ? "axis" : "grid" }, svg);
    stext(svg, m.l - 5, Y(vv) + 3.5, (vv > 0 ? "+" : vv < 0 ? "−" : "") + Math.abs(vv).toFixed(tickStep < 0.1 ? 2 : 1), "tick", "end");
  }
  [-0.5, -0.25, 0, 0.25, 0.5].forEach((u) => stext(svg, X(u), hgt - m.b + 16, (u > 0 ? "+" : u < 0 ? "−" : "") + Math.abs(u), "tick",
    u === -0.5 ? "start" : u === 0.5 ? "end" : "middle"));
  stext(svg, (m.l + w - m.r) / 2, hgt - 6, "true center − nearest pixel center, px", "axis-label", "middle");
  let d = "", pen = false;
  curve.forEach((v, i) => { if (!ok(v)) { pen = false; return; } d += (pen ? "L" : "M") + X(xc(i)).toFixed(1) + " " + Y(v).toFixed(1); pen = true; });
  sv("path", { d, class: "ln-method" }, svg);
  curve.forEach((v, i) => {
    if (!ok(v)) return;
    const c = sv("circle", { cx: X(xc(i)), cy: Y(v), r: 4, class: "dot" }, svg);
    const hit = sv("circle", { cx: X(xc(i)), cy: Y(v), r: 12, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (e) => showTip(e, `true center ${signed(xc(i), 3)} px from the pixel center`, [[v, "mean y error", "accent"]]));
    hit.addEventListener("pointerleave", hideTip);
    void c;
  });
  fig.appendChild(svg);
  fig.appendChild(h("figcaption", null, `${sel.name}: pixel-position error ${fmt(sel.phase_error)} nm (half the peak-to-peak). Empty places: no hole sits there (few distinct places when the pitch is close to a whole number of pixels).`));
}
function multiples(im, sel) {
  const box = $("multiples");
  box.replaceChildren();
  const M = byKey(im);
  const yMax = yRange(im, state.n);
  for (const m of tableOrder(im)) {
    if (m.group === "truth") continue;
    const card = h("div", "mcard" + (m.key === sel.key ? " on" : ""));
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-pressed", String(m.key === sel.key));
    const t = h("div", "t");
    const need = needOf(m, state.mode);
    t.append(h("span", null, m.name), h("span", null, m.failed ? "" : `${fmt(m.center_sd_y)} nm · ${need == null ? "never" : needText(need, im) + " col"}`));
    card.appendChild(t);
    if (m.failed || !m.profiles || !m.profiles[state.n]) card.appendChild(h("div", "none", "No centers on this image."));
    else {
      if (m.found < 0.99) card.appendChild(h("div", "t", `found ${pct(m.found)} of the holes`));
      card.appendChild(pitchPlot({ w: 320, hgt: 96, compact: true, planted: cut(im.planted), yMax, name: m.name,
        curve: cut(m.profiles[state.n]), truth: cut(M.truth.profiles[state.n]), label: `${m.name} y pitch` }));
    }
    const select = () => { state.method = m.key; renderImage(); };
    card.addEventListener("click", select);
    card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } });
    box.appendChild(card);
  }
}
function tableOrder(im) {
  const rank = (m) => (m.group === "truth" ? -1 : m.failed ? 3 : m.found < 0.99 ? 2 : 0);
  return im.methods.slice().sort((a, b) => rank(a) - rank(b)
    || (needOf(a, state.mode) ?? 1e9) - (needOf(b, state.mode) ?? 1e9)
    || (a.center_sd_y ?? 1e9) - (b.center_sd_y ?? 1e9));
}
function table(im, sel) {
  const t = $("table");
  t.replaceChildren();
  const head = h("thead"), tr = h("tr");
  [["Method", ""], ["Found", "num"], ["Center error, nm", "num"], ["Pixel-position error, nm", "num"], ["Step seen, nm", "num"],
    ["Columns needed", "num"], ["Columns needed, method alone", "num"], ["Seconds (CD methods: one shared call)", "num"]].forEach(([c, cls]) => tr.appendChild(h("th", cls, c)));
  head.appendChild(tr);
  t.appendChild(head);
  const body = h("tbody");
  for (const m of tableOrder(im)) {
    const row = h("tr", "pick" + (m.key === sel.key ? " on" : "") + (m.failed || m.found < 0.99 ? " weak" : ""));
    row.tabIndex = 0;
    const who = h("td", null, m.name);
    who.appendChild(h("span", "tag", groupTag(m)));
    if (m.needs_blur) who.appendChild(h("span", "tag", "true blur"));
    if (m.key === im.best_gpu) who.appendChild(h("span", "tag", "best GPU"));
    if (m.key === im.best_cd) who.appendChild(h("span", "tag", "best CD"));
    row.appendChild(who);
    const cells = m.failed ? [pct(m.found), "–", "–", "–", "–", "–", fmt(m.seconds, 1)] :
      [pct(m.found), fmt(m.center_sd_y), fmt(m.phase_error), fmt(m.spike), needText(m.need_seen, im), needText(m.need_method, im),
        m.group === "truth" ? "–" : fmt(m.seconds, 1)];
    cells.forEach((v) => row.appendChild(h("td", "num", v)));
    const select = () => { state.method = m.key; renderImage(); };
    row.addEventListener("click", select);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } });
    body.appendChild(row);
  }
  t.appendChild(body);
}
function renderImage() {
  const im = IMAGES[state.img];
  const M = byKey(im);
  if (!state.method || !M[state.method]) state.method = defaultMethod(im);
  const sel = M[state.method];
  nButtons(im);
  rowButtons(im);
  imageHead(im);
  focusPlot(im, sel);
  noiseChart(im, sel);
  phaseChart(im, sel);
  multiples(im, sel);
  table(im, sel);
}

/* ---------- notes ---------- */
function notes() {
  const box = $("notes");
  const block = (title, items, para) => {
    const d = h("div");
    d.appendChild(h("h3", null, title));
    if (para) d.appendChild(h("p", null, para));
    if (items) { const ul = h("ul"); items.forEach((t) => ul.appendChild(h("li", null, t))); d.appendChild(ul); }
    box.appendChild(d);
  };
  block("The test pages", [
    "Square or staggered arrays of circles, squares and ellipses on a 2048 × 1378 px page of 21 nm pixels, 8-bit, bright holes on a dark background.",
    "Each hole is drawn from its exact shape: shape × blur × pixel area in Fourier space, so a noiseless hole's centroid is its true center to 1e-9 px. No drawing grid can fake a pixel-position error.",
    `Per hole: CD ${S.cd_sigma} nm, placement ${S.placement_sigma} nm (x and y) and brightness ${Math.round(S.brightness_sigma * 100)} % at random (1 sd); photon noise (${S.photons} photons per grey level) and read noise.`,
    `The rows closest to every ${S.row_period} nm are moved +${S.row_shift} nm in y. Rows run parallel to the pixel rows.`,
  ]);
  block("The numbers", [
    "Found: share of holes (one pitch or more from the page edge) the method gave a center for, within half a spacing of the true one.",
    "Center error: spread (1 sd) of measured − true y over all found holes.",
    "Pixel-position error: half the peak-to-peak of the mean y error against where the true center sits inside its pixel. It is shared by a whole row and never averages away.",
    "y pitch: as analyze_pitch, the mean of the (row r, column k) → (row r + 1, column k) pairs over a band of N columns. Curve noise: its root-mean-square error over every band, after removing each band's mean.",
    "Columns needed: fewest N whose curve noise is at most a third of the step. With the mask's placement noise (against design + planted step) or method alone (against the true centers).",
    "Step seen: least-squares size of the planted pattern in the all-columns curve; the true centers give about the planted size.",
    "Best possible per hole: Cramér–Rao bound for one hole alone with its brightness and background fitted, photon, read and rounding noise only.",
  ]);
  block("The methods", [
    "CD methods: centers from one measure_cds call with all methods on the GPU (summary mode). The blurred-hole fit and flux ratio are given the true blur, which real pages lack.",
    "GPU finders: detect_centers with each refine method on the GPU, Otsu threshold, area limits from the design hole: the finders the workbench offers for pitch.",
    "Template fit (prototype): each hole fitted with the image's own average hole (scripts/template_fit_prototype.py). Not in the library: it did not pass the bar set before the test (best on at least 6 of the 8 images).",
    "Fixed on the way: the GPU edge-erf finder measured its straight-line background from the page's left or top edge, so holes more than ~140 px from it were refused. On 300 nm squares it found 0.6 % of the holes before the fix and 100 % after.",
  ]);
  block("Not in these pages", [
    "Rotation: rows tilted against the pixel rows spread the pixel-position error over the row and help every method.",
    "Brightness or background changing row by row, stage or scan wobble, distortion.",
    "The real ACTIS blur shape, line/space areas next to the holes, larger or correlated placement noise.",
  ]);
  const st = h("div");
  st.appendChild(h("h3", null, "Settings"));
  const kv = h("dl", "kv");
  for (const k of ["seed", "blur", "lens", "photons", "cd_sigma", "placement_sigma", "brightness_sigma", "row_shift", "row_period", "rotation", "backend"]) {
    if (S[k] === undefined) continue;
    kv.append(h("dt", null, k), h("dd", null, String(S[k])));
  }
  st.appendChild(kv);
  st.appendChild(h("p", null, "Re-run: python scripts/pitch_signature_test.py (options: --help)."));
  box.appendChild(st);
}

runFacts();
answer();
featured();
bestTable();
modeButtons();
matrix();
tabs();
renderImage();
notes();
})();
</script>"""


def main(argv=None):
    path = Path((argv or sys.argv[1:])[0])
    data = json.loads(path.read_text(encoding="utf-8"))
    print(write_report(data, path.with_name("report.html")))


if __name__ == "__main__":
    main()
