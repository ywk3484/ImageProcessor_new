"""HTML report of scripts/pitch_signature_test.py: one self-contained page (data, charts and code inline).

    python scripts/pitch_signature_view.py artifacts/pitch_signature/results.json      # rewrite report.html

``write_report(data, path)`` writes a full HTML document; ``standalone=False`` leaves out the document skeleton
(doctype, html, head, body) for hosts that add their own.

Every y-pitch graph is a stack of panels on one x axis with one line per panel: the true centers in grey, each
measured method in blue, all on the same scale, so truth and measurement are never drawn over each other. In "One
image at a time" the method list stays beside the graph (click, or the up / down keys), so methods are switched
without scrolling.
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
    """The numbers the page needs: per image the methods in a list (the run's order: CD methods, GPU finders,
    prototype, true centers) and their y-pitch curves at 0.01 nm."""
    images = []
    for image in data["images"]:
        methods = image["methods"]
        truth = methods.get("truth", {})
        keys = list(truth.get("noise_seen", {}).keys())
        entries = []
        for key, method in methods.items():
            entry = {k: method.get(k) for k in KEEP_METHOD}
            entry["key"] = key
            entry["profiles"] = _clean(method.get("profiles", {}), 2)
            entries.append(entry)
        images.append(dict(
            key=image["key"], label=image["label"], meta={k: image["meta"].get(k) for k in KEEP_META},
            limit_nm=image.get("limit_nm"), crop_png=image.get("crop_png"), evaluated_holes=image["evaluated_holes"],
            pairs=image["pairs"], seconds=image.get("seconds"), best_gpu=image.get("best_gpu"),
            best_cd=image.get("best_cd"), best_new=image.get("best_new"), keys=keys,
            columns_total=truth.get("columns_total"), row_intervals=truth.get("row_intervals"), methods=entries))
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
  /* Layout: one column of stacked sections (80rem); the explorer is a sticky method list beside the graphs; wide
     tables scroll inside their own frames. Palette: cool instrument greys; grey line = the true centers, blue line =
     a measured method; never both in one panel. */
  --bg: #f3f5f7;
  --surface: #fcfdfd;
  --ink: #0f1419;
  --ink-2: #46505b;
  --muted: #76808a;
  --line: #e1e5e9;
  --axis: #c3c9d0;
  --accent: #2a78d6;
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
    --axis: #3c434a; --accent: #3987e5; --truth: #a0a8b0; --ctx: #2f353b; --chip: #232930;
    --chip-on: #eef1f4; --chip-on-ink: #0e1113; --focus: #6da7ec;
    --s1: #104281; --s1-ink: #ffffff; --s2: #184f95; --s2-ink: #ffffff; --s3: #256abf; --s3-ink: #ffffff;
    --s4: #3987e5; --s4-ink: #ffffff; --s5: #6da7ec; --s5-ink: #0b0b0b; --s6: #9ec5f4; --s6-ink: #0b0b0b;
    --never: #262c31; --never-ink: #c1c8cf;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1113; --surface: #161a1d; --ink: #eef1f4; --ink-2: #c1c8cf; --muted: #8c959e; --line: #2a3035;
  --axis: #3c434a; --accent: #3987e5; --truth: #a0a8b0; --ctx: #2f353b; --chip: #232930;
  --chip-on: #eef1f4; --chip-on-ink: #0e1113; --focus: #6da7ec;
  --s1: #104281; --s1-ink: #ffffff; --s2: #184f95; --s2-ink: #ffffff; --s3: #256abf; --s3-ink: #ffffff;
  --s4: #3987e5; --s4-ink: #ffffff; --s5: #6da7ec; --s5-ink: #0b0b0b; --s6: #9ec5f4; --s6-ink: #0b0b0b;
  --never: #262c31; --never-ink: #c1c8cf;
  color-scheme: dark;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body {
  margin: 0; background: var(--bg); color: var(--ink); font: 400 var(--step-0)/1.55 var(--font-ui);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
}
.page { max-width: 80rem; margin: 0 auto; padding-block: 2.5rem 4rem; display: grid; gap: 3rem; }
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
figcaption { color: var(--ink-2); font-size: var(--step--1); max-width: 90ch; }
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; font-size: var(--step--1); color: var(--ink-2); align-items: center; }
.key { display: inline-flex; align-items: center; gap: 0.4rem; }
.key i { display: inline-block; width: 18px; height: 0; border-top: 2px solid; }
.key i.method { border-color: var(--accent); }
.key i.compare { border-color: var(--accent); border-top-style: dashed; }
.key i.truth { border-color: var(--truth); }
.key i.ctx { border-color: var(--ctx); border-top-width: 1px; }
.key b { display: inline-block; width: 14px; height: 14px; border-radius: 3px; }
.table-wrap { overflow-x: auto; min-width: 0; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: var(--step--1); }
th, td { padding: 0.45rem 0.6rem; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { font-weight: 500; color: var(--ink-2); background: var(--surface); position: sticky; top: 0; }
th.num { text-align: right; }
td.num { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
#table td:first-child, #best td:first-child { min-width: 14rem; }
tbody tr:last-child td { border-bottom: 0; }
tr.pick { cursor: pointer; }
tr.pick:hover td { background: color-mix(in srgb, var(--accent) 6%, transparent); }
tr.pick.on td { background: color-mix(in srgb, var(--accent) 13%, transparent); }
tr.pick:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; }
tr.weak td { color: var(--muted); }
.tag { display: inline-block; font: 500 0.625rem/1 var(--font-mono); letter-spacing: 0.04em; text-transform: uppercase;
  padding: 0.2rem 0.3rem; border-radius: 4px; background: var(--chip); color: var(--ink-2); margin-left: 0.35rem; vertical-align: 1px; white-space: nowrap; }
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
button.chip:focus-visible, .mcard:focus-visible, select.pick:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.tabs { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.tabs button.chip { padding: 0.42rem 0.7rem; }
select.pick { font: 500 var(--step--1)/1.2 var(--font-ui); background: var(--chip); color: var(--ink); border: 0; border-radius: 8px;
  padding: 0.45rem 0.6rem; max-width: 100%; }
/* explorer: method list beside the graphs */
.explorer { display: grid; grid-template-columns: 17.5rem minmax(0, 1fr); gap: 1.25rem; align-items: start; }
.mlist { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 0.75rem); max-height: calc(100vh - 1.5rem);
  display: flex; flex-direction: column; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; min-width: 0; }
.mlist-head { display: grid; gap: 0.15rem; padding: 0.7rem 0.8rem 0.55rem; border-bottom: 1px solid var(--line); font-size: var(--step--1); }
.mlist-head strong { font-weight: 600; font-size: var(--step-0); }
.mlist-head span { color: var(--muted); font-size: 0.75rem; line-height: 1.35; }
.mlist-box { overflow-y: auto; padding: 0.2rem 0.35rem 0.5rem; position: relative; flex: 1 1 auto; min-height: 0; }
.mlist-box:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; border-radius: 0 0 10px 10px; }
.grp { display: grid; grid-template-columns: minmax(0, 1fr) auto 2.4rem; gap: 0.5rem; font: 500 0.6875rem/1 var(--font-mono);
  letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); padding: 0.85rem 0.45rem 0.35rem; }
.grp span:not(:first-child) { text-align: right; text-transform: none; letter-spacing: 0; }
.opt { display: grid; grid-template-columns: minmax(0, 1fr) auto 2.4rem; gap: 0.5rem; align-items: center; padding: 0.3rem 0.45rem;
  border-radius: 6px; cursor: pointer; font-size: var(--step--1); line-height: 1.3; }
.opt:hover { background: color-mix(in srgb, var(--accent) 7%, transparent); }
.opt.on { background: color-mix(in srgb, var(--accent) 16%, transparent); box-shadow: inset 3px 0 0 var(--accent); }
.opt .nm { min-width: 0; }
.opt .need { font: 500 0.75rem/1 var(--font-mono); padding: 0.22rem 0.35rem; border-radius: 4px; min-width: 2.7rem; text-align: center;
  font-variant-numeric: tabular-nums; }
.opt .err { font: 400 0.75rem/1 var(--font-mono); color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }
.opt.weak .nm { color: var(--muted); }
.explorer-main { display: grid; gap: 1rem; min-width: 0; }
.image-head { display: grid; grid-template-columns: 104px minmax(0, 1fr); gap: 1rem; align-items: start; }
.image-head img { width: 104px; height: 104px; image-rendering: pixelated; border-radius: 6px; border: 1px solid var(--line); display: block; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 0.25rem 1rem; margin: 0; font-size: var(--step--1); align-content: start; }
.kv.two-up { grid-template-columns: max-content minmax(0, 1fr) max-content minmax(0, 1fr); }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; font-family: var(--font-mono); font-variant-numeric: tabular-nums; min-width: 0; }
.multiples { display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: 0.6rem; }
.mcard { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 0.5rem 0.55rem 0.35rem; display: grid;
  gap: 0.2rem; cursor: pointer; min-width: 0; }
.mcard.on { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.mcard.truth { cursor: default; }
.mcard .t { display: flex; justify-content: space-between; gap: 0.5rem; font-size: var(--step--1); font-weight: 500; }
.mcard .t span:last-child { font-family: var(--font-mono); font-weight: 400; color: var(--ink-2); font-variant-numeric: tabular-nums; white-space: nowrap; }
.mcard .none { color: var(--muted); font-size: var(--step--1); padding: 1.2rem 0; }
.two { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 1rem; }
svg.plot { width: 100%; height: auto; display: block; overflow: visible; }
svg .grid { stroke: var(--line); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .tick { fill: var(--muted); font: 400 10px var(--font-mono); }
svg .axis-label { fill: var(--ink-2); font: 400 11px var(--font-ui); }
svg .ptitle { fill: var(--ink); font: 600 12.5px var(--font-ui); }
svg .psub { fill: var(--ink-2); font: 400 11px var(--font-mono); }
svg .pnone { fill: var(--muted); font: 400 12px var(--font-ui); }
svg .sw-truth { stroke: var(--truth); stroke-width: 3; stroke-linecap: round; }
svg .sw-method { stroke: var(--accent); stroke-width: 3; stroke-linecap: round; }
svg .ln-truth { fill: none; stroke: var(--truth); stroke-width: 1.75; stroke-linejoin: round; stroke-linecap: round; }
svg .ln-method { fill: none; stroke: var(--accent); stroke-width: 1.75; stroke-linejoin: round; stroke-linecap: round; }
svg .ln-compare { fill: none; stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 5 4; stroke-linejoin: round; }
svg .ln-ctx { fill: none; stroke: var(--ctx); stroke-width: 1; }
svg .ln-limit { stroke: var(--muted); stroke-width: 1; }
svg .ln-now { stroke: var(--axis); stroke-width: 6; opacity: 0.45; }
svg .dot { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
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
@media (max-width: 1100px) {
  .kv.two-up { grid-template-columns: max-content minmax(0, 1fr); }
}
@media (max-width: 860px) {
  .explorer { grid-template-columns: minmax(0, 1fr); }
  .mlist { position: static; max-height: 17rem; }
  .two { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 560px) {
  .image-head { grid-template-columns: minmax(0, 1fr); }
  h1 { font-size: 1.9rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; scroll-behavior: auto !important; } }
</style>
<!--/head-->
<main class="page">
  <header class="masthead">
    <p class="eyebrow" id="eyebrow">subpx · synthetic hole arrays</p>
    <h1>Pitch signature test</h1>
    <p class="lede">Rows about 1 µm apart are moved 1 nm. Which center methods show that in the y pitch when only a few columns are averaged?</p>
    <dl class="facts" id="run-facts"></dl>
  </header>

  <section aria-labelledby="answer-h">
    <div class="section-head"><h2 id="answer-h">Answer</h2></div>
    <figure class="card" id="featured"></figure>
    <div class="answer" id="answer"></div>
  </section>

  <section aria-labelledby="best-h">
    <div class="section-head">
      <h2 id="best-h">Best method per image</h2>
      <p>Columns that must be averaged before the 1 nm step stands out (curve noise at most a third of it), and each method's own center error per hole. Click a row to open that image below.</p>
    </div>
    <div class="table-wrap"><table id="best"></table></div>
  </section>

  <section aria-labelledby="matrix-h">
    <div class="section-head">
      <h2 id="matrix-h">Columns needed, every method</h2>
      <p>Lighter is better. "With the mask's placement noise" is what you would see; "method alone" compares the methods as if the mask were perfect. Click a row to look at that method below.</p>
    </div>
    <div class="controls">
      <span class="label">Count</span>
      <div class="seg" id="mode"></div>
    </div>
    <div class="table-wrap"><table class="matrix" id="matrix"></table></div>
    <div class="legend" id="matrix-legend"></div>
  </section>

  <section aria-labelledby="explore-h" id="explore">
    <div class="section-head">
      <h2 id="explore-h">One image at a time</h2>
      <p>Pick an image, then a method in the list on the left: click it, or use ↑ and ↓ once the list has focus. The graphs change in place.</p>
    </div>
    <div class="tabs" id="tabs" role="group" aria-label="Test image"></div>
    <div class="explorer">
      <nav class="mlist" aria-label="Methods">
        <div class="mlist-head"><strong>Methods</strong><span id="mlist-note"></span></div>
        <div class="mlist-box" id="mlist-box" role="listbox" tabindex="0" aria-label="Method shown in the graphs"></div>
      </nav>
      <div class="explorer-main">
        <div class="image-head" id="image-head"></div>
        <div class="controls">
          <span class="label">Columns averaged</span><div class="seg" id="ncols"></div>
          <span class="label">Rows shown</span><div class="seg" id="rows"></div>
          <label class="label" for="compare">Compare with</label><select class="pick" id="compare"></select>
        </div>
        <figure class="card" id="focus"></figure>
        <div class="two">
          <figure class="card" id="noise"></figure>
          <figure class="card" id="phase"></figure>
        </div>
        <div class="section-head">
          <h3 id="multiples-h">Every method at once</h3>
          <p id="multiples-note"></p>
        </div>
        <div class="multiples" id="multiples"></div>
        <div class="table-wrap"><table id="table"></table></div>
      </div>
    </div>
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
const FEATURED = Math.max(0, IMAGES.findIndex((im) => im.key === "circle100_square"));
const state = { img: FEATURED, n: "4", method: null, compare: null, mode: "seen", rows: 50 };
const Y_MAX = 2;  // nm, every y-pitch panel
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
const measured = (im) => im.methods.filter((m) => m.group !== "truth");
function needText(v, im) {
  if (v == null) return "never";
  return v === im.columns_total ? "all " + v : String(v);
}
function colsText(v, im) { return v == null ? "never shows it" : needText(v, im) + (v === 1 ? " column" : " columns"); }
function needOf(m, mode) { return mode === "method" ? m.need_method : m.need_seen; }
function noiseOf(m, mode) { return mode === "method" ? m.noise_method : m.noise_seen; }
function needWords(m, im) {
  if (!m || m.failed) return "no centers";
  const v = needOf(m, state.mode);
  return v == null ? "never shows the step" : "needs " + colsText(v, im);
}
const cut = (a) => (a && state.rows ? a.slice(0, state.rows) : a);
function groupTag(m) { return { truth: "reference", cd: "CD", gpu: "GPU", new: "prototype" }[m.group] || m.group; }
function needClass(v) {
  if (v == null) return "cn";
  if (v <= 2) return "c1";
  if (v <= 4) return "c2";
  if (v <= 8) return "c3";
  if (v <= 16) return "c4";
  if (v <= 32) return "c5";
  return "c6";
}
function nLabel(n, im) { return n === "all" ? `all ${im.columns_total}` : n; }

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
    r.append(i, h("b", null, typeof value === "string" ? value : ok(value) ? signed(value, 2) + " nm" : "–"), h("span", null, label));
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

/* ---------- stacked y-pitch panels: one line per panel, one x axis, one scale ---------- */
function linePath(values, X, Y) {
  let d = "", pen = false;
  values.forEach((v, i) => {
    if (!ok(v)) { pen = false; return; }
    d += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
    pen = true;
  });
  return d;
}
function stackPlot(o) {
  const w = o.w || 960, compact = !!o.compact;
  const m = { l: compact ? 24 : 40, r: 8, t: compact ? 5 : 2 };
  const head = compact ? 0 : 24, ph = o.panelH || 104, gap = compact ? 0 : 12;
  const axisH = o.axis === false ? 5 : 36;
  const P = o.panels.length, n = Math.max(1, o.count);
  const hgt = m.t + P * (head + ph) + (P - 1) * gap + axisH;
  const X = (i) => m.l + (w - m.l - m.r) * (n > 1 ? i / (n - 1) : 0.5);
  const top = (p) => m.t + p * (head + ph + gap) + head;
  const Y = (p, v) => top(p) + ph * (1 - (Math.max(-Y_MAX * 1.08, Math.min(Y_MAX * 1.08, v)) + Y_MAX) / (2 * Y_MAX));
  const svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "plot", role: "img", "aria-label": o.label });
  const defs = sv("defs", {}, svg);
  const ticks = compact ? [-1, 0, 1] : [-2, -1, 0, 1, 2];
  o.panels.forEach((pn, p) => {
    if (!compact) {
      const ty = top(p) - 8;
      sv("line", { x1: m.l + 1, x2: m.l + 17, y1: ty - 4, y2: ty - 4, class: pn.truth ? "sw-truth" : "sw-method" }, svg);
      stext(svg, m.l + 24, ty, pn.label, "ptitle");
      if (pn.sub) stext(svg, w - m.r, ty, pn.sub, "psub", "end");
    }
    for (const v of ticks) {
      sv("line", { x1: m.l, x2: w - m.r, y1: Y(p, v), y2: Y(p, v), class: v === 0 ? "axis" : "grid" }, svg);
      stext(svg, m.l - 5, Y(p, v) + 3.5, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v), "tick", "end");
    }
    if (pn.values) {
      const id = "clip" + ++clipCount;
      sv("rect", { x: m.l, y: top(p) - 1, width: w - m.l - m.r, height: ph + 2 }, sv("clipPath", { id }, defs));
      sv("path", { d: linePath(pn.values, X, (v) => Y(p, v)), class: pn.truth ? "ln-truth" : "ln-method", "clip-path": `url(#${id})` }, svg);
    } else {
      stext(svg, (m.l + w - m.r) / 2, top(p) + ph / 2 + 4, pn.empty || "No centers from this method on this image.", "pnone", "middle");
    }
  });
  const yb = top(P - 1) + ph;
  if (o.axis !== false && o.rowPitch) {
    const span = (n * o.rowPitch) / 1000;
    const step = [0.5, 1, 2, 5, 10, 20, 50].find((s) => span / s <= 12) || 100;
    for (let u = 0; u <= span + 1e-9; u += step) {
      const i = (u * 1000) / o.rowPitch - 0.5;
      if (i < -0.01 || i > n - 1 + 0.01) continue;
      sv("line", { x1: X(i), x2: X(i), y1: yb, y2: yb + 4, class: "axis" }, svg);
      stext(svg, X(i), yb + 16, String(+u.toFixed(1)), "tick", "middle");
    }
    stext(svg, w - m.r, yb + 31, "µm down the page →", "axis-label", "end");
    stext(svg, m.l, yb + 31, "y pitch − design, nm", "axis-label");
  }
  const cross = sv("line", { y1: top(0), y2: yb, class: "cross", visibility: "hidden" }, svg);
  const hit = sv("rect", { x: m.l, y: 0, width: w - m.l - m.r, height: hgt, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) * w) / r.width;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (w - m.l - m.r)) * (n - 1))));
    cross.setAttribute("x1", X(i));
    cross.setAttribute("x2", X(i));
    cross.setAttribute("visibility", "visible");
    const where = o.rowPitch ? ` · ${(((i + 0.5) * o.rowPitch) / 1000).toFixed(2)} µm down` : "";
    const rows = o.panels.map((pn) => [pn.values ? pn.values[i] : undefined, pn.label, pn.truth ? "truth" : "accent"]);
    if (o.extra) rows.push(...o.extra(i));
    showTip(e, `rows ${i + 1} → ${i + 2}${where}`, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  return svg;
}
function curveOf(m, n) {
  return m && !m.failed && m.profiles && m.profiles[n] ? m.profiles[n] : null;
}

/* ---------- run facts and answer ---------- */
function runFacts() {
  const dl = $("run-facts");
  const items = [
    ["Images", `${IMAGES.length} · ${S.shape[1]} × ${S.shape[0]} px`],
    ["Pixel", `${S.pixel_nm} nm`],
    ["Blur", `${S.blur} nm wide (lens ${S.lens} nm)`],
    ["Noise per hole (1 sd)", `CD ${S.cd_sigma} nm · placement ${S.placement_sigma} nm · brightness ${Math.round(S.brightness_sigma * 100)} %`],
    ["Moved rows", `+${S.row_shift} nm, ~${S.row_period / 1000} µm apart, at least ${S.min_rows_apart ?? 2} rows`],
    ["Run", `${Math.round(S.total_seconds)} s on the ${S.backend.toUpperCase()}${S.timing_note ? " (faster of 2 identical runs)" : ""}`],
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
function span(values) {
  const v = values.filter((x) => x != null);
  if (!v.length) return null;
  const lo = Math.min(...v), hi = Math.max(...v);
  return lo === hi ? String(lo) : `${lo} to ${hi}`;
}
function answer() {
  const box = $("answer");
  const rows = bestPerImage();
  const need = (m) => (m && usable(m) ? m.need_seen : null);
  const total = rows.length;
  const truthSpan = span(rows.map((r) => need(r.truth))) || "–";
  const cdSpan = span(rows.map((r) => need(r.cd)));
  const gpuSpan = span(rows.map((r) => need(r.gpu)));
  const gpuNever = rows.filter((r) => need(r.gpu) == null).length;
  let cdWins = 0, ties = 0, gpuWins = 0;
  for (const r of rows) {
    const a = need(r.cd) ?? 1e9, b = need(r.gpu) ?? 1e9;
    if (a < b) cdWins++; else if (a > b) gpuWins++; else ties++;
  }
  const p1 = h("p");
  p1.append("Grey lines are the true y pitch, blue lines what a method measured, both averaged over a few columns. A row moved by ",
    `${shift} nm shows as a short jump up and then down; between moved rows the line stays flat.`);
  const cdNever = rows.filter((r) => need(r.cd) == null).length;
  const tally = {};
  for (const r of rows) if (r.cd) tally[r.cd.name] = (tally[r.cd.name] || 0) + 1;
  const [topName, topCount] = Object.entries(tally).sort((a, b) => b[1] - a[1])[0] || ["", 0];
  const p2 = h("p");
  p2.append("The true centers show every step with ", h("strong", null, `${truthSpan} column${truthSpan === "1" ? "" : "s"}`),
    " averaged. The best CD method on each image needs ", h("strong", null, cdSpan ? `${cdSpan} columns` : "more than all columns"),
    topCount > 1 ? ` (the ${topName} on ${topCount} of the ${total} images)` : "",
    cdNever ? `, and no CD method shows it on ${cdNever}` : "",
    "; the best GPU finder needs ", h("strong", null, gpuSpan ? `${gpuSpan} columns` : "more than all columns"),
    gpuNever ? ` (on ${gpuNever} of the ${total} images no GPU finder shows it at all)` : "", ".");
  const p3 = h("p");
  p3.append("The best CD method needs fewer columns than the best GPU finder on ", h("strong", null, `${cdWins} of ${total}`),
    ` images, as many on ${ties}, more on ${gpuWins}.`);
  const p4 = h("p");
  p4.append("A method needs more columns when its own error per hole is large, or when that error depends on where the hole sits inside its pixel: rows run along the pixel rows, so a whole row shares that error and averaging never removes it.");
  box.append(p1, p2, p3, p4);
}
function featured() {
  const fig = $("featured");
  const im = IMAGES[FEATURED];
  if (!im) { fig.hidden = true; return; }
  const M = byKey(im);
  const n = im.keys.includes("4") ? "4" : im.keys[0];
  const rows = [[M.truth, "True centers", true], [M[im.best_cd], "best CD method"], [M[im.best_gpu], "best GPU finder"]];
  const bad = measured(im).filter((m) => m.group === "gpu" && usable(m) && m.need_seen == null)
    .sort((a, b) => (b.phase_error ?? 0) - (a.phase_error ?? 0))[0];
  if (bad) rows.push([bad, "never shows the step"]);
  const panels = rows.filter((r) => r[0]).map(([m, role, truth]) => ({
    label: truth ? role : `${m.name} (${role})`,
    sub: truth ? "the real pitch: mask placement noise only" : `${needWords(m, im)} · ${fmt(m.center_sd_y)} nm per hole`,
    values: (curveOf(m, n) || []).slice(0, 50), truth: !!truth,
  }));
  fig.appendChild(h("h3", null, `${im.label}: y pitch with ${n} columns averaged`));
  fig.appendChild(stackPlot({ w: 960, panelH: 84, count: panels[0].values.length, rowPitch: im.meta.pitch_nm[1], panels,
    label: `${im.label}: y pitch of the true centers and of measured centers` }));
  fig.appendChild(h("figcaption", null, `Each line is the y pitch from one row to the next, minus the design ${Math.round(im.meta.pitch_nm[1])} nm, averaged over ${n} neighbouring columns in the middle of the page; the first 50 rows. Grey: the true centers. Blue: measured. Same scale in every panel.`));
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
  bestPerImage().forEach((r, idx) => {
    const row = h("tr", "pick");
    row.tabIndex = 0;
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
    const open = () => { state.img = idx; renderImage(); $("explore").scrollIntoView({ behavior: "smooth", block: "start" }); };
    row.addEventListener("click", open);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    body.appendChild(row);
  });
  t.appendChild(body);
}

/* ---------- matrix ---------- */
const GROUP_ORDER = { truth: 0, new: 1, cd: 2, gpu: 3 };
function matrix() {
  const t = $("matrix");
  t.replaceChildren();
  const keys = [];
  const info = {};
  for (const im of IMAGES) for (const m of im.methods) if (!info[m.key]) { info[m.key] = m; keys.push(m.key); }
  const score = (key) => IMAGES.reduce((s, im) => {
    const m = byKey(im)[key];
    const v = m && usable(m) ? needOf(m, state.mode) : null;
    return s + Math.log2(v == null ? 256 : v);
  }, 0);
  keys.sort((a, b) => (GROUP_ORDER[info[a].group] ?? 9) - (GROUP_ORDER[info[b].group] ?? 9) || score(a) - score(b));
  const head = h("thead"), tr = h("tr");
  tr.appendChild(h("th", null, "Method"));
  for (const im of IMAGES) tr.appendChild(h("th", "img", im.label));
  head.appendChild(tr);
  t.appendChild(head);
  const body = h("tbody");
  for (const key of keys) {
    const row = h("tr", info[key].group === "truth" ? "" : "pick");
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
    if (info[key].group !== "truth") {
      row.tabIndex = 0;
      const select = () => { pick(key); $("explore").scrollIntoView({ behavior: "smooth", block: "start" }); };
      row.addEventListener("click", select);
      row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } });
    }
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
    b.addEventListener("click", () => { state.img = i; renderImage(); });
    box.appendChild(b);
  });
}
function defaultMethod(im) {
  return im.best_cd || im.best_gpu || im.best_new || measured(im)[0]?.key;
}
function methodList(im) {
  const box = $("mlist-box");
  box.replaceChildren();
  $("mlist-note").textContent = state.mode === "method"
    ? "Chip: columns needed, method alone. Last: center error per hole, nm."
    : "Chip: columns needed to show the step. Last: center error per hole, nm.";
  const groups = [["cd", "CD methods"], ["gpu", "GPU finders"], ["new", "New (prototype)"]];
  for (const [g, title] of groups) {
    const list = im.methods.filter((m) => m.group === g);
    if (!list.length) continue;
    const head = h("div", "grp");
    head.setAttribute("role", "presentation");
    head.append(h("span", null, title), h("span", null, "cols"), h("span", null, "nm"));
    box.appendChild(head);
    for (const m of list) {
      const opt = h("div", "opt" + (usable(m) ? "" : " weak"));
      opt.id = "opt-" + m.key;
      opt.dataset.key = m.key;
      opt.setAttribute("role", "option");
      const name = h("span", "nm", m.name);
      if (m.key === im.best_cd) name.appendChild(h("span", "tag", "best CD"));
      if (m.key === im.best_gpu) name.appendChild(h("span", "tag", "best GPU"));
      if (m.needs_blur) name.appendChild(h("span", "tag", "true blur"));
      let chip;
      if (m.failed) chip = h("span", "need cw", "none");
      else if (m.found < 0.99) chip = h("span", "need cw", Math.floor(100 * m.found) + " %");
      else {
        const v = needOf(m, state.mode);
        chip = h("span", "need " + needClass(v), v == null ? "never" : v === im.columns_total ? "all" : String(v));
      }
      chip.title = m.failed ? "no centers" : m.found < 0.99 ? `found ${pct(m.found)} of the holes` : needWords(m, im);
      opt.append(name, chip, h("span", "err", m.failed ? "–" : fmt(m.center_sd_y)));
      opt.addEventListener("click", () => pick(m.key, true));
      box.appendChild(opt);
    }
  }
}
function showPicked() {
  const box = $("mlist-box");
  let on = null;
  box.querySelectorAll(".opt").forEach((o) => {
    const yes = o.dataset.key === state.method;
    o.classList.toggle("on", yes);
    o.setAttribute("aria-selected", String(yes));
    if (yes) on = o;
  });
  if (!on) { box.removeAttribute("aria-activedescendant"); return; }
  box.setAttribute("aria-activedescendant", on.id);
  // Keep the picked entry in view by scrolling the list only, never the page.
  const top = on.offsetTop, bottom = top + on.offsetHeight;
  if (top < box.scrollTop + 4) box.scrollTop = Math.max(0, top - 30);
  else if (bottom > box.scrollTop + box.clientHeight - 4) box.scrollTop = bottom - box.clientHeight + 8;
}
function pick(key, focusList) {
  state.method = key;
  renderMethod();
  if (focusList) $("mlist-box").focus({ preventScroll: true });
}
function listKeys(e) {
  const keys = [...$("mlist-box").querySelectorAll(".opt")].map((o) => o.dataset.key);
  if (!keys.length) return;
  let i = keys.indexOf(state.method);
  if (e.key === "ArrowDown") i = Math.min(keys.length - 1, i + 1);
  else if (e.key === "ArrowUp") i = Math.max(0, i - 1);
  else if (e.key === "Home") i = 0;
  else if (e.key === "End") i = keys.length - 1;
  else return;
  e.preventDefault();
  if (keys[i] !== state.method) pick(keys[i]);
}
function compareKey(im) {
  const M = byKey(im);
  if (state.compare === "none") return null;
  if (state.compare && M[state.compare] && state.compare !== state.method) return state.compare;
  if (im.best_gpu && im.best_gpu !== state.method) return im.best_gpu;
  if (im.best_cd && im.best_cd !== state.method) return im.best_cd;
  return null;
}
function compareSelect(im) {
  const sel = $("compare");
  sel.replaceChildren();
  const none = h("option", null, "nothing");
  none.value = "none";
  sel.appendChild(none);
  for (const [g, title] of [["cd", "CD methods"], ["gpu", "GPU finders"], ["new", "New (prototype)"]]) {
    const list = im.methods.filter((m) => m.group === g && m.key !== state.method);
    if (!list.length) continue;
    const og = h("optgroup");
    og.label = title;
    for (const m of list) {
      const o = h("option", null, m.name + (m.key === im.best_gpu ? " (best GPU)" : m.key === im.best_cd ? " (best CD)" : ""));
      o.value = m.key;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  sel.value = compareKey(im) || "none";
}
function nButtons(im) {
  const box = $("ncols");
  box.replaceChildren();
  if (!im.keys.includes(state.n)) state.n = im.keys.includes("4") ? "4" : im.keys[0];
  for (const k of im.keys) {
    const b = h("button", "chip", k === "all" ? `all ${im.columns_total}` : k);
    b.type = "button";
    b.setAttribute("aria-pressed", String(k === state.n));
    b.addEventListener("click", () => {
      state.n = k;
      box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderMethod();
    });
    box.appendChild(b);
  }
}
function rowButtons(im) {
  const box = $("rows");
  box.replaceChildren();
  [[50, "first 50"], [0, `all ${im.row_intervals}`]].forEach(([v, label]) => {
    const b = h("button", "chip", label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(state.rows === v));
    b.addEventListener("click", () => {
      state.rows = v;
      box.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderMethod();
    });
    box.appendChild(b);
  });
}
function imageHead(im) {
  const box = $("image-head");
  box.replaceChildren();
  if (im.crop_png) {
    const img = h("img");
    img.src = "data:image/png;base64," + im.crop_png;
    img.alt = `Center of the ${im.label} test page, 72 × 72 px`;
    img.title = "Page center, 72 × 72 px, contrast stretched";
    box.appendChild(img);
  }
  const kv = h("dl", "kv two-up");
  const meta = im.meta;
  const add = (k, v) => { kv.append(h("dt", null, k), h("dd", null, v)); };
  const period = (meta.row_shift_every * meta.pitch_nm[1]) / 1000;
  add("Holes", `${meta.hole} ${meta.size_nm.map((v) => Math.round(v)).join(" × ")} nm (${meta.size_px.map((v) => v.toFixed(1)).join(" × ")} px)`);
  add("Layout", `${meta.layout}, pitch ${meta.pitch_nm.map((v) => Math.round(v)).join(" / ")} nm (x / row)`);
  add("Moved rows", `every ${meta.row_shift_every} rows (${period.toFixed(2)} µm), +${meta.row_shift_nm} nm`);
  add("Measured", `${im.evaluated_holes.toLocaleString("en-US")} holes · ${im.columns_total} columns × ${im.row_intervals} rows`);
  add("Best possible", ok(im.limit_nm) ? `${fmt(im.limit_nm)} nm per hole` : "–");
  add("Seconds", `${fmt(im.seconds, 1)} for all methods`);
  box.appendChild(kv);
}
function focusPlot(im) {
  const fig = $("focus");
  fig.replaceChildren();
  const M = byKey(im);
  const sel = M[state.method];
  const ck = compareKey(im);
  const cmp = ck ? M[ck] : null;
  const nl = nLabel(state.n, im);
  fig.appendChild(h("h3", null, `y pitch with ${nl} column${state.n === "1" ? "" : "s"} averaged`));
  const panel = (m, truth) => ({
    label: truth ? "True centers" : m.name,
    sub: truth ? "the real pitch: mask placement noise only"
      : `${needWords(m, im)} · ${m.failed ? "–" : fmt(m.center_sd_y)} nm per hole${!m.failed && m.found < 0.99 ? " · found " + pct(m.found) : ""}`,
    values: cut(curveOf(m, state.n)), truth,
  });
  const panels = [panel(M.truth, true), panel(sel, false)];
  if (cmp) panels.push(panel(cmp, false));
  const count = (cut(curveOf(M.truth, state.n)) || []).length;
  fig.appendChild(stackPlot({ w: 960, panelH: 104, count, rowPitch: im.meta.pitch_nm[1], panels,
    label: `y pitch of the true centers, ${sel.name}${cmp ? " and " + cmp.name : ""}` }));
  const over = state.n === "1" ? "in one column in the middle of the page (no averaging)" : `averaged over ${nl} neighbouring columns in the middle of the page`;
  fig.appendChild(h("figcaption", null, `Each line is the y pitch from one row to the next, minus the design ${Math.round(im.meta.pitch_nm[1])} nm, ${over}${state.rows ? `; the first ${state.rows} rows` : ""}. Grey: the true centers. Blue: measured. Same scale (±${Y_MAX} nm) in every panel.`));
}
function noiseChart(im) {
  const fig = $("noise");
  fig.replaceChildren();
  const M = byKey(im);
  const sel = M[state.method];
  const ck = compareKey(im);
  const cmp = ck ? M[ck] : null;
  fig.appendChild(h("h3", null, state.mode === "method" ? "Curve noise from the method alone" : "Curve noise you would see"));
  const w = 560, hgt = 300, m = { l: 46, r: 12, t: 10, b: 38 };
  const Ns = im.keys.map((k) => (k === "all" ? im.columns_total : +k));
  const nMax = Math.max(...Ns);
  const X = (N) => m.l + ((w - m.l - m.r) * Math.log2(N)) / Math.log2(nMax);
  const lo = 0.02, hi = 10;
  const Y = (v) => m.t + (hgt - m.t - m.b) * (1 - (Math.log10(Math.max(lo, Math.min(hi, v))) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)));
  const svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "plot", role: "img", "aria-label": "Curve noise against columns averaged" });
  const now = Ns[im.keys.indexOf(state.n)];
  if (now) sv("line", { x1: X(now), x2: X(now), y1: m.t, y2: hgt - m.b, class: "ln-now" }, svg);
  [0.03, 0.1, 0.3, 1, 3, 10].forEach((v) => {
    sv("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg);
    stext(svg, m.l - 5, Y(v) + 3.5, String(v), "tick", "end");
  });
  im.keys.forEach((k, i) => stext(svg, X(Ns[i]), hgt - m.b + 14, k === "all" ? String(im.columns_total) : k, "tick", "middle"));
  stext(svg, (m.l + w - m.r) / 2, hgt - 6, "columns averaged", "axis-label", "middle");
  sv("line", { x1: m.l, x2: w - m.r, y1: Y(visible), y2: Y(visible), class: "ln-limit" }, svg);
  stext(svg, w - m.r, Y(visible) - 5, `step clear below ${fmt(visible)} nm`, "note", "end");
  const line = (method, cls) => {
    const nz = method && !method.failed ? noiseOf(method, state.mode) : null;
    if (!nz) return null;
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
  for (const other of measured(im)) if (usable(other) && other.key !== sel.key && other.key !== ck) line(other, "ln-ctx");
  const tv = state.mode === "method" ? null : line(M.truth, "ln-truth");
  const cv = line(cmp, "ln-compare");
  const mv = line(sel, "ln-method");
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
    if (cv) rows.push([fmt(cv[best]) + " nm", cmp.name + " (dashed)", "accent"]);
    if (tv) rows.push([fmt(tv[best]) + " nm", "true centers", "truth"]);
    showTip(e, `${Ns[best]} column${Ns[best] === 1 ? "" : "s"} averaged`, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  fig.appendChild(svg);
  const lg = h("div", "legend");
  const key = (cls, text) => { const k = h("span", "key"); k.append(h("i", cls), text); return k; };
  lg.append(key("method", sel.name));
  if (cv) lg.append(key("compare", cmp.name));
  if (tv) lg.append(key("truth", "true centers"));
  lg.append(key("ctx", "other methods"));
  fig.appendChild(lg);
  fig.appendChild(h("figcaption", null, "Root-mean-square error of the y-pitch curve over every band of that many columns (shaded: the number shown above). Averaging alone makes it fall along a straight line; a method whose error repeats row after row levels off."));
}
function phaseChart(im) {
  const fig = $("phase");
  fig.replaceChildren();
  const sel = byKey(im)[state.method];
  fig.appendChild(h("h3", null, "Error against place inside the pixel"));
  const curve = sel && sel.phase_curve;
  if (!curve || sel.failed) {
    fig.appendChild(h("p", null, "No centers from this method on this image."));
    return;
  }
  const w = 420, hgt = 300, m = { l: 46, r: 10, t: 10, b: 38 };
  const vals = curve.filter(ok);
  const range = Math.max(0.1, Math.ceil(Math.max(...vals.map(Math.abs), 0.05) * 1.25 * 20) / 20);
  const nb = curve.length;
  const xc = (i) => -0.5 + (i + 0.5) / nb;
  const X = (u) => m.l + (w - m.l - m.r) * (u + 0.5);
  const Y = (v) => m.t + (hgt - m.t - m.b) * (1 - (v + range) / (2 * range));
  const svg = sv("svg", { viewBox: `0 0 ${w} ${hgt}`, class: "plot", role: "img", "aria-label": "Mean y error against place inside the pixel" });
  const tickStep = range > 0.6 ? 0.5 : range > 0.25 ? 0.1 : 0.05;
  for (let v = -Math.floor(range / tickStep) * tickStep; v <= range + 1e-9; v += tickStep) {
    const vv = Math.round(v * 1000) / 1000;
    sv("line", { x1: m.l, x2: w - m.r, y1: Y(vv), y2: Y(vv), class: Math.abs(vv) < 1e-9 ? "axis" : "grid" }, svg);
    stext(svg, m.l - 5, Y(vv) + 3.5, (vv > 0 ? "+" : vv < 0 ? "−" : "") + Math.abs(vv).toFixed(tickStep < 0.1 ? 2 : 1), "tick", "end");
  }
  [-0.5, -0.25, 0, 0.25, 0.5].forEach((u) => stext(svg, X(u), hgt - m.b + 16, (u > 0 ? "+" : u < 0 ? "−" : "") + Math.abs(u), "tick",
    u === -0.5 ? "start" : u === 0.5 ? "end" : "middle"));
  stext(svg, (m.l + w - m.r) / 2, hgt - 6, "true center − nearest pixel center, px", "axis-label", "middle");
  sv("path", { d: linePath(curve, (i) => X(xc(i)), Y), class: "ln-method" }, svg);
  curve.forEach((v, i) => {
    if (!ok(v)) return;
    sv("circle", { cx: X(xc(i)), cy: Y(v), r: 4, class: "dot" }, svg);
    const hit = sv("circle", { cx: X(xc(i)), cy: Y(v), r: 12, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (e) => showTip(e, `true center ${signed(xc(i), 3)} px from the pixel center`, [[v, "mean y error", "accent"]]));
    hit.addEventListener("pointerleave", hideTip);
  });
  fig.appendChild(svg);
  fig.appendChild(h("figcaption", null, `${sel.name}: mean y error (nm) by where the true center sits inside its pixel. Pixel-position error ${fmt(sel.phase_error)} nm (half the peak-to-peak); a whole row shares it, so it never averages away. Gaps: no hole sits there.`));
}
function multiples(im) {
  const box = $("multiples");
  box.replaceChildren();
  const M = byKey(im);
  const truthCurve = cut(curveOf(M.truth, state.n));
  const count = (truthCurve || []).length;
  $("multiples-note").textContent = `${nLabel(state.n, im)} column${state.n === "1" ? "" : "s"} averaged, same scale (±${Y_MAX} nm) as above. Grey: true centers; blue: each method alone. Click one to pick it.`;
  const card = (m, truth) => {
    const c = h("div", "mcard" + (truth ? " truth" : "") + (!truth && m.key === state.method ? " on" : ""));
    const t = h("div", "t");
    const need = needOf(m, state.mode);
    t.append(h("span", null, truth ? "True centers" : m.name),
      h("span", null, m.failed ? "" : `${truth ? "" : fmt(m.center_sd_y) + " nm · "}${need == null ? "never" : needText(need, im) + " col"}`));
    c.appendChild(t);
    const values = cut(curveOf(m, state.n));
    if (!values) c.appendChild(h("div", "none", "No centers on this image."));
    else {
      if (!truth && m.found < 0.99) c.appendChild(h("div", "t", `found ${pct(m.found)} of the holes`));
      c.appendChild(stackPlot({ w: 320, compact: true, panelH: 64, axis: false, count, rowPitch: im.meta.pitch_nm[1],
        panels: [{ label: truth ? "true centers" : m.name, values, truth }], label: `${m.name} y pitch`,
        extra: truth ? null : (i) => [[truthCurve ? truthCurve[i] : undefined, "true centers", "truth"]] }));
    }
    if (!truth) {
      c.tabIndex = 0;
      c.setAttribute("role", "button");
      c.setAttribute("aria-pressed", String(m.key === state.method));
      c.addEventListener("click", () => pick(m.key));
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(m.key); } });
    }
    return c;
  };
  box.appendChild(card(M.truth, true));
  for (const m of measured(im)) box.appendChild(card(m, false));
}
function tableOrder(im) {
  const rank = (m) => (m.group === "truth" ? -1 : m.failed ? 3 : m.found < 0.99 ? 2 : 0);
  return im.methods.slice().sort((a, b) => rank(a) - rank(b)
    || (needOf(a, state.mode) ?? 1e9) - (needOf(b, state.mode) ?? 1e9)
    || (a.center_sd_y ?? 1e9) - (b.center_sd_y ?? 1e9));
}
function table(im) {
  const t = $("table");
  t.replaceChildren();
  const head = h("thead"), tr = h("tr");
  [["Method", ""], ["Found", "num"], ["Center error, nm", "num"], ["Pixel-position error, nm", "num"], ["Step seen, nm", "num"],
    ["Columns needed", "num"], ["Columns needed, method alone", "num"], ["Seconds (CD methods: one shared call)", "num"]].forEach(([c, cls]) => tr.appendChild(h("th", cls, c)));
  head.appendChild(tr);
  t.appendChild(head);
  const body = h("tbody");
  for (const m of tableOrder(im)) {
    const truth = m.group === "truth";
    const row = h("tr", (truth ? "" : "pick") + (m.key === state.method ? " on" : "") + (m.failed || m.found < 0.99 ? " weak" : ""));
    const who = h("td", null, m.name);
    who.appendChild(h("span", "tag", groupTag(m)));
    if (m.needs_blur) who.appendChild(h("span", "tag", "true blur"));
    if (m.key === im.best_gpu) who.appendChild(h("span", "tag", "best GPU"));
    if (m.key === im.best_cd) who.appendChild(h("span", "tag", "best CD"));
    row.appendChild(who);
    const cells = m.failed ? [pct(m.found), "–", "–", "–", "–", "–", fmt(m.seconds, 1)] :
      [pct(m.found), fmt(m.center_sd_y), fmt(m.phase_error), fmt(m.spike), needText(m.need_seen, im), needText(m.need_method, im),
        truth ? "–" : fmt(m.seconds, 1)];
    cells.forEach((v) => row.appendChild(h("td", "num", v)));
    if (!truth) {
      row.tabIndex = 0;
      row.addEventListener("click", () => pick(m.key));
      row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(m.key); } });
    }
    body.appendChild(row);
  }
  t.appendChild(body);
}
function renderMethod() {
  const im = IMAGES[state.img];
  showPicked();
  compareSelect(im);
  focusPlot(im);
  noiseChart(im);
  phaseChart(im);
  multiples(im);
  table(im);
}
function renderImage() {
  const im = IMAGES[state.img];
  const M = byKey(im);
  if (!state.method || !M[state.method] || M[state.method].group === "truth") state.method = defaultMethod(im);
  $("tabs").querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-pressed", String(j === state.img)));
  methodList(im);
  nButtons(im);
  rowButtons(im);
  imageHead(im);
  renderMethod();
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
  const periods = IMAGES.map((im) => (im.meta.row_shift_every * im.meta.pitch_nm[1]) / 1000);
  const periodText = `${Math.min(...periods).toFixed(2)} to ${Math.max(...periods).toFixed(2)} µm`;
  let wins = 0, ties = 0, losses = 0;
  for (const r of bestPerImage()) {
    if (!r.neu || !usable(r.neu)) continue;
    const others = measured(r.im).filter((m) => m.group !== "new" && usable(m) && !m.needs_blur);
    const best = Math.min(...others.map((m) => m.need_seen ?? 1e9));
    const own = r.neu.need_seen ?? 1e9;
    if (own < best) wins++; else if (own === best) ties++; else losses++;
  }
  block("The test pages", [
    `Square or staggered arrays of circles, squares and ellipses on a ${S.shape[1]} × ${S.shape[0]} px page of ${S.pixel_nm} nm pixels, 8-bit, bright holes on a dark background.`,
    "Each hole is drawn from its exact shape: shape × blur × pixel area in Fourier space, so a noiseless hole's centroid is its true center to better than 1e-6 px. No drawing grid can fake a pixel-position error.",
    `Per hole at random (1 sd): CD ${S.cd_sigma} nm, placement ${S.placement_sigma} nm (x and y), brightness ${Math.round(S.brightness_sigma * 100)} %; photon noise (${S.photons} photons per grey level) and read noise.`,
    `Moved rows: +${S.row_shift} nm in y, the rows closest to ${S.row_period} nm apart but at least ${S.min_rows_apart ?? 2} rows apart, so each moved row shows as its own jump up and down; here ${periodText} apart. Rows run parallel to the pixel rows.`,
  ]);
  block("The numbers", [
    "Found: share of holes (one pitch or more from the page edge) the method gave a center for, within half a spacing of the true one.",
    "Center error: spread (1 sd) of measured − true y over all found holes.",
    "Pixel-position error: half the peak-to-peak of the mean y error against where the true center sits inside its pixel. It is shared by a whole row and never averages away.",
    "y pitch: as analyze_pitch, the mean of the (row r, column k) → (row r + 1, column k) pairs over a band of N columns. The graphs show the band in the middle of the page. Curve noise: the root-mean-square error over every band, after removing each band's mean.",
    "Columns needed: fewest N whose curve noise is at most a third of the step. With the mask's placement noise (against design + planted step) or method alone (against the true centers).",
    "Step seen: least-squares size of the planted pattern in the all-columns curve; the true centers give about the planted size.",
    "Best possible per hole: the smallest center error any method can reach on one hole alone (Cramér–Rao bound with its brightness and background fitted; photon, read and rounding noise only).",
  ]);
  block("The methods", [
    "CD methods: centers from one measure_cds call with all methods on the GPU (summary mode). The blurred-hole fit and flux ratio are given the true blur, which real pages lack, so they never count as best.",
    "GPU finders: detect_centers with each refine method on the GPU, Otsu threshold, area limits from the design hole: the finders the workbench offers for pitch.",
    `Template fit (prototype): each hole fitted with the image's own average hole (scripts/template_fit_prototype.py), not in the library. Here it needs fewer columns than the best existing method on ${wins} images, as many on ${ties}, more on ${losses}.`,
  ]);
  block("Not in these pages", [
    "Rotation: rows tilted against the pixel rows spread the pixel-position error over the row and help every method.",
    "Brightness or background changing row by row, stage or scan wobble, distortion.",
    "The real ACTIS blur shape, line/space areas next to the holes, larger or correlated placement noise.",
  ]);
  const st = h("div");
  st.appendChild(h("h3", null, "Settings"));
  const kv = h("dl", "kv");
  for (const k of ["seed", "blur", "lens", "photons", "cd_sigma", "placement_sigma", "brightness_sigma", "row_shift", "row_period", "min_rows_apart", "rotation", "backend", "timing_note"]) {
    if (S[k] === undefined) continue;
    kv.append(h("dt", null, k), h("dd", null, String(S[k])));
  }
  st.appendChild(kv);
  st.appendChild(h("p", null, "Re-run: python scripts/pitch_signature_test.py (options: --help)."));
  box.appendChild(st);
}

$("mlist-box").addEventListener("keydown", listKeys);
$("compare").addEventListener("change", (e) => { state.compare = e.target.value || "none"; renderMethod(); });
runFacts();
featured();
answer();
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
