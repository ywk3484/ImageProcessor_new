"""Measure every PNG in a folder with all CD methods and write an offline HTML report.

    python scripts/cd_report.py FOLDER --output REPORT --nominal-cd 100 --pixel-size 21 --unit nm --backend gpu

Open REPORT/index.html in a browser (works from the folder, no server). It has a summary tab (per image and
method: valid holes, mean CD, sigma and 3 sigma % = 3 sigma / mean x 100, over the holes that method measured
validly) and one tab per image with its statistics, CD histograms and the image with every detected hole drawn
at its CD (drag to pan, wheel to zoom, click a hole for its CDs under every method). The same statistics go to
REPORT/summary.csv. --save-results also keeps the full per-hole results in REPORT/results/<image>/
(``subpx.load_cd_results``; about 1 GB per 1e5-hole image with all methods).

disk and flux_ratio need --nominal-cd or --psf-fwhm; without either they are skipped.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx import measure_cds, save_cd_results
from subpx.cd import METHOD_LABELS

ALL_METHODS = ("gaussian", "logquad", "gradient", "halfmax", "disk", "flux_ratio", "projection")
DESCRIPTIONS = {
    "gaussian": "FWHM of a 2-D Gaussian fitted to the spot (deblended). Describes the blurred spot, not the hole.",
    "logquad": "FWHM of a Gaussian fitted to log intensity. Describes the blurred spot.",
    "gradient": "Contour through the steepest intensity drop along each ray. Spot edge.",
    "halfmax": "Contour at half the spot's peak height. Spot edge.",
    "disk": "Diameter of a round hole that, blurred by the optics and pixel-averaged, matches the spot. The hole itself; "
            "unaffected by the per-hole pulse strength. Absolute value follows --nominal-cd (or --psf-fwhm).",
    "flux_ratio": "Hole light divided by its own background, converted to a diameter and scaled so the median equals "
                  "--nominal-cd. Cancels the pulse when the background is pulse-lit absorber light (set --dark-level).",
    "projection": "Mean of the x and y widths between the steepest rise and fall of the cell's projected profiles.",
}


def cd_statistics(result) -> dict[str, dict]:
    """Per method: cells, valid holes, mean, sigma and 3 sigma % of the valid holes' CDs (in result.unit)."""
    stats = {}
    for name, measured in result.measurements.items():
        values = measured.metrics["cd_equivalent_px"][measured.valid] * result.pixel_size
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if values.size else float("nan")
        std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        stats[name] = {"cells": int(len(measured.valid)), "valid": int(values.size), "mean": mean, "std": std,
                       "three_sigma_pct": 300 * std / mean if values.size > 1 and mean else float("nan")}
    return stats


def display_image(image: np.ndarray) -> np.ndarray:
    """8-bit display copy, contrast-stretched between the 0.5 and 99.5 percentiles (measurements use the original)."""
    lo, hi = np.percentile(image, [0.5, 99.5])
    return np.clip(np.rint((image.astype(float) - lo) * (255 / max(hi - lo, 1e-12))), 0, 255).astype(np.uint8)


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def hole_data(result) -> dict:
    """Per method: hole centers, CD (px), validity and status, as base64 arrays for the viewer."""
    seeds = result.seeds_xy
    methods = {}
    for name, measured in result.measurements.items():
        centers = np.where(np.isfinite(measured.centers_xy), measured.centers_xy, seeds)
        statuses, codes = np.unique(measured.status, return_inverse=True)
        methods[name] = {
            "label": METHOD_LABELS[name], "x": _b64(centers[:, 0], "<f4"), "y": _b64(centers[:, 1], "<f4"),
            "cd": _b64(np.where(measured.valid, measured.metrics["cd_equivalent_px"], np.nan), "<f4"),
            "valid": _b64(measured.valid, "u1"), "status": _b64(codes, "u1" if len(statuses) < 256 else "<u2"),
            "status_names": statuses.tolist(), "status_wide": len(statuses) >= 256,
        }
    return {"n": int(len(seeds)), "width": int(result.image.shape[1]), "height": int(result.image.shape[0]),
            "scale": float(result.pixel_size), "unit": result.unit, "methods": methods}


def _fmt(value: float, digits: int = 2) -> str:
    return "–" if not np.isfinite(value) else f"{value:.{digits}f}"


def _stats_table(stats: dict, unit: str) -> str:
    rows = "".join(
        f"<tr><td><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(METHOD_LABELS[m])} "
        f"<code>{m}</code></td><td>{s['valid']:,} / {s['cells']:,}</td><td>{_fmt(s['mean'])}</td>"
        f"<td>{_fmt(s['std'])}</td><td><b>{_fmt(s['three_sigma_pct'])}</b></td></tr>" for m, s in stats.items())
    return (f"<table class='stats'><thead><tr><th>Method</th><th>Valid holes</th><th>Mean CD ({html.escape(unit)})</th>"
            f"<th>σ ({html.escape(unit)})</th><th>3σ (%)</th></tr></thead><tbody>{rows}</tbody></table>")


COLORS = {"gaussian": "#5bbf2a", "logquad": "#e09a00", "gradient": "#00a9c4", "halfmax": "#e0468f",
          "disk": "#e03b3b", "flux_ratio": "#8f6ad8", "projection": "#7a7a7a"}


def write_report(output: Path, entries: list[dict], methods: tuple[str, ...], notes: list[str], unit: str) -> Path:
    """index.html: summary tab, one tab per image (data loaded on demand from data/<id>.js)."""
    def table(title, key):
        head = "".join(f"<th><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(m)}</th>"
                       for m in methods)
        body = []
        for e in entries:
            if e.get("error"):
                body.append(f"<tr><td>{html.escape(e['name'])}</td><td colspan='{1 + len(methods)}' class='error'>"
                            f"failed: {html.escape(e['error'])}</td></tr>")
                continue
            cells = "".join(f"<td title='{e['stats'][m]['valid']:,} valid holes'>{_fmt(e['stats'][m][key])}</td>"
                            if m in e["stats"] else "<td>–</td>" for m in methods)
            body.append(f"<tr><td><a href='#' data-tab='{e['id']}'>{html.escape(e['name'])}</a></td>"
                        f"<td>{e['holes']:,}</td>{cells}</tr>")
        return (f"<h3>{title}</h3><table class='stats wide'><thead><tr><th>Image</th><th>Holes</th>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>")

    summary = table(f"Mean CD ({html.escape(unit)})", "mean") + table("3σ (%)", "three_sigma_pct")
    legend = "".join(f"<li><span class='swatch' style='background:{COLORS[m]}'></span><b>{m}</b> — "
                     f"{html.escape(DESCRIPTIONS[m])}</li>" for m in methods)
    tabs = "".join(f"<button class='tab' data-tab='{e['id']}'{' disabled' if e.get('error') else ''}>"
                   f"{html.escape(e['name'])}</button>" for e in entries)
    panels = "".join(
        f"<section class='panel' id='panel-{e['id']}' hidden><h2>{html.escape(e['name'])}</h2>"
        f"<p class='meta'>{e['holes']:,} holes · {e['width']} × {e['height']} px · {e['seconds']:.1f} s"
        f"{html.escape(e['extra'])}</p>{_stats_table(e['stats'], unit)}<div class='hists'></div></section>"
        for e in entries if not e.get("error"))
    page = (_HTML.replace("__SUMMARY__", summary).replace("__LEGEND__", legend).replace("__TABS__", tabs)
            .replace("__PANELS__", panels)
            .replace("__NOTES__", "".join(f"<p class='note'>{html.escape(n)}</p>" for n in notes))
            .replace("__ENTRIES__", json.dumps({e["id"]: {"image": e["image"], "data": e["data"]}
                                                for e in entries if not e.get("error")}))
            .replace("__COLORS__", json.dumps(COLORS)))
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main(argv=None) -> None:
    from PIL import Image
    from scripts.analyze_contact_holes import load_image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder holding the .png images")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cd_report")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--nominal-cd", type=float, default=None, help="Typical hole diameter in --unit (disk, flux_ratio)")
    parser.add_argument("--psf-fwhm", type=float, default=None, help="Known blur FWHM in --unit (overrides --nominal-cd)")
    parser.add_argument("--dark-level", type=float, default=0.0, help="Detector offset removed before flux_ratio divides")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold",
                        help="peaks: local maxima, for arrays too dense to threshold")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--save-results", action="store_true",
                        help="Also save full per-hole results (compact NPZ + CSVs, ~1 GB per 1e5-hole image)")
    args = parser.parse_args(argv)

    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".png") if args.folder.is_dir() else []
    if not images:
        raise SystemExit(f"No .png images in {args.folder}")
    methods, notes = tuple(args.methods), []
    if args.nominal_cd is None and args.psf_fwhm is None and {"disk", "flux_ratio"} & set(methods):
        notes.append("disk and flux_ratio were skipped: they need --nominal-cd or --psf-fwhm.")
        print(notes[-1], flush=True)
        methods = tuple(m for m in methods if m not in ("disk", "flux_ratio"))
    output = args.output
    for sub in ("images", "data"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    entries, rows, used = [], [], set()
    started = time.perf_counter()
    for number, path in enumerate(images, 1):
        ident = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem) or "image"
        while ident in used:
            ident += "_"
        used.add(ident)
        print(f"[{number}/{len(images)}] {path.name}", flush=True)
        start = time.perf_counter()
        try:
            image = load_image(path)
            result = measure_cds(
                image, methods=methods, seeds=args.seeds, invert=args.invert, area_min=args.area_min,
                area_max=args.area_max, pixel_size=args.pixel_size, unit=args.unit, backend=args.backend,
                nominal_cd=args.nominal_cd, psf_fwhm=args.psf_fwhm, dark_level=args.dark_level,
            )
        except Exception as error:  # one bad image must not stop the batch
            message = str(error).replace(str(path), path.name) or type(error).__name__
            print(f"  failed: {message}", flush=True)
            entries.append({"id": ident, "name": path.name, "error": message})
            rows.append({"image": path.name, "status": "failed", "error": message})
            continue
        seconds = time.perf_counter() - start
        stats = cd_statistics(result)
        Image.fromarray(display_image(image)).save(output / "images" / f"{ident}.png")
        (output / "data" / f"{ident}.js").write_text(
            f"window.CD_DATA[{json.dumps(ident)}]=" + json.dumps(hole_data(result), allow_nan=False) + ";\n",
            encoding="utf-8")
        if args.save_results:
            save_cd_results(result, output / "results" / ident, compact=True)
        extra = ""
        if "psf_fwhm_px" in result.meta:
            extra = f" · blur FWHM {result.meta['psf_fwhm_px'] * args.pixel_size:.3g} {args.unit} ({result.meta['psf_source']})"
        entries.append({"id": ident, "name": path.name, "holes": len(result.seeds_xy), "stats": stats,
                        "width": image.shape[1], "height": image.shape[0], "seconds": seconds, "extra": extra,
                        "image": f"images/{ident}.png", "data": f"data/{ident}.js"})
        for method, s in stats.items():
            rows.append({"image": path.name, "status": "ok", "method": method, "holes": len(result.seeds_xy), **s,
                         "unit": args.unit})
            print(f"  {method:11s} {s['valid']:7d} valid  mean {_fmt(s['mean'], 3)} {args.unit}  "
                  f"3sigma {_fmt(s['three_sigma_pct'])} %", flush=True)
        print(f"  {seconds:.1f} s", flush=True)
    fields = ["image", "status", "method", "holes", "cells", "valid", "mean", "std", "three_sigma_pct", "unit", "error"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path = write_report(output, entries, methods, notes, args.unit)
    print(f"Report: {path} ({time.perf_counter() - started:.1f} s)", flush=True)


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CD Report</title>
<style>
:root{--bg:#f6f7f9;--fg:#1d2330;--muted:#5d6675;--card:#fff;--line:#d9dde3;--accent:#2f6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--fg:#e4e7ec;--muted:#9aa3b2;--card:#1e222a;--line:#343a46;--accent:#6e9ef0}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:14px 20px 0}h1{font-size:20px;margin:0 0 10px}h2{font-size:17px;margin:4px 0}
nav{display:flex;flex-wrap:wrap;gap:4px;padding:0 20px;border-bottom:1px solid var(--line)}
.tab{border:1px solid var(--line);border-bottom:none;background:transparent;color:var(--fg);padding:7px 12px;border-radius:6px 6px 0 0;cursor:pointer;font:inherit}
.tab.active{background:var(--card);font-weight:600;border-color:var(--accent)}.tab:disabled{opacity:.45;cursor:default}
main{padding:16px 20px}.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
table.stats{border-collapse:collapse;margin:8px 0}table.stats th,table.stats td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:right;white-space:nowrap}
table.stats th:first-child,table.stats td:first-child{text-align:left}table.wide{display:block;overflow-x:auto;max-width:100%}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:baseline}
.meta,.note{color:var(--muted);margin:2px 0 6px}.error{color:#d33;text-align:left!important;white-space:normal!important;max-width:60ch}
h3{font-size:15px;margin:14px 0 2px}code{font-size:12px;color:var(--muted)}
ul.legend{padding-left:18px;color:var(--muted)}ul.legend b{color:var(--fg)}
.hists{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.hist{border:1px solid var(--line);border-radius:6px;padding:4px}
.hist div{font-size:12px;color:var(--muted)}a{color:var(--accent)}
#viewer{display:none}#viewer.on{display:block}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px}
.controls select,.controls button{font:inherit;padding:3px 8px}
#stage{position:relative;height:70vh;min-height:380px;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#000;cursor:grab}
#stage canvas{width:100%;height:100%;display:block}#bar{display:flex;align-items:center;gap:6px;font-size:12px}
#ramp{width:180px;height:10px;border-radius:2px}#info{font-size:13px;margin-top:8px;min-height:20px}
#info table{border-collapse:collapse}#info td{padding:1px 10px 1px 0}
</style></head><body>
<header><h1>CD report</h1></header>
<nav><button class="tab active" data-tab="summary">Summary</button>__TABS__</nav>
<main>
<section class="panel" id="panel-summary"><h2>Mean CD and 3σ per image</h2>
<p class="meta">3σ % = 3 × standard deviation / mean × 100, over the holes each method measured validly (hover a value for the count). Click an image name to open it.</p>
__NOTES__ __SUMMARY__<h2>Methods</h2><ul class="legend">__LEGEND__</ul></section>
__PANELS__
<section class="panel" id="viewer"><div class="controls">
<label>Method <select id="method"></select></label><button id="fit">Fit image</button>
<label><input type="checkbox" id="show" checked> Show holes</label>
<span id="bar"><span id="lo"></span><span id="ramp"></span><span id="hi"></span></span>
<span class="meta">Drag to pan · wheel to zoom · click a hole</span></div>
<div id="stage"><canvas id="canvas"></canvas></div><div id="info"></div></section>
</main>
<script>
window.CD_DATA={};
const ENTRIES=__ENTRIES__, COLORS=__COLORS__, $=id=>document.getElementById(id);
const VIRIDIS=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
function viridis(t){t=Math.min(1,Math.max(0,t))*9;const i=Math.min(8,Math.floor(t)),f=t-i,a=VIRIDIS[i],b=VIRIDIS[i+1];
  return `rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(",")})`;}
function unpack(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer);}
function quantile(sorted,q){return sorted.length?sorted[Math.min(sorted.length-1,Math.floor(q*(sorted.length-1)))]:NaN;}
const canvas=$("canvas"),ctx=canvas.getContext("2d");
let cur=null,img=null,view={k:1,tx:0,ty:0},sel=-1,bins=null,range=[0,1];
function prepare(d){if(d.ready)return;for(const m of Object.values(d.methods)){
  m.x=unpack(m.x,Float32Array);m.y=unpack(m.y,Float32Array);m.cd=unpack(m.cd,Float32Array);
  m.valid=unpack(m.valid,Uint8Array);m.status=unpack(m.status,m.status_wide?Uint16Array:Uint8Array);
  m.sorted=Float32Array.from(m.cd.filter(Number.isFinite)).sort();}d.ready=true;}
function show(id){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.tab===id));
  document.querySelectorAll(".panel").forEach(p=>p.hidden=p.id!=="panel-"+id&&p.id!=="viewer");
  $("viewer").classList.toggle("on",id!=="summary");if(id==="summary"){cur=null;return;}
  const go=()=>{const d=window.CD_DATA[id];prepare(d);cur=d;sel=-1;$("info").textContent="";
    const keep=$("method").value;$("method").innerHTML="";
    for(const [k,m] of Object.entries(d.methods)){const o=document.createElement("option");o.value=k;o.textContent=k+" — "+m.label;$("method").appendChild(o);}
    if(keep&&d.methods[keep])$("method").value=keep;histograms(id,d);
    img=new Image();img.onload=()=>{fit();setMethod();};img.src=ENTRIES[id].image;};
  if(window.CD_DATA[id])go();else{$("info").textContent="Loading…";const s=document.createElement("script");s.src=ENTRIES[id].data;s.onload=go;
    s.onerror=()=>$("info").textContent="Could not load "+ENTRIES[id].data;document.body.appendChild(s);}}
function histograms(id,d){const box=document.querySelector(`#panel-${id} .hists`);if(box.dataset.done)return;box.dataset.done=1;
  for(const [k,m] of Object.entries(d.methods)){const v=m.sorted;if(!v.length)continue;
    const lo=quantile(v,.005),hi=quantile(v,.995),n=40,h=new Array(n).fill(0),w=(hi-lo)/n||1;
    for(const x of v){const i=Math.floor((x-lo)/w);if(i>=0&&i<n)h[i]++;}
    const wrap=document.createElement("div");wrap.className="hist";const c=document.createElement("canvas");c.width=240;c.height=110;
    const label=document.createElement("div");label.textContent=`${k} (${d.unit})`;wrap.append(label,c);box.appendChild(wrap);
    const g=c.getContext("2d"),top=Math.max(...h);g.fillStyle=COLORS[k]||"#888";
    h.forEach((y,i)=>g.fillRect(i*6,96-y/top*90,5,y/top*90));g.fillStyle=getComputedStyle(document.body).color;g.font="10px sans-serif";
    g.fillText((lo*d.scale).toPrecision(4),0,108);const t=(hi*d.scale).toPrecision(4);g.fillText(t,240-g.measureText(t).width,108);}}
function setMethod(){if(!cur)return;const m=cur.methods[$("method").value],v=m.sorted;
  range=[quantile(v,.02),quantile(v,.98)];if(!(range[1]>range[0]))range=[range[0]-0.5,range[0]+0.5];
  // Colour bins: one fillStyle per bin keeps 1e5 holes fast to draw.
  bins=Array.from({length:32},()=>[]);for(let i=0;i<cur.n;i++)if(m.valid[i]){
    const b=Math.min(31,Math.max(0,Math.floor((m.cd[i]-range[0])/(range[1]-range[0])*32)));bins[b].push(i);}
  $("lo").textContent=(range[0]*cur.scale).toPrecision(4);$("hi").textContent=(range[1]*cur.scale).toPrecision(4)+" "+cur.unit;
  $("ramp").style.background="linear-gradient(to right,"+[0,.25,.5,.75,1].map(viridis).join(",")+")";draw();}
function fit(){const r=canvas.getBoundingClientRect();canvas.width=r.width*devicePixelRatio;canvas.height=r.height*devicePixelRatio;
  const k=Math.min(canvas.width/cur.width,canvas.height/cur.height);view={k,tx:(canvas.width-k*cur.width)/2,ty:(canvas.height-k*cur.height)/2};draw();}
function draw(){if(!cur||!img)return;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle="#000";ctx.fillRect(0,0,canvas.width,canvas.height);
  const {k,tx,ty}=view;ctx.setTransform(k,0,0,k,tx,ty);ctx.imageSmoothingEnabled=k<2;ctx.drawImage(img,0,0);
  if(!$("show").checked||!bins)return;const m=cur.methods[$("method").value];
  const x0=-tx/k-20,y0=-ty/k-20,x1=(canvas.width-tx)/k+20,y1=(canvas.height-ty)/k+20,dots=k*4<6;
  ctx.lineWidth=1.5/k;
  bins.forEach((list,b)=>{const col=viridis((b+.5)/32);ctx.fillStyle=col;ctx.strokeStyle=col;if(!dots)ctx.beginPath();
    for(const i of list){const x=m.x[i]+.5,y=m.y[i]+.5;if(x<x0||x>x1||y<y0||y>y1)continue;
      if(dots){const s=Math.max(1.6/k,Math.min(m.cd[i],3));ctx.fillRect(x-s/2,y-s/2,s,s);}
      else{ctx.moveTo(x+m.cd[i]/2,y);ctx.arc(x,y,m.cd[i]/2,0,2*Math.PI);}}
    if(!dots)ctx.stroke();});
  if(!dots){ctx.strokeStyle="#9a9a9a";ctx.beginPath();const c=3;
    for(let i=0;i<cur.n;i++){if(m.valid[i])continue;const x=m.x[i]+.5,y=m.y[i]+.5;if(x<x0||x>x1||y<y0||y>y1)continue;
      ctx.moveTo(x-c,y-c);ctx.lineTo(x+c,y+c);ctx.moveTo(x-c,y+c);ctx.lineTo(x+c,y-c);}ctx.stroke();}
  if(sel>=0){const x=m.x[sel]+.5,y=m.y[sel]+.5;ctx.strokeStyle="#ff3b3b";ctx.lineWidth=2.5/k;ctx.beginPath();
    ctx.arc(x,y,Math.max(Number.isFinite(m.cd[sel])?m.cd[sel]/2+1:4,8/k),0,2*Math.PI);ctx.stroke();}}
function pick(px,py){const m=cur.methods[$("method").value],x=(px-view.tx)/view.k-.5,y=(py-view.ty)/view.k-.5;
  let best=-1,bd=Infinity;for(let i=0;i<cur.n;i++){const d=(m.x[i]-x)**2+(m.y[i]-y)**2;if(d<bd){bd=d;best=i;}}
  if(best<0||bd>36)return;sel=best;let rows="";
  for(const [k,mm] of Object.entries(cur.methods)){const v=mm.cd[best];
    rows+=`<tr><td><span class="swatch" style="background:${COLORS[k]}"></span>${k}</td><td>${Number.isFinite(v)?(v*cur.scale).toFixed(3)+" "+cur.unit:"–"}</td><td>${mm.status_names[mm.status[best]]}</td></tr>`;}
  $("info").innerHTML=`<b>Hole ${best}</b> at x ${m.x[best].toFixed(2)}, y ${m.y[best].toFixed(2)} px<table>${rows}</table>`;draw();}
let drag=null;
canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,tx:view.tx,ty:view.ty,moved:false};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener("pointermove",e=>{if(!drag)return;const dx=(e.clientX-drag.x)*devicePixelRatio,dy=(e.clientY-drag.y)*devicePixelRatio;
  if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;view.tx=drag.tx+dx;view.ty=drag.ty+dy;draw();});
canvas.addEventListener("pointerup",e=>{if(drag&&!drag.moved){const r=canvas.getBoundingClientRect();pick((e.clientX-r.left)*devicePixelRatio,(e.clientY-r.top)*devicePixelRatio);}drag=null;});
canvas.addEventListener("wheel",e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),px=(e.clientX-r.left)*devicePixelRatio,py=(e.clientY-r.top)*devicePixelRatio;
  const f=Math.exp(-e.deltaY*0.0015),k=Math.min(80,Math.max(0.02,view.k*f)),g=k/view.k;view.tx=px-(px-view.tx)*g;view.ty=py-(py-view.ty)*g;view.k=k;draw();},{passive:false});
$("method").onchange=setMethod;$("show").onchange=draw;$("fit").onclick=fit;window.addEventListener("resize",()=>{if(cur)fit();});
document.querySelectorAll("[data-tab]").forEach(t=>t.addEventListener("click",e=>{e.preventDefault();show(t.dataset.tab);window.scrollTo(0,0);}));
</script></body></html>
"""


if __name__ == "__main__":
    main()
