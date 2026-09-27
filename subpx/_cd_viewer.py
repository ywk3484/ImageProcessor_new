"""Self-contained canvas viewer for measured contour paths and cell maps."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .types import CDResult


def _packed(array, dtype: str) -> str:
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def write_viewer(result: CDResult, path, *, metric: str) -> Path:
    from matplotlib import colormaps
    from .cd import METHOD_LABELS

    if metric not in {"cd_equivalent_px", "cd_x_px", "cd_y_px", "area_px2"}:
        raise ValueError("unsupported CD map metric")
    lo, hi = float(result.image.min()), float(result.image.max())
    display = np.zeros(result.image.shape, np.uint8)
    if hi > lo:
        display = np.rint((result.image.astype(float) - lo) * (255 / (hi - lo))).astype(np.uint8)
    palette = np.rint(colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    payload = {
        "shape": list(result.image.shape), "count": len(result.seeds_xy),
        "pixel_size": result.pixel_size, "unit": result.unit, "metric": metric,
        "image": _packed(display, "u1"), "display_range": [lo, hi],
        "labels": _packed(result.cell_labels, "<i4"),
        "means": _packed(result.mean_intensity, "<f8"),
        "seeds": _packed(result.seeds_xy, "<f8"),
        "origins": _packed(result.origins_xy, "<i4"),
        "areas": _packed(result.cell_areas_px2, "<i4"),
        "roi_shape": list(result.rois.shape[1:]),
        "palette": palette.tolist(), "methods": {},
        "radial_step": result.meta.get("radial_step"),
        "smooth_sigma": result.meta.get("smooth_sigma"),
    }
    for name, measured in result.measurements.items():
        # Local offsets retain subpixel precision at large global coordinates
        # without a large JSON list of individual floating-point vertices.
        local = measured.contours_xy - result.origins_xy[:, None, :]
        payload["methods"][name] = {
            "label": METHOD_LABELS[name], "vertices": local.shape[1],
            "points": _packed(local, "<f4"), "valid": _packed(measured.valid, "u1"),
            "status": measured.status.tolist(),
            "metrics": {key: _packed(values, "<f8") for key, values in measured.metrics.items()},
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escape '<' so user-supplied unit strings cannot terminate a script tag.
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    before, after = _HTML.split("__CD_DATA__")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(before)
        handle.write(data)
        handle.write(after)
    return path


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contact-hole contours and cell maps</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#192b3c;font:14px system-ui,sans-serif}
header{padding:18px 24px;background:#142c3e;color:#fff;display:flex;align-items:baseline;gap:20px;flex-wrap:wrap}
h1{font-size:21px;font-weight:600;margin:0}header p{margin:0;color:#c4d3df;font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:end;padding:14px 24px;border-bottom:1px solid #d6dfe5;background:#fff}
label{display:flex;flex-direction:column;gap:5px;color:#42586c;font-size:12px;font-weight:600}
select,input[type=number],button{font:14px system-ui;border:1px solid #becbd5;border-radius:5px;padding:7px 10px;background:white;color:#192b3c}
button{cursor:pointer}button:hover{background:#e7eff5}button:disabled{opacity:.45;cursor:default}
label.check{flex-direction:row;align-items:center;align-self:center;margin-top:14px;gap:6px;font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1fr) 285px;gap:16px;padding:16px 24px}
.plot{min-width:0;background:#fff;border:1px solid #d6dfe5;border-radius:8px;overflow:hidden}
.plotbar{padding:10px 14px;border-bottom:1px solid #d6dfe5;display:flex;justify-content:space-between;gap:14px;align-items:center}
#zoom-readout{font-variant-numeric:tabular-nums;color:#526c80;font-size:12px;margin-left:8px}
#stage{height:clamp(320px,65vh,900px);position:relative;background:#080c11;overflow:hidden}
#scene{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
#scene:active{cursor:grabbing}#tooltip{position:absolute;pointer-events:none;white-space:pre-line;padding:9px 12px;background:#fffef3;color:#213446;box-shadow:0 2px 12px #0005;border-radius:5px;font:12px/1.6 system-ui;display:none}
.legend{padding:12px 16px;min-height:64px}.ramp{height:10px;border-radius:3px;margin:7px 0 3px}.ticks{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums;font-size:12px}
#method-legend{display:flex;flex-wrap:wrap;gap:18px;font-size:13px;padding:10px 0}.swatch{display:inline-block;width:20px;height:3px;margin:0 6px 3px 0}
.help{padding:10px 16px;border-top:1px solid #e2e9ee;color:#536b7c;font-size:12px;line-height:1.6}
aside{background:white;border:1px solid #d6dfe5;border-radius:8px;padding:16px;align-self:start}
h2{font-size:16px;margin:0 0 12px}.find{display:flex;gap:8px}.find input{width:94px}#cell-meta{line-height:1.7;margin:12px 0;color:#526a7c;font-size:13px;white-space:pre-line}
#focus{width:100%;margin-bottom:10px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:7px 2px;border-bottom:1px solid #e2e9ee;vertical-align:top}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.note{color:#526a7c;font-size:12px;line-height:1.65;margin:14px 0 0}.error{color:#b02d30}
@media(max-width:900px){main{grid-template-columns:1fr;padding:12px}.controls{padding:12px}header{padding:14px}#stage{height:55vh}}
</style>
</head>
<body>
<header><h1>Contact-hole contours &amp; cell maps</h1><p id="summary">Loading measured contours…</p></header>
<div class="controls">
<label>View<select id="view"><option value="contours">Contour overlay</option><option value="cd">CD cell map</option><option value="intensity">Mean intensity map</option></select></label>
<label>Method<select id="method"></select></label>
<label>CD metric<select id="metric"><option value="cd_equivalent_px">Area-equivalent CD</option><option value="cd_x_px">Horizontal CD</option><option value="cd_y_px">Vertical CD</option><option value="area_px2">Contour area</option></select></label>
<label class="check"><input type="checkbox" id="compare">Compare all contours</label>
<label class="check"><input type="checkbox" id="vertices">Show measured vertices</label>
<label class="check"><input type="checkbox" id="smooth">Smooth background display</label>
</div>
<main>
<section class="plot">
<div class="plotbar"><strong id="plot-title">Contour overlay</strong><div><button id="zoom-out" aria-label="Zoom out">−</button> <button id="zoom-in" aria-label="Zoom in">+</button> <button id="fit">Fit image</button><span id="zoom-readout"></span></div></div>
<div id="stage"><canvas id="scene" tabindex="0" aria-label="Interactive contour map. Drag to pan, scroll to zoom, or use the zoom and cell selection buttons."></canvas><div id="tooltip"></div></div>
<div class="legend"><div id="color-legend"><span id="legend-title"></span><div class="ramp" id="ramp"></div><div class="ticks"><span id="low"></span><span id="high"></span></div></div><div id="method-legend" hidden></div></div>
<div class="help">Scroll to zoom at the pointer · Drag to pan · Click to select a cell · Double-click to inspect it<br>Contours redraw at every zoom. Image pixels retain their original resolution. Coordinates are x, y in pixels.</div>
</section>
<aside>
<h2>Inspect a cell</h2><div class="find"><input id="cell-id" aria-label="Cell ID" type="number" min="0" step="1" placeholder="Cell ID"><button id="select-cell">Select</button></div>
<div id="cell-meta" aria-live="polite">Click a cell or enter its zero-based ID.</div><button id="focus" disabled>Zoom to selected cell</button>
<table id="measurements"><thead><tr><th>Method / metric</th><th>Value</th></tr></thead><tbody></tbody></table>
<p class="note">Mean intensity includes all original pixels in the full Voronoi cell, including background. Padding, polarity inversion, and physical calibration do not change it.</p>
<p class="note" id="sampling-note"></p><p class="note">Gaussian FWHM, radial gradient maxima, and half-height crossings define different edges. Inspect them together using “Compare all contours”.</p>
</aside>
</main>
<script id="cd-data" type="application/json">__CD_DATA__</script>
<script>
"use strict";
(() => {
const dataNode = document.getElementById("cd-data");
const data = JSON.parse(dataNode.textContent); dataNode.remove();
const $ = id => document.getElementById(id);
function unpack(text, Type) {
  const raw = atob(text), bytes = new Uint8Array(raw.length);
  for (let i=0;i<raw.length;i++) bytes[i] = raw.charCodeAt(i);
  return new Type(bytes.buffer);
}
const [H,W] = data.shape, N=data.count;
const labels=unpack(data.labels,Int32Array), means=unpack(data.means,Float64Array);
const seeds=unpack(data.seeds,Float64Array), origins=unpack(data.origins,Int32Array);
const areas=unpack(data.areas,Int32Array), gray=unpack(data.image,Uint8Array);
delete data.labels; delete data.means; delete data.seeds; delete data.origins; delete data.areas; delete data.image;
const names=Object.keys(data.methods), methods=data.methods;
const methodColors={logquad:"#ffb000",gradient:"#00cfec",halfmax:"#ff5aa5"};
const metricLabels={cd_equivalent_px:"Area-equivalent CD",cd_x_px:"Horizontal CD",cd_y_px:"Vertical CD",area_px2:"Contour area"};
for(const name of names){
  const m=methods[name]; m.points=unpack(m.points,Float32Array); m.valid=unpack(m.valid,Uint8Array);
  for(const key of Object.keys(m.metrics))m.metrics[key]=unpack(m.metrics[key],Float64Array);
  const option=document.createElement("option");option.value=name;option.textContent=m.label;$("method").appendChild(option);
  const item=document.createElement("span"), swatch=document.createElement("span");
  swatch.className="swatch";swatch.style.background=methodColors[name];item.append(swatch,document.createTextNode(m.label));$("method-legend").appendChild(item);
}
$("metric").value=data.metric;
$("summary").textContent=`${N.toLocaleString()} cells · ${W} × ${H} pixels · offline viewer`;
$("cell-id").max=Math.max(N-1,0);
$("sampling-note").textContent=`Gradient edges use cubic-interpolated radial profiles, sampled at ${data.radial_step} px. Source smoothing σ = ${data.smooth_sigma} px. The display-smoothing toggle changes only the background image.`;
$("ramp").style.background="linear-gradient(to right,"+data.palette.filter((_,i)=>i%16===0||i===255).map(c=>`rgb(${c.join(",")})`).join(",")+")";
const image=document.createElement("canvas");image.width=W;image.height=H;
const imageContext=image.getContext("2d"), rgba=imageContext.createImageData(W,H);
for(let i=0;i<gray.length;i++){rgba.data[i*4]=rgba.data[i*4+1]=rgba.data[i*4+2]=gray[i];rgba.data[i*4+3]=255;}
imageContext.putImageData(rgba,0,0);
const heat=document.createElement("canvas");heat.width=W;heat.height=H;
const heatContext=heat.getContext("2d");
const canvas=$("scene"), ctx=canvas.getContext("2d"), stage=$("stage");
let width=1,height=1,dpr=1,selected=-1,queued=false,initialized=false;
let view={x:(W-1)/2,y:(H-1)/2,k:1}, values=means, limits=[0,1], colors=[], heatKey="";
function fmt(v){return Number.isFinite(v)?v.toFixed(4):"—";}
function mode(){return $("view").value;}
function current(){return methods[$("method").value];}
function scale(){return data.pixel_size**($("metric").value==="area_px2"?2:1);}
function unit(){return data.unit+($("metric").value==="area_px2"?"²":"");}
function valueRange(v,valid){
  let low=Infinity,high=-Infinity;
  for(let i=0;i<N;i++)if((!valid||valid[i])&&Number.isFinite(v[i])){low=Math.min(low,v[i]);high=Math.max(high,v[i]);}
  if(low===Infinity)return [0,1];
  if(high===low){const delta=Math.max(Math.abs(low)*.01,.01);return [low-delta,high+delta];}
  return [low,high];
}
function paletteIndex(value){return Math.max(0,Math.min(255,Math.round(255*(value-limits[0])/(limits[1]-limits[0]))));}
function updateColors(){
  const m=current(), intensity=mode()==="intensity";
  values=intensity?means:m.metrics[$("metric").value];
  limits=valueRange(values,intensity?null:m.valid);
  colors=Array.from({length:N},(_,i)=>!Number.isFinite(values[i])||(!intensity&&!m.valid[i])?"#aaaaaa":`rgb(${data.palette[paletteIndex(values[i])].join(",")})`);
  const factor=intensity?1:scale();
  $("low").textContent=fmt(limits[0]*factor);$("high").textContent=fmt(limits[1]*factor);
  $("legend-title").textContent=intensity?"Mean cell intensity (original intensity units)":metricLabels[$("metric").value]+" ("+unit()+")";
  const compare=mode()==="contours"&&$("compare").checked;
  $("color-legend").hidden=compare;$("method-legend").hidden=!compare;
  $("method-legend").style.display=compare?"flex":"none";
  $("method").disabled=intensity;$("metric").disabled=intensity;
  $("compare").disabled=mode()!=="contours";$("vertices").disabled=mode()!=="contours";
  $("smooth").disabled=mode()!=="contours";
  $("plot-title").textContent=intensity?"Mean intensity by cell":mode()==="cd"?m.label+" · CD map":compare?"All measured contours":m.label+" · contours";
  const key=mode()+":"+$("method").value+":"+$("metric").value;
  if(mode()!=="contours"&&key!==heatKey){
    const pixels=heatContext.createImageData(W,H);
    for(let p=0;p<labels.length;p++){
      const i=labels[p], good=i>=0&&Number.isFinite(values[i])&&(intensity||m.valid[i]);
      const color=good?data.palette[paletteIndex(values[i])]:[170,170,170];
      pixels.data[p*4]=color[0];pixels.data[p*4+1]=color[1];pixels.data[p*4+2]=color[2];pixels.data[p*4+3]=255;
    }
    heatContext.putImageData(pixels,0,0);heatKey=key;
  }
}
function requestDraw(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;draw();});}}
function toWorld(sx,sy){return {x:view.x+(sx-width/2)/view.k,y:view.y+(sy-height/2)/view.k};}
function drawContours(name,compare){
  const m=methods[name], p=m.points, count=m.vertices;
  const a=toWorld(0,0),b=toWorld(width,height);
  for(let i=0;i<N;i++){
    if(!m.valid[i])continue;
    const ox=origins[2*i],oy=origins[2*i+1];
    if(ox+data.roi_shape[1]<a.x||oy+data.roi_shape[0]<a.y||ox>b.x||oy>b.y)continue;
    const start=i*count*2;
    ctx.strokeStyle=compare?methodColors[name]:colors[i];ctx.lineWidth=(i===selected?2:1.15)/view.k;
    ctx.beginPath();
    for(let j=0;j<count;j++){
      const x=ox+p[start+j*2],y=oy+p[start+j*2+1];
      if(j===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
    if($("vertices").checked&&view.k>=12){
      ctx.fillStyle=ctx.strokeStyle;
      for(let j=0;j<count-1;j++)ctx.fillRect(ox+p[start+j*2]-1.4/view.k,oy+p[start+j*2+1]-1.4/view.k,2.8/view.k,2.8/view.k);
    }
  }
}
function draw(){
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle="#080c11";ctx.fillRect(0,0,width,height);
  ctx.translate(width/2-view.x*view.k,height/2-view.y*view.k);ctx.scale(view.k,view.k);
  ctx.imageSmoothingEnabled=mode()==="contours"&&$("smooth").checked;
  ctx.drawImage(mode()==="contours"?image:heat,-.5,-.5,W,H);
  if(mode()==="contours"){
    const compare=$("compare").checked;
    for(const name of compare?names:[$("method").value])drawContours(name,compare);
    const m=current();ctx.strokeStyle="#aaa";ctx.lineWidth=1/view.k;
    for(let i=0;i<N;i++)if(!m.valid[i]){
      const x=seeds[i*2],y=seeds[i*2+1],r=3/view.k;
      ctx.beginPath();ctx.moveTo(x-r,y-r);ctx.lineTo(x+r,y+r);ctx.moveTo(x-r,y+r);ctx.lineTo(x+r,y-r);ctx.stroke();
    }
  }
  if(selected>=0){
    ctx.strokeStyle="white";ctx.lineWidth=1/view.k;ctx.setLineDash([4/view.k,3/view.k]);
    ctx.strokeRect(origins[selected*2]-.5,origins[selected*2+1]-.5,data.roi_shape[1],data.roi_shape[0]);ctx.setLineDash([]);
  }
  $("zoom-readout").textContent=`${view.k.toFixed(1)} screen px / image px`;
}
function fit(){view={x:(W-1)/2,y:(H-1)/2,k:Math.min((width-32)/W,(height-32)/H)};requestDraw();}
function resize(){
  const rect=stage.getBoundingClientRect();width=rect.width;height=rect.height;dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  if(!initialized){initialized=true;fit();}else requestDraw();
}
function zoom(factor,sx=width/2,sy=height/2){
  const before=toWorld(sx,sy),minimum=.2*Math.min(width/W,height/H);
  view.k=Math.min(256,Math.max(minimum,view.k*factor));
  view.x=before.x-(sx-width/2)/view.k;view.y=before.y-(sy-height/2)/view.k;requestDraw();
}
function addRow(body,label,value,color){
  const row=document.createElement("tr"),a=document.createElement("td"),b=document.createElement("td");
  a.textContent=label;b.textContent=value;if(color)a.style.borderLeft="3px solid "+color;
  row.append(a,b);body.appendChild(row);
}
function select(i){
  if(!Number.isInteger(i)||i<0||i>=N){$("cell-meta").textContent=N?`Enter a cell ID from 0 to ${N-1}.`:"No detected cells.";return;}
  selected=i;$("cell-id").value=i;$("focus").disabled=false;
  $("cell-meta").textContent=`Cell ${i}\nx = ${fmt(seeds[2*i])}, y = ${fmt(seeds[2*i+1])} px\nVoronoi area = ${areas[i]} px²\nMean intensity = ${fmt(means[i])}`;
  const body=$("measurements").querySelector("tbody");body.replaceChildren();
  for(const name of names){
    const m=methods[name];
    addRow(body,m.label,m.valid[i]?"Valid":m.status[i],methodColors[name]);
    for(const key of ["cd_equivalent_px","cd_x_px","cd_y_px","area_px2"]){
      const exponent=key==="area_px2"?2:1,value=m.metrics[key][i]*data.pixel_size**exponent;
      addRow(body,metricLabels[key],fmt(value)+" "+data.unit+(exponent===2?"²":""));
    }
  }
  requestDraw();
}
function focus(){if(selected>=0){view.x=origins[2*selected]+(data.roi_shape[1]-1)/2;view.y=origins[2*selected+1]+(data.roi_shape[0]-1)/2;view.k=Math.min(128,(width-60)/(data.roi_shape[1]+3),(height-60)/(data.roi_shape[0]+3));requestDraw();}}
function screen(event){const r=canvas.getBoundingClientRect();return {x:event.clientX-r.left,y:event.clientY-r.top};}
function cellAt(point){const w=toWorld(point.x,point.y),x=Math.floor(w.x+.5),y=Math.floor(w.y+.5);return x>=0&&x<W&&y>=0&&y<H?labels[y*W+x]:-1;}
let drag=null;
canvas.addEventListener("pointerdown",event=>{if(event.button!==0)return;const p=screen(event);drag={...p,startX:p.x,startY:p.y,moved:false};canvas.setPointerCapture(event.pointerId);$("tooltip").style.display="none";});
canvas.addEventListener("pointermove",event=>{
  const p=screen(event);
  if(drag){view.x-=(p.x-drag.x)/view.k;view.y-=(p.y-drag.y)/view.k;drag.moved ||= Math.hypot(p.x-drag.startX,p.y-drag.startY)>3;drag.x=p.x;drag.y=p.y;requestDraw();return;}
  const i=cellAt(p),tip=$("tooltip");
  if(i<0){tip.style.display="none";return;}
  const m=current(),value=m.metrics[$("metric").value][i]*scale();
  tip.textContent=`Cell ${i}\nMean intensity ${fmt(means[i])}\n${m.label}: ${m.valid[i]?fmt(value)+" "+unit():m.status[i]}`;
  tip.style.display="block";tip.style.left=Math.max(4,Math.min(p.x+14,width-tip.offsetWidth-8))+"px";tip.style.top=Math.max(4,Math.min(p.y+14,height-tip.offsetHeight-8))+"px";
});
canvas.addEventListener("pointerup",event=>{if(drag&&!drag.moved){const i=cellAt(screen(event));if(i>=0)select(i);}drag=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);});
canvas.addEventListener("pointercancel",()=>{drag=null;});
canvas.addEventListener("pointerleave",()=>{$("tooltip").style.display="none";});
canvas.addEventListener("dblclick",event=>{const i=cellAt(screen(event));if(i>=0){select(i);focus();}});
canvas.addEventListener("wheel",event=>{event.preventDefault();const p=screen(event);zoom(Math.exp(-event.deltaY*.0015),p.x,p.y);$("tooltip").style.display="none";},{passive:false});
canvas.addEventListener("keydown",event=>{if(["+","=","-","0"].includes(event.key)){event.preventDefault();if(event.key==="0")fit();else zoom(event.key==="-"?1/1.5:1.5);}});
$("zoom-in").addEventListener("click",()=>zoom(1.6));$("zoom-out").addEventListener("click",()=>zoom(1/1.6));$("fit").addEventListener("click",fit);
$("select-cell").addEventListener("click",()=>select($("cell-id").valueAsNumber));$("cell-id").addEventListener("keydown",event=>{if(event.key==="Enter")select($("cell-id").valueAsNumber);});$("focus").addEventListener("click",focus);
for(const id of ["view","method","metric","compare","vertices","smooth"]){$(id).addEventListener("change",()=>{updateColors();requestDraw();});}
new ResizeObserver(resize).observe(stage);updateColors();resize();
if(N===0){$("cell-meta").textContent="No detected cells.";$("select-cell").disabled=true;}
})();
</script></body></html>'''
