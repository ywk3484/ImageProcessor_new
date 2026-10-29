"""Offline interactive pitch report template (no network or web server)."""

HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Pattern pitch and registration</title>
<link rel="icon" href="data:,">
<style>
body{font:15px system-ui,sans-serif;background:#f4f6f8;color:#172536;margin:24px auto;max-width:1320px;padding:0 20px}
h1{font-size:25px}h2{font-size:18px}p{line-height:1.5}label{display:inline-flex;flex-direction:column;gap:5px;font-size:13px}
select,input,button{font:inherit;padding:7px;border:1px solid #aebdcc;border-radius:5px;background:white}input{width:105px}
.controls{display:flex;align-items:end;flex-wrap:wrap;gap:12px;padding:16px;background:white;border-radius:8px;position:sticky;top:0;z-index:1}
.panel{background:white;padding:16px;border-radius:8px;margin-top:16px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}
svg{width:100%;height:auto}canvas{display:block;max-width:100%;margin:auto;cursor:crosshair}small{color:#465b70}
.legend{display:flex;gap:20px;flex-wrap:wrap}.x{color:#c15c00}.y{color:#1469b8}.ref{color:#9828b9}
@media(max-width:800px){.plots{grid-template-columns:1fr}.controls{position:static}}
</style>
<h1>Pattern pitch and registration</h1><p id="source"></p>
<div class="controls">
<label>CD center method<select id="method"></select></label>
<label>Rows averaged for x<input id="rows" type="number" min="1" step="1"></label>
<label>Columns averaged for y<input id="columns" type="number" min="1" step="1"></label>
<label>Nominal x pitch<input id="nx" type="number" min="0" step="any" placeholder="measured mean"></label>
<label>Nominal y pitch<input id="ny" type="number" min="0" step="any" placeholder="measured mean"></label>
<button id="reset">First reference</button><button id="download">Download current CSV</button>
</div>
<p><small>Click a hole in the image to move the reference and averaging bands. Leave nominal pitches blank to use the
selected pairs' mean for each method. Pitch and registration error use <span id="unit"></span>; image coordinates use pixels.</small></p>
<div class="panel"><div class="legend"><span class="x">Orange: rows averaged for x</span><span class="y">Blue: columns averaged for y</span>
<span class="ref">Purple: reference</span></div><p id="status"></p><canvas id="image"></canvas></div>
<div class="panel plots"><div><h2>x pitch averaged over rows</h2><div id="xp"></div></div>
<div><h2>y pitch averaged over columns</h2><div id="yp"></div></div>
<div><h2>x registration error</h2><div id="xr"></div></div><div><h2>y registration error</h2><div id="yr"></div></div></div>
<p><small>Pitch is the mean projected separation of valid adjacent pairs. Staggered rows/columns use logical lattice neighbours,
so the row offset does not double the y pitch. The sample standard deviation and pair counts are in the CSV.
Registration error sums (pitch − mean pitch) from the reference by default; entering a nominal pitch uses that baseline instead.
Missing intervals break the integral; curves beyond a gap are unavailable. The default mean removes the average pitch
offset and therefore shows relative distortion, not absolute placement against a design.</small></p>
<p><a href="profiles.csv">Initial profiles CSV</a> · <a href="centers.csv">All method centers CSV</a> ·
<a href="summary.csv">Initial summary CSV</a> · <a href="report.png">Initial plot PNG</a> · <a href="data.json">Raw report data</a></p>
<script src="data.js"></script><script>
"use strict";
const D=window.PITCH_DATA, $=id=>document.getElementById(id), finite=Number.isFinite;
let reference=D.reference, current;
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:NaN;
const minimum=a=>a.reduce((v,x)=>Math.min(v,x),Infinity),maximum=a=>a.reduce((v,x)=>Math.max(v,x),-Infinity);
function band(count,ref,size){const start=Math.min(Math.max(ref-Math.floor((count-1)/2),0),size-count);return [start,start+count-1];}
function integrate(v,ref){const out=Array(v.length+1).fill(NaN);out[ref]=0;
 for(let i=ref;i<v.length;i++)out[i+1]=out[i]+v[i];for(let i=ref-1;i>=0;i--)out[i]=out[i+1]-v[i];return out;}
function profile(axis,m,b,nominal){
 const coord=axis==='x'?0:1,dim=1-coord,n=D.shape[dim],ref=D.indices[reference][dim];
 const values=Array.from({length:n-1},()=>[]),coords=Array.from({length:n},()=>[]),fallback=Array.from({length:n},()=>[]);
 const selected=[];
 for(let i=0;i<D.indices.length;i++){const rc=D.indices[i];if(rc[1-dim]<b[0]||rc[1-dim]>b[1])continue;
  selected.push(i);fallback[rc[dim]].push(D.seeds[i][coord]*D.pixel_size);if(m.valid[i])coords[rc[dim]].push(m.centers[i][coord]*D.pixel_size);}
 const pairs=[];
 for(const pair of D.pairs[axis]){const [i,j]=pair,rc=D.indices[i];if(rc[1-dim]<b[0]||rc[1-dim]>b[1]||!m.valid[i]||!m.valid[j])continue;
  values[rc[dim]].push((m.centers[j][coord]-m.centers[i][coord])*D.pixel_size);pairs.push(pair);}
 const pitch=values.map(mean),counts=values.map(a=>a.length),std=values.map((a,i)=>a.length>1?Math.sqrt(a.reduce((s,v)=>s+(v-pitch[i])**2,0)/(a.length-1)):NaN);
 if(!finite(nominal))nominal=values.reduce((s,a)=>s+a.reduce((t,v)=>t+v,0),0)/counts.reduce((s,n)=>s+n,0);
 return {axis,position:coords.map((a,i)=>mean(a.length?a:fallback[i])),pitch,counts,std,nominal,
  distance:integrate(pitch,ref),registration:integrate(pitch.map(v=>v-nominal),ref),selected,pairs,ref};
}
function svgPlot(id,x,y,color,xlabel,ylabel,baseline){
 const good=x.map((v,i)=>finite(v)&&finite(y[i])?[v,y[i]]:null).filter(Boolean),w=570,h=265,l=85,r=18,t=22,b=52;
 if(!good.length){$(id).textContent='No connected valid data in this selection';return;}
 let xmin=minimum(good.map(p=>p[0])),xmax=maximum(good.map(p=>p[0]));
 let ymin=minimum(good.map(p=>p[1])),ymax=maximum(good.map(p=>p[1]));
 if(finite(baseline)){ymin=Math.min(ymin,baseline);ymax=Math.max(ymax,baseline);}
 const pad=(ymax-ymin)*.08||Math.max(Math.abs(ymin)*.002,.001);ymin-=pad;ymax+=pad;if(xmax===xmin)xmax=xmin+1;
 const sx=v=>l+(v-xmin)/(xmax-xmin)*(w-l-r),sy=v=>h-b-(v-ymin)/(ymax-ymin)*(h-t-b);
 const fmt=v=>Math.abs(v)>=1e5||Math.abs(v)>0&&Math.abs(v)<.001?v.toExponential(2):Number(v.toPrecision(6)).toString();
 let s=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${ylabel}"><g font-family="system-ui" font-size="11" fill="#465b70">`;
 for(let i=0;i<=4;i++){const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;
  s+=`<path d="M${l} ${sy(yv)}H${w-r}" stroke="#e5eaf0"/><text x="${l-8}" y="${sy(yv)+4}" text-anchor="end">${fmt(yv)}</text><text x="${sx(xv)}" y="${h-b+20}" text-anchor="middle">${fmt(xv)}</text>`;}
 if(finite(baseline))s+=`<path d="M${l} ${sy(baseline)}H${w-r}" stroke="#8595a8" stroke-dasharray="5 4"/>`;
 let path='',pen=false;for(let i=0;i<x.length;i++){if(!finite(x[i])||!finite(y[i])){pen=false;continue;}path+=`${pen?'L':'M'}${sx(x[i])} ${sy(y[i])}`;pen=true;}
 s+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
 for(const p of good)s+=`<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="2" fill="${color}"><title>${fmt(p[0])}, ${fmt(p[1])}</title></circle>`;
 s+=`<text x="${(l+w-r)/2}" y="${h-8}" text-anchor="middle">${xlabel}</text><text transform="translate(16 ${(t+h-b)/2}) rotate(-90)" text-anchor="middle">${ylabel}</text></g></svg>`;$(id).innerHTML=s;
}
const image=new Image();image.src='image.png';image.onload=()=>drawImage();
function drawImage(){if(!current||!image.complete)return;const canvas=$('image'),ctx=canvas.getContext('2d');
 const scale=Math.min(1,1200/D.width,600/D.height);canvas.width=Math.round(D.width*scale);canvas.height=Math.round(D.height*scale);
 ctx.setTransform(canvas.width/D.width,0,0,canvas.height/D.height,0,0);ctx.drawImage(image,0,0,D.width,D.height);
 const m=D.methods[$('method').value];
 for(const [axis,color] of [['x','#ff9b32'],['y','#4aaeff']]){const p=current[axis];ctx.strokeStyle=color;ctx.lineWidth=1/scale;
  // Draw every selected measured pair and outline the selected seed area.
  ctx.beginPath();for(const [i,j] of p.pairs){ctx.moveTo(...m.centers[i]);ctx.lineTo(...m.centers[j]);}ctx.stroke();
  if(p.selected.length){const xs=p.selected.map(i=>D.seeds[i][0]),ys=p.selected.map(i=>D.seeds[i][1]);const x0=minimum(xs)-D.pitch_xy[0]*.35,y0=minimum(ys)-D.pitch_xy[1]*.35;
   const w=maximum(xs)-x0+D.pitch_xy[0]*.35,h=maximum(ys)-y0+D.pitch_xy[1]*.35;
   ctx.globalAlpha=.12;ctx.fillStyle=color;ctx.fillRect(x0,y0,w,h);ctx.globalAlpha=1;ctx.strokeRect(x0,y0,w,h);}
 }
 // Mark actual method centers; no substitution for failed fits.
 ctx.fillStyle='#fff';for(const i of new Set([...current.x.selected,...current.y.selected]))if(m.valid[i]){ctx.beginPath();ctx.arc(...m.centers[i],1.6/scale,0,2*Math.PI);ctx.fill();}
 const point=m.valid[reference]?m.centers[reference]:D.seeds[reference];ctx.strokeStyle='#d445ff';ctx.lineWidth=2/scale;ctx.beginPath();ctx.arc(...point,6/scale,0,2*Math.PI);ctx.stroke();
}
function update(){
 for(const [id,max] of [['rows',D.shape[0]],['columns',D.shape[1]]]){$(id).value=Math.max(1,Math.min(max,Math.floor(Number($(id).value)||1)));}
 const rc=D.indices[reference],rb=band(Number($('rows').value),rc[0],D.shape[0]),cb=band(Number($('columns').value),rc[1],D.shape[1]);
 const m=D.methods[$('method').value],nom=id=>$(id).value.trim()&&Number($(id).value)>0?Number($(id).value):NaN;
 current={x:profile('x',m,rb,nom('nx')),y:profile('y',m,cb,nom('ny'))};
 for(const [axis,color] of [['x','#c15c00'],['y','#1469b8']]){const p=current[axis],mid=p.position.slice(0,-1).map((v,i)=>(v+p.position[i+1])/2);
  svgPlot(axis+'p',mid,p.pitch,color,`${axis} position (${D.unit})`,`pitch (${D.unit})`,p.nominal);
  svgPlot(axis+'r',p.position,p.registration,color,`${axis} position (${D.unit})`,`registration error (${D.unit})`,0);}
 const stats=axis=>{const p=current[axis],c=p.counts;return `${axis}: ${c.reduce((s,n)=>s+n,0)} pairs, ${minimum(c)}–${maximum(c)} per interval; baseline ${p.nominal.toPrecision(7)} ${D.unit}`;};
 $('status').textContent=`${D.layout}${D.layout==='staggered'?' ('+D.stagger_axis+' offsets)':''}; reference hole ${reference} [row ${rc[0]}, column ${rc[1]}], image [${D.seeds[reference].map(v=>v.toFixed(2)).join(', ')}] px${m.valid[reference]?'':' (this method has no valid reference center)'}. ${stats('x')}. ${stats('y')}.`;
 drawImage();
}
function csvRows(){const rows=[['method','axis','node','position','pitch_to_next','sample_std','pairs','distance','registration','nominal_pitch','unit','reference_hole','average_rows','average_columns']];
 for(const axis of ['x','y']){const p=current[axis];for(let i=0;i<p.position.length;i++)rows.push([$('method').value,axis,i,p.position[i],p.pitch[i],p.std[i],p.counts[i],p.distance[i],p.registration[i],p.nominal,D.unit,reference,$('rows').value,$('columns').value]);}return rows;}
$('download').onclick=()=>{const csv=csvRows().map(row=>row.map(v=>typeof v==='number'&&!finite(v)||v===undefined?'':'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='pitch_selection.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('image').onclick=e=>{const box=$('image').getBoundingClientRect(),x=(e.clientX-box.left)/box.width*D.width,y=(e.clientY-box.top)/box.height*D.height;
 let best=Infinity;D.seeds.forEach((p,i)=>{const d=(p[0]-x)**2+(p[1]-y)**2;if(d<best){best=d;reference=i;}});update();};
$('reset').onclick=()=>{reference=D.first_reference;update();};
for(const id of ['method','rows','columns','nx','ny'])$(id).addEventListener('change',update);
for(const [key,m] of Object.entries(D.methods)){const option=document.createElement('option');option.value=key;option.textContent=key+' — '+m.label;$('method').appendChild(option);}
$('rows').value=D.average_rows;$('rows').max=D.shape[0];$('columns').value=D.average_columns;$('columns').max=D.shape[1];
if(D.nominal){$('nx').value=D.nominal[0];$('ny').value=D.nominal[1];}
$('unit').textContent=D.unit;$('source').textContent=D.title;update();
</script></html>'''
