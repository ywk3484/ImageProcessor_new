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
svg{width:100%;height:auto}small{color:#465b70}
.image-tools{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:12px 0}.image-tools label{flex-direction:row;align-items:center}
.image-tools input{width:auto}.image-tools output{min-width:70px;font-variant-numeric:tabular-nums}
.image-view{height:clamp(300px,50vh,600px);background:#172536;overflow:hidden;border-radius:5px}
canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none}canvas:focus-visible{outline:3px solid #1469b8;outline-offset:-3px}
.center{color:#087b62}.invalid{color:#986600}#inspection{min-height:3em;font-variant-numeric:tabular-nums}
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
<span class="center">Green crosshairs: measured centers</span><span class="ref">Purple: reference</span>
<span class="invalid">Amber crosses: invalid fits (at seeds)</span><span>Red crosses: unindexed detections</span></div><p id="status"></p>
<div class="image-tools" role="group" aria-label="Image view controls">
<button id="zoom-out" aria-label="Zoom out">−</button><output id="zoom-level" aria-label="Image zoom"></output>
<button id="zoom-in" aria-label="Zoom in">+</button><button id="fit-image">Fit image</button><button id="actual-size">1:1 pixels</button>
<label><input id="show-centers" type="checkbox" checked>Show measured centers</label>
<label><input id="show-bands" type="checkbox" checked>Show averaging bands</label></div>
<p><small>Scroll over the image to zoom at the pointer; drag to pan. Hover over a marker for its measured [x, y] coordinates in pixels.
Click a hole to select the reference. Fit image restores the full view. Image pixel centers have integer coordinates.</small></p>
<div id="image-view" class="image-view"><canvas id="image" tabindex="0" aria-label="Pitch image with measured centers; scroll to zoom, drag to pan, click to select a reference"></canvas></div>
<p id="inspection" aria-live="polite">Hover over a center to inspect its coordinates.</p></div>
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
let reference=D.reference, current, hovered=null, drag=null;
const view={scale:0,x:0,y:0,width:0,height:0,fit:0};
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
// View offsets are in CSS pixels; image coordinates identify pixel centers.
const image=new Image();image.onload=()=>drawImage();image.src='image.png';
const imagePoint=p=>[(p[0]-view.x)/view.scale-.5,(p[1]-view.y)/view.scale-.5];
const screenPoint=p=>[(p[0]+.5)*view.scale+view.x,(p[1]+.5)*view.scale+view.y];
function pointerPoint(e){const box=$('image').getBoundingClientRect();
 return [(e.clientX-box.left)*view.width/box.width,(e.clientY-box.top)*view.height/box.height];}
function resizeImage(){const box=$('image').getBoundingClientRect();if(!box.width||!box.height)return;
 const w=box.width,h=box.height;if(w===view.width&&h===view.height)return;
 const center=view.scale?imagePoint([view.width/2,view.height/2]):[(D.width-1)/2,(D.height-1)/2];
 const fit=Math.min(w/D.width,h/D.height),wasFit=!view.scale||Math.abs(view.scale-view.fit)<1e-9;
 view.scale=wasFit?fit:view.scale;view.fit=fit;view.width=w;view.height=h;
 view.x=w/2-(center[0]+.5)*view.scale;view.y=h/2-(center[1]+.5)*view.scale;
 const ratio=window.devicePixelRatio||1;$('image').width=Math.round(w*ratio);$('image').height=Math.round(h*ratio);
}
function inspect(point){const m=D.methods[$('method').value];
 const coord=p=>'['+p.map(v=>v.toFixed(4)).join(', ')+'] px';
 let text=point?'Pointer '+coord(point)+'. ':'Hover over a center to inspect its coordinates.';
 if(hovered!==null){const i=hovered,rc=D.indices[i];text+=`Hole ${i} [row ${rc[0]}, column ${rc[1]}]. `;
  if(m.valid[i]){const used=['x','y'].filter(axis=>current[axis].pairs.some(pair=>pair.includes(i)));
   text+='Measured center '+coord(m.centers[i])+'. '+(used.length?'Used in '+used.join(' and ')+' pitch pairs.':'Outside the current pitch pairs.');}
  else text+=(rc[0]<0?'Unindexed detection':'No valid measured center for this method')+'; seed '+coord(D.seeds[i])+' (excluded from pitch).';
 }$('inspection').textContent=text;
}
function nearestHole(point,indexedOnly=false,maxDistance=Infinity){const m=D.methods[$('method').value];let best=maxDistance**2,index=null;
 D.seeds.forEach((seed,i)=>{if(indexedOnly&&D.indices[i][0]<0)return;const p=m.valid[i]?m.centers[i]:seed;
  const d=(p[0]-point[0])**2+(p[1]-point[1])**2;if(d<=best){best=d;index=i;}});return index;}
function zoomImage(scale,anchor=[view.width/2,view.height/2]){if(!view.scale)return;
 const point=imagePoint(anchor);view.scale=Math.max(Math.min(1,view.fit*.1),Math.min(Math.max(32,view.fit*32),scale));
 view.x=anchor[0]-(point[0]+.5)*view.scale;view.y=anchor[1]-(point[1]+.5)*view.scale;
 hovered=null;inspect();drawImage();
}
function fitImage(){resizeImage();view.scale=view.fit;view.x=(view.width-D.width*view.scale)/2;view.y=(view.height-D.height*view.scale)/2;
 hovered=null;inspect();drawImage();}
function drawImage(){resizeImage();$('zoom-level').textContent=(view.scale*100).toFixed(1)+'%';
 if(!current||!image.complete||!image.naturalWidth||!view.scale)return;
 const canvas=$('image'),ctx=canvas.getContext('2d'),scale=view.scale,rx=canvas.width/view.width,ry=canvas.height/view.height;
 ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;
 ctx.setTransform(rx*scale,0,0,ry*scale,rx*(view.x+scale/2),ry*(view.y+scale/2));
 // Drawing at -0.5 aligns integer [x, y] coordinates with image pixel centers.
 ctx.drawImage(image,-.5,-.5,D.width,D.height);
 const m=D.methods[$('method').value];
 if($('show-bands').checked)for(const [axis,color] of [['x','#ff9b32'],['y','#4aaeff']]){const p=current[axis];ctx.strokeStyle=color;ctx.lineWidth=1/scale;
  if(p.selected.length){const xs=p.selected.map(i=>D.seeds[i][0]),ys=p.selected.map(i=>D.seeds[i][1]);const x0=minimum(xs)-D.pitch_xy[0]*.35,y0=minimum(ys)-D.pitch_xy[1]*.35;
   const w=maximum(xs)-x0+D.pitch_xy[0]*.35,h=maximum(ys)-y0+D.pitch_xy[1]*.35;
   ctx.globalAlpha=.12;ctx.fillStyle=color;ctx.fillRect(x0,y0,w,h);ctx.globalAlpha=1;ctx.strokeRect(x0,y0,w,h);}
  ctx.beginPath();for(const [i,j] of p.pairs){ctx.moveTo(...m.centers[i]);ctx.lineTo(...m.centers[j]);}ctx.stroke();
 }
 function cross(p,s,diagonal){const [x,y]=p;s/=scale;
  ctx.moveTo(x-s,y-(diagonal?s:0));ctx.lineTo(x+s,y+(diagonal?s:0));
  ctx.moveTo(x-(diagonal?s:0),y+s);ctx.lineTo(x+(diagonal?s:0),y-s);}
 if($('show-centers').checked){
  // Every valid center of the selected method, including outside the bands.
  ctx.beginPath();m.centers.forEach((p,i)=>{if(m.valid[i])cross(p,4,false);});
  ctx.strokeStyle='#102d28';ctx.lineWidth=3/scale;ctx.stroke();ctx.strokeStyle='#58ffd1';ctx.lineWidth=1.2/scale;ctx.stroke();
 }
 for(const [unindexed,color] of [[false,'#ffc247'],[true,'#ff4949']]){ctx.beginPath();D.seeds.forEach((p,i)=>{
   if(!m.valid[i]&&(D.indices[i][0]<0)===unindexed)cross(p,4,true);});
  ctx.strokeStyle='#172536';ctx.lineWidth=3/scale;ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=1.5/scale;ctx.stroke();}
 const point=m.valid[reference]?m.centers[reference]:D.seeds[reference];ctx.strokeStyle='#d445ff';ctx.lineWidth=2/scale;ctx.beginPath();ctx.arc(...point,7/scale,0,2*Math.PI);ctx.stroke();
 if(hovered!==null){const p=m.valid[hovered]?m.centers[hovered]:D.seeds[hovered];ctx.strokeStyle='#fff';ctx.lineWidth=1.5/scale;
  ctx.beginPath();ctx.arc(...p,9/scale,0,2*Math.PI);ctx.stroke();}
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
 const skipped=D.indices.filter(rc=>rc[0]<0).length;
 const location=m.valid[reference]?`measured center [${m.centers[reference].map(v=>v.toFixed(4)).join(', ')}] px`:
  `seed [${D.seeds[reference].map(v=>v.toFixed(4)).join(', ')}] px (no valid measured reference center)`;
 $('status').textContent=`${D.layout}${D.layout==='staggered'?' ('+D.stagger_axis+' offsets)':''}; ${skipped} unindexed detections; reference hole ${reference} [row ${rc[0]}, column ${rc[1]}], ${location}. ${stats('x')}. ${stats('y')}.`;
 hovered=null;inspect();drawImage();
}
function csvRows(){const rows=[['method','axis','node','position','pitch_to_next','sample_std','pairs','distance','registration','nominal_pitch','unit','reference_hole','average_rows','average_columns']];
 for(const axis of ['x','y']){const p=current[axis];for(let i=0;i<p.position.length;i++)rows.push([$('method').value,axis,i,p.position[i],p.pitch[i],p.std[i],p.counts[i],p.distance[i],p.registration[i],p.nominal,D.unit,reference,$('rows').value,$('columns').value]);}return rows;}
$('download').onclick=()=>{const csv=csvRows().map(row=>row.map(v=>typeof v==='number'&&!finite(v)||v===undefined?'':'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='pitch_selection.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('zoom-in').onclick=()=>zoomImage(view.scale*1.5);
$('zoom-out').onclick=()=>zoomImage(view.scale/1.5);
$('fit-image').onclick=fitImage;$('actual-size').onclick=()=>zoomImage(1);
for(const id of ['show-centers','show-bands'])$(id).addEventListener('change',drawImage);
const canvas=$('image');
canvas.addEventListener('wheel',e=>{e.preventDefault();const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?view.height:1);
 zoomImage(view.scale*Math.exp(Math.max(-.5,Math.min(.5,-delta*.002))),pointerPoint(e));},{passive:false});
canvas.addEventListener('pointerdown',e=>{if(e.button!==0||drag)return;const p=pointerPoint(e);
 drag={id:e.pointerId,start:p,x:view.x,y:view.y,moved:false};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{const p=pointerPoint(e);
 if(drag){if(e.pointerId!==drag.id)return;const dx=p[0]-drag.start[0],dy=p[1]-drag.start[1];
  if(Math.hypot(dx,dy)>4)drag.moved=true;
  if(drag.moved){view.x=drag.x+dx;view.y=drag.y+dy;hovered=null;canvas.style.cursor='grabbing';inspect(imagePoint(p));drawImage();}return;}
 const point=imagePoint(p);hovered=nearestHole(point,false,12/view.scale);inspect(point);drawImage();});
canvas.addEventListener('pointerup',e=>{if(!drag||e.pointerId!==drag.id)return;const moved=drag.moved;drag=null;
 if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);canvas.style.cursor='grab';
 if(!moved){const point=imagePoint(pointerPoint(e));if(point[0]<-.5||point[0]>D.width-.5||point[1]<-.5||point[1]>D.height-.5)return;
  const i=nearestHole(point,true);if(i!==null){reference=i;update();}}
});
function cancelDrag(){drag=null;canvas.style.cursor='grab';}
canvas.addEventListener('pointercancel',cancelDrag);canvas.addEventListener('lostpointercapture',cancelDrag);
canvas.addEventListener('pointerleave',()=>{if(!drag){hovered=null;inspect();drawImage();}});
canvas.addEventListener('keydown',e=>{if(['+','=','-','0','f','F'].includes(e.key)){e.preventDefault();
 if(e.key==='0'||e.key.toLowerCase()==='f')fitImage();else zoomImage(view.scale*(e.key==='-'?1/1.5:1.5));}});
window.addEventListener('resize',()=>{hovered=null;inspect();drawImage();});
$('reset').onclick=()=>{reference=D.first_reference;update();};
for(const id of ['method','rows','columns','nx','ny'])$(id).addEventListener('change',update);
for(const [key,m] of Object.entries(D.methods)){const option=document.createElement('option');option.value=key;option.textContent=key+' — '+m.label;$('method').appendChild(option);}
$('rows').value=D.average_rows;$('rows').max=D.shape[0];$('columns').value=D.average_columns;$('columns').max=D.shape[1];
if(D.nominal){$('nx').value=D.nominal[0];$('ny').value=D.nominal[1];}
$('unit').textContent=D.unit;$('source').textContent=D.title;update();
</script></html>'''
