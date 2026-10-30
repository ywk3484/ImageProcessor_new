/* Run the offline viewer with DOM doubles and compare its live CSV to Python.
 * Usage: node tests/diag_pitch_viewer.cjs REPORT EXPECTED_JSON
 */
"use strict";
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const folder=process.argv[2],expected=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const html=fs.readFileSync(path.join(folder,'index.html'),'utf8');
assert.doesNotMatch(html,/Integrated [xy] distance/);
assert.equal((html.match(/<div id="[xy][pr]">/g)||[]).length,4);
class Element{
 constructor(id){this.id=id;this.value='';this.textContent='';this.innerHTML='';this.listeners={};this.children=[];
  this.checked=true;this.style={};this.captured=new Set();this.box={left:17,top:29,width:1000,height:800};
  this.context=new Proxy({path:[],strokes:[],draws:[],beginPath(){this.path=[];},
   moveTo(...p){this.path.push(['M',...p]);},lineTo(...p){this.path.push(['L',...p]);},
   stroke(){this.strokes.push({color:this.strokeStyle,path:this.path.slice(),width:this.lineWidth});},
   drawImage(...args){this.draws.push(args);}}, {get:(obj,name)=>name in obj?obj[name]:()=>{}});}
 appendChild(child){this.children.push(child);if(this.id==='method'&&this.children.length===1)this.value=child.value;}
 addEventListener(event,fn){this.listeners[event]=fn;}
 getBoundingClientRect(){return this.box;}
 getContext(){return this.context;}
 setPointerCapture(id){this.captured.add(id);}
 hasPointerCapture(id){return this.captured.has(id);}
 releasePointerCapture(id){this.captured.delete(id);}
 click(){}
}
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element(m[1])]));
const context=vm.createContext({window:{devicePixelRatio:2,addEventListener(){}},document:{getElementById:id=>elements.get(id),createElement:()=>new Element('')},
 Image:class{constructor(){this.complete=true;this.naturalWidth=100;}},Blob:class{},URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},setTimeout:()=>{}});
vm.runInContext(fs.readFileSync(path.join(folder,'data.js'),'utf8'),context);
vm.runInContext(html.match(/<script>\s*([\s\S]*?)<\/script>/)[1],context);
for(const scenario of expected){
 context.scenario=scenario;
 vm.runInContext(`reference=scenario.reference; $('method').value=scenario.method;
 $('rows').value=scenario.rows;$('columns').value=scenario.columns;
 $('nx').value=scenario.nominal?String(scenario.nominal[0]):'';$('ny').value=scenario.nominal?String(scenario.nominal[1]):'';
 update();window.actual=csvRows();`,context);
 const rows=JSON.parse(JSON.stringify(context.window.actual));
 assert.equal(rows.length,scenario.csv.length+1);
 for(let i=0;i<scenario.csv.length;i++){
  const wanted=scenario.csv[i],actual=rows[i+1];assert.equal(actual[0],scenario.method);assert.equal(actual[1],wanted[0]);
  // axis, node, position, pitch, std, pairs, distance, registration, nominal
  for(let j=1;j<wanted.length;j++){
   const v=actual[j+1],e=wanted[j];
   if(e===null)assert.equal(v,null);else assert.ok(Math.abs(v-e)<1e-9,`row ${i}, field ${j}: ${v} != ${e}`);
  }
 }
 assert.match(elements.get('xp').innerHTML,/<svg/);
 assert.match(elements.get('yr').innerHTML,/<svg/);
}
// Exercise reference clicking, bounds clamping, method changes, reset, export.
elements.get('rows').value='100000';elements.get('columns').value='0';
elements.get('rows').listeners.change();
assert.equal(Number(elements.get('columns').value),1);
elements.get('reset').onclick();elements.get('download').onclick();

// Check view geometry and interactions against measured coordinates, including
// nonzero canvas position and HiDPI backing pixels. Zoom must not change CSVs.
const evaluate=code=>vm.runInContext(code,context),canvas=elements.get('image'),ctx=canvas.context;
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
evaluate(`$('method').value='gaussian';$('rows').value=1;$('columns').value=1;reference=D.first_reference;update();fitImage();`);
const before=evaluate('JSON.stringify(csvRows())'),initial=evaluate('view.scale');
elements.get('zoom-in').onclick();close(evaluate('view.scale'),initial*1.5);
elements.get('zoom-out').onclick();close(evaluate('view.scale'),initial);
elements.get('actual-size').onclick();close(evaluate('view.scale'),1);
assert.equal(canvas.width,2000);assert.equal(canvas.height,1600);
const anchor=[321,217];context.anchor=anchor;
const pointBefore=Array.from(evaluate('imagePoint(anchor)'));
let prevented=false;
canvas.listeners.wheel({clientX:anchor[0]+canvas.box.left,clientY:anchor[1]+canvas.box.top,deltaY:-120,deltaMode:0,preventDefault(){prevented=true;}});
assert.ok(prevented);assert.ok(evaluate('view.scale')>1);
Array.from(evaluate('imagePoint(anchor)')).forEach((v,i)=>close(v,pointBefore[i]));
assert.equal(evaluate('JSON.stringify(csvRows())'),before);
// A screen/image round trip includes the half-pixel offset for pixel centers.
Array.from(evaluate('imagePoint(screenPoint([0,0]))')).forEach(v=>close(v,0));
const draw=ctx.draws.at(-1);assert.deepEqual(draw.slice(1),[-.5,-.5,100,80]);

const pointer=(p,id=1)=>({clientX:p[0]+canvas.box.left,clientY:p[1]+canvas.box.top,button:0,pointerId:id});
const oldRef=evaluate('reference'),oldOffset=evaluate('[view.x,view.y]');
canvas.listeners.pointerdown(pointer([400,300]));canvas.listeners.pointermove(pointer([480,350]));canvas.listeners.pointerup(pointer([480,350]));
close(evaluate('view.x'),oldOffset[0]+80);close(evaluate('view.y'),oldOffset[1]+50);assert.equal(evaluate('reference'),oldRef);
elements.get('fit-image').onclick();close(evaluate('view.scale'),initial);
// Hit testing must use the active method's measured centers at any zoom/pan.
evaluate('zoomImage(view.scale*2);');
const target=17,screen=Array.from(evaluate(`screenPoint(D.methods.gaussian.centers[${target}])`));
canvas.listeners.pointermove(pointer(screen));
assert.match(elements.get('inspection').textContent,/Hole 17 .*Measured center \[/);
assert.match(elements.get('inspection').textContent,new RegExp(evaluate(`D.methods.gaussian.centers[${target}][0].toFixed(4)`).replace('.','\\.')));
canvas.listeners.pointerdown(pointer(screen));canvas.listeners.pointerup(pointer(screen));assert.equal(evaluate('reference'),target);
assert.match(elements.get('status').textContent,/measured center/);
elements.get('reset').onclick();assert.equal(evaluate('reference'),evaluate('D.first_reference'));
close(evaluate('view.scale'),initial*2); // Selection changes keep the view.

// Marker geometry covers all valid centers, including outside the bands,
// and excludes failed/nonfinite centers. Invalid markers use seeds instead.
evaluate(`$('method').value='local_projection';update();`);ctx.strokes.length=0;evaluate('drawImage()');
const markers=ctx.strokes.find(s=>s.color==='#58ffd1');assert.ok(markers);
const measured=JSON.parse(evaluate('JSON.stringify(D.methods.local_projection)'));
const validCount=measured.valid.filter(Boolean).length;assert.equal(markers.path.length,validCount*4);
const size=4/evaluate('view.scale');let k=0;
for(let i=0;i<measured.valid.length;i++)if(measured.valid[i]){const [x,y]=measured.centers[i];
 close(markers.path[k][1],x-size);close(markers.path[k][2],y);k+=4;}
assert.ok(evaluate(`D.methods.local_projection.valid.some((v,i)=>v&&!current.x.selected.includes(i)&&!current.y.selected.includes(i))`));
assert.ok(ctx.strokes.find(s=>s.color==='#ffc247').path.length>=8);
const failed=Array.from(evaluate('screenPoint(D.seeds[3])'));canvas.listeners.pointermove(pointer(failed));
assert.match(elements.get('inspection').textContent,/No valid measured center.*seed.*excluded from pitch/);
canvas.listeners.pointerdown(pointer(failed));canvas.listeners.pointerup(pointer(failed));assert.equal(evaluate('reference'),3);
assert.match(elements.get('status').textContent,/seed .*no valid measured reference center/);
elements.get('show-centers').checked=false;ctx.strokes.length=0;elements.get('show-centers').listeners.change();
assert.ok(!ctx.strokes.some(s=>s.color==='#58ffd1'));
elements.get('show-bands').checked=false;ctx.strokes.length=0;elements.get('show-bands').listeners.change();
assert.ok(!ctx.strokes.some(s=>s.color==='#ff9b32'||s.color==='#4aaeff'));
elements.get('show-centers').checked=true;elements.get('show-bands').checked=true;
// Fit resets pan, resize preserves the center, and zoom limits stay finite.
const centerBefore=Array.from(evaluate('imagePoint([view.width/2,view.height/2])'));
canvas.box.width=750;canvas.box.height=500;evaluate('drawImage()');
Array.from(evaluate('imagePoint([view.width/2,view.height/2])')).forEach((v,i)=>close(v,centerBefore[i]));
elements.get('fit-image').onclick();close(evaluate('view.scale'),6.25);
evaluate('zoomImage(1e20)');assert.ok(evaluate('Number.isFinite(view.scale)&&view.scale<=Math.max(32,view.fit*32)'));
evaluate('zoomImage(1e-20)');close(evaluate('view.scale'),evaluate('view.fit*.1'));
canvas.listeners.keydown({key:'f',preventDefault(){}});close(evaluate('view.scale'),6.25);
canvas.listeners.pointerdown(pointer([100,100]));canvas.listeners.pointercancel();assert.equal(evaluate('drag'),null);
canvas.box.width=15000;canvas.box.height=12000;evaluate('drawImage()');
elements.get('actual-size').onclick();close(evaluate('view.scale'),1); // Native size remains reachable for tiny images.
console.log(`${expected.length} live selections match Python; zoom/pan/center overlays/hover/click/reset/export passed`);
