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
 constructor(id){this.id=id;this.value='';this.textContent='';this.innerHTML='';this.listeners={};this.children=[];}
 appendChild(child){this.children.push(child);if(this.id==='method'&&this.children.length===1)this.value=child.value;}
 addEventListener(event,fn){this.listeners[event]=fn;}
 getBoundingClientRect(){return {left:0,top:0,width:1000,height:800};}
 getContext(){return new Proxy({}, {get:(_,name)=>()=>{},set:()=>true});}
 click(){}
}
const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element(m[1])]));
const context=vm.createContext({window:{},document:{getElementById:id=>elements.get(id),createElement:()=>new Element('')},
 Image:class{constructor(){this.complete=true;}},Blob:class{},URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},setTimeout:()=>{}});
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
elements.get('image').onclick({clientX:500,clientY:400});
elements.get('reset').onclick();elements.get('download').onclick();
console.log(`${expected.length} live selections match Python; click/reset/export passed`);
