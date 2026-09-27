/* Execute the exported viewer's JavaScript with DOM/canvas test doubles.
 * No browser, network requests, or third-party modules are used.
 * Usage: node tests/diag_cd_viewer.cjs path/to/contours.html
 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const path = process.argv[2];
if (!path) throw new Error("Pass the generated contours.html path");
const html = fs.readFileSync(path, "utf8");
const payload = html.match(/<script id="cd-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const source = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(payload);
const elements = new Map(), canvases = [], frames = [];

class Context {
    constructor() { this.paths = 0; this.vertices = 0; }
    createImageData(w, h) { return {data: new Uint8ClampedArray(w * h * 4), width: w, height: h}; }
    putImageData(pixels) { this.pixels = pixels; }
    setTransform() { this.paths = 0; this.vertices = 0; }
    translate(x, y) { this.translation = [x, y]; }
    scale(x, y) { this.scaling = [x, y]; }
    fillRect() {}
    drawImage(image) { this.image = image; }
    beginPath() {}
    moveTo(x, y) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); }
    lineTo(x, y) { assert.ok(Number.isFinite(x) && Number.isFinite(y)); this.vertices++; }
    stroke() { this.paths++; }
    setLineDash() {}
    strokeRect() {}
}

class Element {
    constructor(tag, id = "") {
        this.tag = tag; this.id = id; this.value = ""; this.style = {};
        this.children = []; this.listeners = new Map(); this.textContent = "";
        this.offsetWidth = 200; this.offsetHeight = 70; this.checked = false;
        if (tag === "canvas") { this.context = new Context(); canvases.push(this); }
    }
    get valueAsNumber() { return this.value === "" ? NaN : Number(this.value); }
    appendChild(child) {
        this.children.push(child);
        if (this.tag === "select" && this.children.length === 1) this.value = child.value;
        return child;
    }
    append(...children) { for (const child of children) this.appendChild(child); }
    replaceChildren() { this.children = []; }
    remove() {}
    querySelector(selector) {
        assert.equal(selector, "tbody");
        if (!this.body) this.body = new Element("tbody");
        return this.body;
    }
    addEventListener(type, fn) { this.listeners.set(type, fn); }
    getBoundingClientRect() { return {left: 0, top: 0, width: 960, height: 540}; }
    getContext(type) { assert.equal(type, "2d"); return this.context; }
    setPointerCapture() { this.captured = true; }
    hasPointerCapture() { return this.captured; }
    releasePointerCapture() { this.captured = false; }
}

for (const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const el = new Element(match[1], match[2]);
    el.disabled = /\bdisabled\b/.test(match[0]);
    elements.set(el.id, el);
}
for (const match of html.matchAll(/<select id="([^"]+)">([\s\S]*?)<\/select>/g)) {
    const first = match[2].match(/<option value="([^"]+)"/);
    if (first) elements.get(match[1]).value = first[1];
}
elements.get("cd-data").textContent = payload;
const document = {
    getElementById(id) { assert.ok(elements.has(id), `Missing element ${id}`); return elements.get(id); },
    createElement(tag) { return new Element(tag); },
    createTextNode(text) { return {textContent: text}; },
};
const sandbox = {
    document, window: {devicePixelRatio: 2}, console,
    atob: text => Buffer.from(text, "base64").toString("binary"),
    requestAnimationFrame: fn => frames.push(fn),
    ResizeObserver: class { constructor(fn) { this.fn = fn; } observe() { this.fn(); } },
};
vm.runInNewContext(source, sandbox, {timeout: 15000, filename: "contour-viewer.js"});
function flush() { while (frames.length) frames.shift()(); }
function event(id, type, extra = {}) {
    const fn = elements.get(id).listeners.get(type);
    assert.ok(fn, `${id} should handle ${type}`);
    fn({button: 0, pointerId: 1, preventDefault() {}, ...extra});
    flush();
}
function change(id, value) { elements.get(id).value = value; event(id, "change"); }
flush();
const scene = elements.get("scene"), context = scene.context;
assert.match(elements.get("summary").textContent, /offline viewer/);
assert.equal(elements.get("view").value, "contours");
const first = Object.keys(data.methods)[0];
const validCount = name => [...Buffer.from(data.methods[name].valid, "base64")].reduce((a, b) => a + b, 0);
assert.equal(context.paths, validCount(first));

change("view", "intensity");
assert.equal(elements.get("method").disabled, true);
assert.match(elements.get("legend-title").textContent, /original intensity units/);
assert.equal(context.paths, 0);
assert.ok(context.image.context.pixels);
if (data.count) {
    const labelBytes = Buffer.from(data.labels, "base64"), meanBytes = Buffer.from(data.means, "base64");
    const labels = new Int32Array(labelBytes.buffer, labelBytes.byteOffset, labelBytes.length / 4);
    const means = new Float64Array(meanBytes.buffer, meanBytes.byteOffset, meanBytes.length / 8);
    const pixel = labels.findIndex(id => id >= 0);
    assert.ok(pixel >= 0);
    assert.equal(context.image.context.pixels.data[pixel * 4 + 3], 255);
    let low = Math.min(...means), high = Math.max(...means);
    if (low === high) { const delta = Math.max(Math.abs(low) * .01, .01); low -= delta; high += delta; }
    const index = Math.max(0, Math.min(255, Math.round(255 * (means[labels[pixel]] - low) / (high - low))));
    assert.deepEqual([...context.image.context.pixels.data.slice(pixel * 4, pixel * 4 + 3)], data.palette[index]);
}
change("view", "cd");
assert.equal(elements.get("method").disabled, false);
change("metric", "area_px2");
assert.match(elements.get("legend-title").textContent, /Contour area/);
change("view", "contours");
elements.get("compare").checked = true;
event("compare", "change");
assert.equal(context.paths, Object.keys(data.methods).reduce((n, name) => n + validCount(name), 0));
assert.equal(elements.get("color-legend").hidden, true);

const originalZoom = parseFloat(elements.get("zoom-readout").textContent);
event("zoom-in", "click");
assert.ok(parseFloat(elements.get("zoom-readout").textContent) > originalZoom);
event("scene", "wheel", {clientX: 480, clientY: 270, deltaY: -100});
const beforePan = [...context.translation];
event("scene", "pointerdown", {clientX: 480, clientY: 270});
event("scene", "pointermove", {clientX: 520, clientY: 300});
event("scene", "pointerup", {clientX: 520, clientY: 300});
assert.notDeepEqual(context.translation, beforePan);
event("fit", "click");
assert.equal(parseFloat(elements.get("zoom-readout").textContent), originalZoom);
if (data.count) {
    const id = Math.min(1335, data.count - 1);
    elements.get("cell-id").value = String(id);
    event("select-cell", "click");
    assert.match(elements.get("cell-meta").textContent, new RegExp(`Cell ${id}`));
    assert.match(elements.get("cell-meta").textContent, /Mean intensity/);
    assert.equal(elements.get("focus").disabled, false);
    event("focus", "click");
    assert.ok(parseFloat(elements.get("zoom-readout").textContent) > originalZoom);
    elements.get("vertices").checked = true;
    event("vertices", "change");
    assert.ok(context.vertices > 0);
    assert.equal(elements.get("measurements").querySelector("tbody").children.length, Object.keys(data.methods).length * 5);
}
console.log(JSON.stringify({cells: data.count, checks: ["contour paths", "intensity map", "CD metric switching", "method comparison", "zoom", "wheel", "pan", "reset", "cell selection", "vertex display"], result: "passed"}, null, 2));
