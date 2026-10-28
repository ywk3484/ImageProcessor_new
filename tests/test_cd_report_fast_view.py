"""Numerical summaries keep the report's maps, statistics and per-hole values."""

import base64
import json
import shutil
import subprocess

import numpy as np
import pytest

from scripts import cd_report as report
from scripts.benchmark_cd import lattice
from subpx import measure_cds
from subpx.cd import LOCAL_METHODS, PIXEL_METHODS


METHODS = ("local_half_area", "local_half_area_min_avg", "local_tail_area", *PIXEL_METHODS)


def _array(encoded, dtype=np.float32):
    return np.frombuffer(base64.b64decode(encoded), dtype=dtype)


@pytest.fixture(scope="module")
def mixed():
    image = lattice((64, 96), 10.0, 4.4, 30.0, seed=7)[0]
    result = measure_cds(image, methods=(*report.ALL_METHODS, *LOCAL_METHODS), backend="cpu", area_max=200,
                         pixel_size=21, unit="nm", n_angles=128, half_area_level=0.42, tail_area_level=0.16)
    return image, result


def test_mixed_detail_keeps_raw_inputs_separate_and_preserves_original_brightness(mixed):
    _, result = mixed
    detail = report.detail_data(result)
    local = detail["local"]
    raw = result.measurements["local_projection"].inputs
    shape = raw["cells"].shape
    assert (local["h"], local["w"]) == shape[1:]
    codes = _array(local["clean"]["codes"], np.uint8).reshape(shape)
    lo = _array(local["clean"]["lo"])[:, None, None]
    step = _array(local["clean"]["step"])[:, None, None]
    np.testing.assert_array_equal(codes != 255, raw["inside"])
    np.testing.assert_array_equal(_array(local["ox"], np.int32), raw["origins"][:, 0])
    np.testing.assert_array_equal(_array(local["oy"], np.int32), raw["origins"][:, 1])
    residual = np.abs((lo + codes * step) - raw["cells"])
    assert np.all((residual <= 0.51 * step + 1e-4)[raw["inside"]])
    level = result.measurements["local_projection"].metrics["background"]
    cleaned = np.where(raw["inside"], raw["cells"] - level[:, None, None], 0.0)
    np.testing.assert_array_equal(_array(local["projection_level"]), level.astype(np.float32))
    for key, axis in (("prof_x", 1), ("prof_y", 2)):
        np.testing.assert_array_equal(_array(local[key]), cleaned.sum(axis=axis).astype(np.float32).ravel())
        model = np.where(result.masks, result.rois, 0.0).sum(axis=axis)
        np.testing.assert_array_equal(_array(detail[key]), model.astype(np.float32).ravel())
    for name in METHODS:
        panel = detail["half"][name]
        measured = result.measurements[name]
        assert panel["input"] == "local" and panel["factor"] == (1 if name in PIXEL_METHODS else 4)
        assert panel["level_fraction"] == (0.16 if name.endswith("tail_area") else 0.42)
        assert panel["background"].startswith("minimum" if name.endswith("min_avg") else "median")
        if name.endswith("min_avg") or name.endswith("tail_area"):
            assert ("original pixels" in panel["peak_rule"]) == (name in PIXEL_METHODS)
        for key, metric in (("level", "local_level"), ("peak", "peak_signal"), ("threshold", "threshold"),
                            ("mean", "mean_signal")):
            np.testing.assert_array_equal(_array(panel[key]), measured.metrics[metric].astype(np.float32))
        np.testing.assert_array_equal(_array(panel["cx"]), measured.geometry["center_local"][:, 0].astype(np.float32))
    data = report.hole_data(result, report.layer_values(result))
    model = result.deblend_model
    assert data["peak_label"] == "fitted spot amplitude"
    np.testing.assert_array_equal(_array(data["peak"]), model["params"][model["report"], 0].astype(np.float32))
    assert data["methods"]["cell_intensity"]["base"] == "tail_area"
    assert all(data["methods"][name]["outline_kind"] == "measured" for name in LOCAL_METHODS)


def test_inverted_local_projection_detail_uses_signal_polarity():
    image = lattice((48, 64), 10.0, 4.4, 30.0, seed=3)[0]
    result = measure_cds(255 - image, methods=("projection", "local_projection"), backend="cpu", area_max=200,
                         invert=True)
    local = report.detail_data(result)["local"]
    measured = result.measurements["local_projection"]
    raw = measured.inputs
    level = -measured.metrics["background"]
    assert len(level) > 0 and np.any(np.abs(level) > 1)
    np.testing.assert_array_equal(_array(local["projection_level"]), level.astype(np.float32))
    cleaned = np.where(raw["inside"], raw["cells"] - level[:, None, None], 0.0)
    np.testing.assert_array_equal(_array(local["prof_x"]), cleaned.sum(axis=1).astype(np.float32).ravel())


def test_empty_cells_have_serializable_details():
    result = measure_cds(np.zeros((24, 32)), methods=(*report.ALL_METHODS, *LOCAL_METHODS), backend="cpu")
    detail = report.detail_data(result)
    assert detail["clean"] == {"codes": "", "lo": "", "step": ""}
    # No raw inputs exist when there are no detected holes.
    assert "local" not in detail
    json.dumps(detail, allow_nan=False)


@pytest.fixture()
def numerical():
    image = lattice((64, 96), 10.0, 4.4, 30.0, seed=7)[0]
    result = measure_cds(image, methods=METHODS, backend="cpu", area_max=200, pixel_size=21, unit="nm",
                         summary_only=True)
    return image, result


def test_summary_map_uses_explicit_image_and_real_peak_without_inventing_outlines(numerical):
    image, result = numerical
    layers = report.layer_values(result)
    with pytest.raises(ValueError, match="image="):
        report.hole_data(result, layers)
    data = report.hole_data(result, layers, "test", image=image)
    assert (data["height"], data["width"], data["n"]) == (*image.shape, len(result.seeds_xy))
    assert data["scale"] == 21 and data["unit"] == "nm"
    assert data["peak_label"] == "local_half_area peak above local level"
    peak = np.frombuffer(base64.b64decode(data["peak"]), np.float32)
    np.testing.assert_allclose(peak, result.measurements["local_half_area"].metrics["peak_signal"], rtol=1e-6)
    for name in METHODS:
        assert data["methods"][name]["outline_kind"] == "equivalent_circle"
        assert "rays_file" not in data["methods"][name]
        assert data["methods"][name + "_x"]["base"] == name
        assert name in report.DESCRIPTIONS and name in report.COLORS
    intensity = data["methods"]["cell_intensity"]
    assert intensity["base"] == "local_tail_area" and intensity["unit"] == "counts"
    assert "raw cell, neighbour light retained" in intensity["label"]
    json.dumps(data, allow_nan=False)


def test_intensity_selects_an_available_tail_method_and_keeps_its_validity(numerical):
    image, result = numerical
    tail = result.measurements["local_tail_area"]
    expected = np.where(tail.valid, tail.metrics["mean_signal"], np.nan)
    np.testing.assert_array_equal(report.cell_intensity(result), expected)
    del result.measurements["local_tail_area"]
    assert report.intensity_source(result.measurements) == "pixel_tail_area"
    data = report.hole_data(result, report.layer_values(result), image=image)
    assert data["methods"]["cell_intensity"]["base"] == "pixel_tail_area"
    result.measurements["tail_area"] = tail
    assert report.intensity_source(result.measurements) == "tail_area"
    data = report.hole_data(result, report.layer_values(result), image=image)
    assert "background and neighbours removed" in data["methods"]["cell_intensity"]["label"]
    del result.measurements["tail_area"], result.measurements["pixel_tail_area"]
    assert report.cell_intensity(result) is None


def test_summary_report_keeps_statistics_and_inserts_reference_section_first(numerical, tmp_path):
    image, result = numerical
    stats = report.cd_statistics(result)
    entries = [{"id": "one", "name": "one.png", "holes": len(result.seeds_xy), "stats": stats,
                "width": image.shape[1], "height": image.shape[0], "seconds": 0.1, "image": "images/one.png",
                "data": "data/one.js", "detail": None, "tail": None, "agreement": report.agreement(result),
                "intensity": report._spread(report.cell_intensity(result)), "image_mean": float(image.mean())}]
    extra = "<h3>Reference comparison</h3><p>Matched image means</p>"
    path = report.write_report(tmp_path, entries, (*METHODS, "half_area", "tail_area"), [], "nm", extra_summary=extra)
    text = path.read_text(encoding="utf-8")
    assert text.index(extra) < text.index("<h3>Between images: do the images show the same mean CD?")
    assert "local_tail_area's region (raw cell, neighbour light retained)" in text
    assert "equivalent-CD circles; measured outlines were not stored" in text
    assert "<td>\u2013</td>" in text and "<td>\u2013/td>" not in text
    assert "<b>half_area</b>" in text and "<b>tail_area</b>" in text
    assert all(name in text for name in METHODS)
    values = result.measurements["local_half_area"].metrics["cd_equivalent_px"]
    valid = result.measurements["local_half_area"].valid
    assert stats["local_half_area"]["mean"] == np.mean(values[valid] * 21)
    # Baseline statistics may share the table while intensity is still from the
    # fast result; the explicit source keeps its raw-cell meaning.
    entries[0]["stats"]["tail_area"] = stats["local_tail_area"]
    entries[0]["intensity_source"] = "local_tail_area"
    path = report.write_report(tmp_path, entries, (*METHODS, "tail_area"), [], "nm")
    text = path.read_text(encoding="utf-8")
    assert "local_tail_area's region (raw cell, neighbour light retained)" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to exercise the report's browser functions")
def test_viewer_handles_empty_rays_and_uses_cd_geometry_for_intensity_maps():
    draw = report._HTML.split("function draw(){", 1)[1].split("// Measured outlines:", 1)[0]
    detail = report._HTML.split("function renderDetail(){", 1)[1].split("let drag=null;", 1)[0]
    source = {"draw": "function draw(){" + draw, "detail": "function renderDetail(){" + detail}
    script = r"""
const source=JSON.parse(require('fs').readFileSync(0,'utf8'));
const arcs=[], detail={innerHTML:''}, ctx=new Proxy({}, {get:(_,key)=>key==='arc'?
    ((x,y,r)=>arcs.push(r)):(()=>{}), set:()=>true});
const $=id=>id==='method'?{value:'cell_intensity'}:id==='show'?{checked:true}:detail;
const base={x:[10],y:[10],cd:[4],valid:[1],status:[0],status_names:['ok'],
    rays:new Uint8Array(),ray_n:0,ray_unit:1,outline_kind:'equivalent_circle'};
const light={...base,base:'pixel_tail_area',cd:[123],unit:'counts',scale:1};
let cur={n:1,scale:21,unit:'nm',methods:{pixel_tail_area:base,cell_intensity:light}},
    curId='one', img={}, view={k:2,tx:0,ty:0}, bins=[[0]], sel=0;
const canvas={width:100,height:100}, RAY_ANG={}, COLORS={}, ENTRIES={one:{detail:null}},
    window={CD_DETAIL:{}}, fin=Number.isFinite, viridis=()=>'', loadRays=()=>{};
eval(source.draw);eval(source.detail);draw();renderDetail();
if(arcs[0]!==2||arcs.some(r=>r>10))throw Error('intensity was used as a circle diameter');
if(!detail.innerHTML.includes('123.000 counts')||!detail.innerHTML.includes('ok'))throw Error('missing hole values');
if(!detail.innerHTML.includes('Cell images were not stored'))throw Error('missing explanation');
"""
    subprocess.run([shutil.which("node"), "-e", script], input=json.dumps(source), text=True,
                   capture_output=True, check=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed to exercise the report's browser functions")
def test_fast_detail_panels_render_their_actual_raw_cells_and_projection_profiles(mixed):
    _, result = mixed
    selected = np.flatnonzero(np.logical_and.reduce([result.measurements[name].valid for name in LOCAL_METHODS]))[0]
    raw = result.measurements["local_projection"].inputs
    payload = {
        "prepare": "function prepareDetail(D)" + report._HTML.split("function prepareDetail(D)", 1)[1].split("function drawCell", 1)[0],
        "render": "function renderDetail(){" + report._HTML.split("function renderDetail(){", 1)[1].split("let drag=null;", 1)[0],
        "data": report.hole_data(result, report.layer_values(result)), "detail": report.detail_data(result),
        "selected": int(selected), "methods": METHODS,
        "raw": report._b64(raw["cells"][selected], "<f4"),
        "model": report._b64(np.where(result.masks[selected], result.rois[selected], np.nan), "<f4"),
    }
    script = r"""
const p=JSON.parse(require('fs').readFileSync(0,'utf8'));
function unpack(x,T){const b=Buffer.from(x,'base64');return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
let cur=p.data,curId='one',sel=p.selected,method='local_projection',drawn=[],profiles=[];
for(const m of Object.values(cur.methods)){
  m.cd=unpack(m.cd,Float32Array);
  if(!m.base){m.x=unpack(m.x,Float32Array);m.y=unpack(m.y,Float32Array);m.valid=unpack(m.valid,Uint8Array);
    m.status=unpack(m.status,m.status_wide?Uint16Array:Uint8Array);}}
for(const [k,m] of Object.entries(cur.methods))if(m.base)cur.methods[k]={...cur.methods[m.base],...m};
const panel={innerHTML:''},$=id=>id==='method'?{value:method}:panel,fin=Number.isFinite,
  window={CD_DETAIL:{one:p.detail}},ENTRIES={one:{detail:'one.detail.js'}},COLORS={},
  outline=(...args)=>args,drawCell=(id,G,vals,overlay)=>drawn.push({id,G,vals,overlay}),
  profilePlots=(id,prof,edges)=>profiles.push({id,prof,edges});
eval(p.prepare);eval(p.render);
function checkCell(expected,subtract=0){
  const a=drawn[0],tol=.51*a.G.clean.step[sel]+1e-4;
  if(a.vals.length!==expected.length)throw Error('wrong cell dimensions');
  for(let j=0;j<expected.length;j++)if(fin(expected[j])){
    if(Math.abs(a.vals[j]-(expected[j]-subtract))>tol)throw Error('wrong displayed cell input');
  }else if(fin(a.vals[j]))throw Error('outside-cell pixel became visible');
}
const raw=unpack(p.raw,Float32Array),model=unpack(p.model,Float32Array);
renderDetail();
const L=p.detail.local,level=L.projection_level[sel],P=cur.methods.local_projection;
checkCell(raw,level);
if(drawn[0].G!==L||!panel.innerHTML.includes('median edge level; neighbour light remains'))throw Error('wrong projection source');
if(Math.abs(drawn[0].overlay[0]-(P.x[sel]-L.ox[sel]))>1e-6)throw Error('wrong raw-cell origin');
for(const [j,key,size] of [[0,'prof_x',L.w],[1,'prof_y',L.h]])
  if(profiles[j].prof.some((v,k)=>v!==L[key][sel*size+k]))throw Error('profile did not use saved full-precision sums');
for(const name of p.methods){
  method=name;drawn=[];profiles=[];renderDetail();checkCell(raw);
  if(drawn[0].G!==L||!panel.innerHTML.includes('fitted background and neighbours have not been removed'))throw Error('wrong area source');
  const native=name.startsWith('pixel_');
  if(native&&!panel.innerHTML.includes('original pixels, without enlargement'))throw Error('native grid not explained');
  if(native&&panel.innerHTML.includes('enlarged 4x'))throw Error('native pixels described as enlarged');
  if(!native&&!panel.innerHTML.includes('enlarged 4x'))throw Error('local interpolation not explained');
}
method='projection';drawn=[];profiles=[];renderDetail();checkCell(model);
if(drawn[0].G!==p.detail||!panel.innerHTML.includes('background and neighbours removed'))throw Error('original projection changed');
method='half_area';drawn=[];renderDetail();checkCell(model);
if(drawn[0].G!==p.detail||!panel.innerHTML.includes('background and neighbours removed'))throw Error('original area changed');
"""
    subprocess.run([shutil.which("node"), "-e", script], input=json.dumps(payload), text=True,
                   capture_output=True, check=True)
