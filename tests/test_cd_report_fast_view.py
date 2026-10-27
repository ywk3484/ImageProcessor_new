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
from subpx.cd import PIXEL_METHODS


METHODS = ("local_half_area", "local_half_area_min_avg", "local_tail_area", *PIXEL_METHODS)


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
