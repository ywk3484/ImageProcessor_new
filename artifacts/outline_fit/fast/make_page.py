"""Scratch: builds docs/outline_fit_one_pass.html from page_template.html, runs/page_data.json (make_page_data.py)
and the picture runs/fig_average_hole.png. Every number in the page's text comes from the measurements here.

    python make_page.py
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs"

# seconds per page measured on the 40 hole pages of the test file, 20 pages per call (run_file.py, checks.py load,
# util.py): the middle of the repeats and their range
L1_MS, L1_RANGE = 80.0, "78–89"
L2_MS, L2_RANGE = 137.0, "129–151"
GPU_BUSY = (0.67, 0.72)   # share of the time the GPU is busy (util.py), page values / with sizes
REMOTE_GPU = 2.3          # remote GPU against this PC's for the other CD methods (3.5 s against 7.9 s per page)
PAGES = 1500


def duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def main():
    data = json.loads((RUNS / "page_data.json").read_text())
    pic = json.loads((RUNS / "fig_average_hole.json").read_text())
    s = data["file_summary"]
    dense = {"circle60_dense", "circle50_dense", "ellipse90x60_stagger"}
    both = [r for r in data["truth"] if "existing" in r and r["name"] not in dense]
    closer = [r for r in both if abs(r["one_pass"]["error"]) < abs(r["existing"]["error"])]
    other = [r for r in both if r not in closer]
    worst_other = max((abs(r["one_pass"]["error"]) - abs(r["existing"]["error"]) for r in other), default=0.0)
    ph = s["phase_ms"]
    ref_s = s["existing_seconds"]

    def signed(v, digits=2):
        return f"{v:+.{digits}f}".replace("-", "−")

    def find(name, family="lens", **extra):
        rows = [r for r in data["truth"] if r["name"] == name and r["family"] == family and r["extra"] == extra]
        assert len(rows) == 1, (name, extra, len(rows))
        return rows[0]

    lg = [r["one_pass_lens"] - r["one_pass"] for r in data["file"]]
    lg_mean = sum(lg) / len(lg)
    lg_sd = (sum((v - lg_mean) ** 2 for v in lg) / len(lg)) ** 0.5

    def remote(ms, busy):
        fast = ms * ((1 - busy) + busy / REMOTE_GPU) / 4 * PAGES / 1e3
        slow = ms / 4 * PAGES / 1e3
        return f"{fast:.0f}–{slow:.0f}"

    tokens = dict(
        l1_ms=f"{L1_MS:.0f}", l2_ms=f"{L2_MS:.0f}", l1_s=f"{L1_MS / 1e3:.2f}", l2_s=f"{L2_MS / 1e3:.2f}",
        busy=f"{GPU_BUSY[0] * 100:.0f} to {GPU_BUSY[1] * 100:.0f}", diff_abs=f"{abs(s['diff_mean']):.2f}",
        ell_op=signed(find("ellipse160x80_square")["one_pass"]["error"]),
        ell_ex=signed(find("ellipse160x80_square")["existing"]["error"]),
        stag_op=signed(find("circle100_stagger")["one_pass"]["error"]),
        stag_ex=signed(find("circle100_stagger")["existing"]["error"]),
        cd5=signed(find("circle100_square", cd="5.0")["one_pass"]["error"], 3),
        part_op=signed(find("circle100_square", part="0.4")["one_pass"]["error"], 3),
        part_ex=signed(find("circle100_square", part="0.4")["existing"]["error"]),
        lens_gauss=f"{signed(lg_mean, 3)} ± {lg_sd:.3f}",
        l1_range=L1_RANGE, l2_range=L2_RANGE,
        l1_image=duration(L1_MS * PAGES / 1e3), l2_image=duration(L2_MS * PAGES / 1e3),
        l1_stripe=duration(4 * L1_MS * PAGES / 1e3), l2_stripe=duration(4 * L2_MS * PAGES / 1e3),
        l1_four=f"{L1_MS * PAGES / 4e3:.0f}", l2_four=f"{L2_MS * PAGES / 4e3:.0f}",
        l1_four_stripe=duration(L1_MS * PAGES / 1e3), l2_four_stripe=duration(L2_MS * PAGES / 1e3),
        l1_remote=remote(L1_MS, GPU_BUSY[0]), l2_remote=remote(L2_MS, GPU_BUSY[1]),
        ref_s=f"{ref_s:.0f}", ref_image=duration(ref_s * PAGES), ref_stripe=duration(4 * ref_s * PAGES),
        n_compared=str(len(both)), n_closer=str(len(closer)), n_other=str(len(other)), worst_other=f"{max(0.01, math.ceil(worst_other * 100) / 100):.2f}",
        pages=str(s["pages"]), diff_mean=f"{s['diff_mean']:+.2f}".replace("-", "−"), diff_sd=f"{s['diff_sd']:.2f}",
        diff_max=f"{s['diff_max']:.2f}", one_pass_sd=f"{s['one_pass_sd']:.3f}", existing_sd=f"{s['existing_sd']:.3f}",
        hole_sd=f"{s['hole_sd']:.2f}", hole_slope=f"{s['hole_slope']:.3f}", centre_sd=f"{s['centre_sd']:.3f}",
        window=str(pic["window"]), holes=f"{pic['holes']:,}", leftover=f"{pic['leftover_rms']:.2f}",
        noise_alone=f"{pic['noise_alone']:.2f}",
        t_a=f"{ph['A stack']:.0f}", t_fit_a=f"{ph['A page fit']:.0f}", t_b=f"{ph['B hole pass + stack']:.0f}",
        t_fit_c=f"{ph['C page fit']:.0f}", t_d=f"{ph['D hole pass']:.0f}",
        png=base64.b64encode((RUNS / "fig_average_hole.png").read_bytes()).decode("ascii"),
    )
    html = (HERE / "page_template.html").read_text(encoding="utf-8")
    for key, value in tokens.items():
        assert "{{" + key + "}}" in html, key
        html = html.replace("{{" + key + "}}", value)
    left = [part.split("}}")[0] for part in html.split("{{")[1:]]
    assert not left, left
    html = html.replace("/*DATA*/", json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/"))
    out = ROOT / "docs" / "outline_fit_one_pass.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    for key in ("l1_image", "l2_image", "l1_four", "l2_four", "l1_remote", "l2_remote", "ref_image", "ref_stripe", "n_compared",
                "n_closer", "n_other", "worst_other", "diff_mean", "diff_sd", "diff_max"):
        print(f"  {key} = {tokens[key]}")
    print("  not closer:", [(r["name"], r["family"], r["extra"], round(r["one_pass"]["error"], 3), round(r["existing"]["error"], 3)) for r in other])


if __name__ == "__main__":
    main()
