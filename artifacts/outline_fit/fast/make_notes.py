"""Scratch: writes the measurements section of docs/outline_fit_one_pass.md from runs/page_data.json and the logs'
numbers (the constants below are copied from runs/file_l*_v3.log, check_load.log, check_util.log, check_repeat.log,
bench2.py, stable2.py, checks.py other, seed_test.py). The section sits between two marker lines.

    python make_notes.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
DOC = ROOT / "docs" / "outline_fit_one_pass.md"
BEGIN, END = "<!-- measurements: begin (make_notes.py) -->", "<!-- measurements: end -->"


def signed(v, d=3):
    return f"{v:+.{d}f}"


def label(r):
    x = r["extra"]
    parts = []
    if "blur" in x:
        parts.append(f"blur {float(x['blur']):.0f} nm")
    if "rot" in x:
        parts.append("grid turned 3 deg")
    if "flicker" in x:
        parts.append("brightness wave +-10 %")
    if "part" in x:
        parts.append("holes on 40 % of the page")
    if "pitch" in x:
        parts.append("8.5 px pitch, placement 3.6 nm")
    if "cd" in x:
        parts.append(f"sizes spread {float(x['cd']):.1f} nm")
    if "bright" in x:
        parts.append("brightness spread 15 %")
    return r["label"] + (" (" + ", ".join(parts) + ")" if parts else "")


def main():
    data = json.loads((HERE / "runs" / "page_data.json").read_text())
    s = data["file_summary"]
    ph = s["phase_ms"]
    out = []
    out.append("### 5.1 Time\n")
    out.append("The 40 hole pages of the test file (39,042 or 37,842 holes each), 20 pages per call, three repeats.\n")
    out.append("| | ms per page | 1,500-page image, this GPU | four such GPUs (not measured) |")
    out.append("|---|---|---|---|")
    out.append("| Page values (A to C) | 80 (78 to 89) | 2.0 min | 30 s |")
    out.append("| With a size per hole (A to D) | 137 (129 to 151) | 3.4 min | 51 s |")
    out.append("| Existing fit, library | 104,000 to 150,000 | 2 to 2.5 days | |")
    out.append(f"| Existing fit, same-numbers prototype of the earlier proposal | {s['existing_seconds'] * 1e3:,.0f} (9,400 to 14,700) | 4.6 h | |")
    out.append("")
    out.append("By phase, ms per page: " + ", ".join(f"{k} {v:.1f}" for k, v in ph.items()) + ".")
    out.append("Inside them: upload 0.5, seeds 3.9, spacing 1.0, stacks 7.4, pair maps 6.2, both page fits 15.4 at 20 pages "
               "per call, templates 12.3, hole-pass set-up 11.9, five sweeps 60.3, noise line 1.3, results 2.4.")
    out.append("One sweep (39,042 holes, window 15 x 15 px, render 23 x 23 px): `hole_normal` 6.3 ms, `render_scatter` "
               "4.5 ms, `hole_update` 0.5 ms.")
    out.append("Pages per call 4 / 10 / 20 / 40: 111 / 89 / 80 / 87 ms (page values), 164 / 144 / 136 / 134 ms (with sizes).")
    out.append("GPU busy 67 % (page values) and 72 % (with sizes); one CPU core busy all the time; GPU memory pool 2.8 to "
               "3.0 GB at 20 pages per call, 5.5 GB at 40.")
    out.append("Remote estimate on the page (19 to 30 s and 30 to 51 s per image): lower end = GPU share of the time x "
               "1 / 2.3 (the remote's ratio for the other CD methods, 3.5 s against 7.9 s per page), CPU share unchanged, four "
               "GPUs; upper end = this PC's time / 4.\n")
    out.append("### 5.2 Generated pages with known sizes\n")
    out.append("`subpx.synthetic_arrays.render_hole_array`, 2048 x 1378 px, 21 nm pixels, blur 65 nm with the lens term "
               "45 nm, sizes spread 2.5 nm, placement 1 nm, brightness 5 %, photon noise 30 per level, unless said. "
               "Error = page CD - mean true CD of the stacked holes (squares: against the area-equivalent diameter). "
               "Response = slope of the fitted against the true hole CD; scatter in nm of true CD; light = move for a "
               "10 % brighter hole. Existing = the existing fit on the same page (`*` = not settled in 14 rounds).\n")
    out.append("| Page | Blur form | One pass: error | response | scatter | light | Existing: error | response | scatter | light |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in data["truth"]:
        o, e = r["one_pass"], r.get("existing")
        line = (f"| {label(r)} | {r['family']} | {signed(o['error'])} | {o['slope']:.3f} | {o['scatter']:.2f} | "
                f"{signed(o['light10'])} |")
        if e:
            star = "" if e["converged"] else " *"
            line += f" {signed(e['error'])}{star} | {e['slope']:.3f} | {e['scatter']:.2f} | {signed(e['light10'])} |"
        else:
            line += " | | | |"
        out.append(line)
    out.append("")
    out.append("Copy of the test file (100 nm circles, pitch 8.5 px, placement 3.6 nm, Gaussian blur 63 nm, sizes spread "
               "2.6 nm), gauss2: one pass +0.022 nm (first page fit +0.165, page fit with the true centres +0.044), response "
               "1.002, scatter 1.06 nm; existing -0.036 nm, response 0.994, scatter 1.04 nm (14 rounds, not settled).\n")
    out.append("### 5.3 The test file (40 hole pages, gauss2)\n")
    out.append(f"- Page CD: one pass {s['one_pass_mean']:.3f} nm (sd over the pages {s['one_pass_sd']:.3f}), existing "
               f"{s['existing_mean']:.3f} (sd {s['existing_sd']:.3f}); difference {signed(s['diff_mean'])} +- {s['diff_sd']:.3f}, "
               f"largest {s['diff_max']:.3f} nm; correlation 0.84.")
    out.append(f"- Per hole against the existing fit: CD difference {s['hole_sd']:.3f} nm rms about the page's offset, "
               f"response {s['hole_slope']:.4f}, centres {s['centre_sd']:.3f} nm apart.")
    lg = [r["one_pass_lens"] - r["one_pass"] for r in data["file"]]
    m = sum(lg) / len(lg)
    sd = (sum((v - m) ** 2 for v in lg) / len(lg)) ** 0.5
    out.append(f"- Lens form with one pass: page CD {signed(m)} +- {sd:.3f} nm from gauss2's. The existing fit's lens "
               "results on pages 0, 12, 24, 36 (earlier session): 100.488, 100.575, 100.691, 100.664; one pass: 100.661, "
               "100.731, 100.774, 100.723.")
    out.append("- Repeats (pages 0, 5, 24, 30, both forms): page CD equal to 0.0002 nm, hole CDs to 0.0006 nm rms "
               "(one hole 0.06 nm).\n")
    out.append("### 5.4 Other checks\n")
    out.append("- Centring passes against the settled page CD (pages 0 and 24): one pass 0.017 / 0.022 nm short, two "
               "0.003 / 0.004, three 0.0003 / 0.002; a further round of B and C after two passes adds 0.002 to 0.003 nm.")
    out.append("- Seeds on the 13 generated patterns: all holes on 10; missed 0.3 % (50 nm dense), 1.2 % (300 nm squares), "
               "0.01 % (70 x 45 nm dense); no strays; 2 to 6 ms.")
    out.append("- Pages of the test file without the hole pattern: blank pages 3, 43, 47: about 27,700 seeds, page CD 50 nm, "
               "blur 14 nm, chi-square per bin 1.2; page 46 (another pattern): 5,845 seeds, chi-square per bin 227.")
    text = "\n".join(out)
    doc = DOC.read_text(encoding="utf-8")
    if "@@MEASUREMENTS@@" in doc:
        doc = doc.replace("@@MEASUREMENTS@@", BEGIN + "\n" + text + "\n" + END)
    else:
        a, b = doc.index(BEGIN), doc.index(END)
        doc = doc[:a] + BEGIN + "\n" + text + "\n" + doc[b:]
    DOC.write_text(doc, encoding="utf-8")
    print(f"wrote the measurements into {DOC} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
