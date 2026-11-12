#!/bin/sh
# Scratch: every measurement behind docs/outline_fit_one_pass.html with the prototype's defaults (two centring
# passes, three size passes x 1.3). The existing fit's runs (runs/truth_main.json, truth_more.json,
# truth_extra_existing.json, truth_part.json, ref/) are not repeated: they do not depend on the prototype.
PY=/e/PythonProjects/ImageProcessor_new/.venv/Scripts/python.exe
cd /e/PythonProjects/ImageProcessor_new/artifacts/outline_fit/fast
$PY -X utf8 truth_matrix.py set=main existing=0 out=truth_main_v4 > runs/truth_main_v4.log 2>&1
$PY -X utf8 truth_matrix.py set=blur existing=0 out=truth_blur_v4 > runs/truth_blur_v4.log 2>&1
$PY -X utf8 truth_matrix.py set=more existing=0 out=truth_more_v4 > runs/truth_more_v4.log 2>&1
$PY -X utf8 run_file.py batch=20 level=2 lines=0 repeat=3 out=file_l2_v4 > runs/file_l2_v4.log 2>&1
$PY -X utf8 run_file.py batch=20 level=2 family=lens lines=0 repeat=2 out=file_l2_lens_v4 > runs/file_l2_lens_v4.log 2>&1
$PY -X utf8 run_file.py batch=20 level=1 lines=0 repeat=3 out=file_l1_v3 > runs/file_l1_v3.log 2>&1
$PY -X utf8 checks.py load > runs/check_load.log 2>&1
$PY -X utf8 checks.py repeat > runs/check_repeat.log 2>&1
$PY -X utf8 util.py > runs/check_util.log 2>&1
$PY -X utf8 truth_gpu.py circle100_square lens=0 blur=63 pitch=178.5 place=3.6 cd=2.6 variants=fast,truth > runs/truth_copy.log 2>&1
# added with the gauss2 exchange step (the end points of the fit, small images, the pair reach): about 25 minutes
$PY -X utf8 small_pages.py existing=1 > runs/small_pages_v5.log 2>&1
$PY -X utf8 port_checks.py > runs/port_checks_v5.log 2>&1
$PY -X utf8 basins.py pages=c100,c80,c100b72,c80b72,file out=basins_a > runs/check_basins_a.log 2>&1
$PY -X utf8 basins.py pages=stag,sq150,ell,c250 out=basins_b > runs/check_basins_b.log 2>&1
$PY -X utf8 reach.py ends > runs/check_reach_ends.log 2>&1
$PY -X utf8 reach.py time > runs/check_reach_time.log 2>&1
$PY -X utf8 ring.py > runs/check_ring.log 2>&1
echo done > runs/rerun_all.done
