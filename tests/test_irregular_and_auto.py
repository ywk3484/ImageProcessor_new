import numpy as np

from maskproc import recommend_refine_method
from maskproc.spectra import periodicity_lombscargle


def test_recommend_refine_method_threshold():
    assert recommend_refine_method(5, 5) == "logquad"
    assert recommend_refine_method(20, 18) == "edge_gradmoment"


def test_periodicity_lombscargle_detects_known_period():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 200.0, size=300))
    true_period = 20.0
    y = 1.5 * np.sin(2 * np.pi * x / true_period) + 0.1 * rng.normal(size=x.size)
    spec = periodicity_lombscargle(x, y, freq_min=1/100, freq_max=1/5, n_freq=4000, detrend_order=None)
    i = int(np.nanargmax(spec.amplitude))
    est = float(spec.period[i])
    assert abs(est - true_period) < 1.0
