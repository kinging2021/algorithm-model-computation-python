import numpy as np

from .denoiser import wavelet_filter, sav_gol_filter
from .utils import check_results


def test_wavelet_fileter():
    check_results(wavelet_filter([1, 2, 2, 3], wavefunc="db4", lv=5, m=1, n=5),
                  np.array([2.4617611, 2.34601375, 2.30436428, 2.36340651]))

def test_sg_filter():
    check_results(sav_gol_filter([1, 2, 2, 3]),
                  np.array([1.25714286, 1.65714286, 2.34285714, 2.74285714]))
