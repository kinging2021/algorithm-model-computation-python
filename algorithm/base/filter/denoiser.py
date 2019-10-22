import pywt
import numpy as np
from scipy.signal import savgol_filter


def wavelet_filter(data, wavefunc="db4", lv=5, m=1, n=5):
    coeff = pywt.wavedec(data, wavefunc, mode='sym', level=lv)

    def sgn(x):
        return 1 if x > 0 else -1 if x < 0 else 0  # sgn函数

    # 去噪过程
    for i in range(m, n + 1):  # 选取小波系数层数为 m 至 n 层
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零
    denoised_data = pywt.waverec(coeff, wavefunc)
    if len(denoised_data) > len(data):
        denoised_data = denoised_data[:-1]
    assert (len(data) == len(denoised_data))
    return denoised_data


def sav_gol_filter(data):
    return savgol_filter(data, 5, 2, mode="nearest")
