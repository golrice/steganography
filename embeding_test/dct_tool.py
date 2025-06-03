import scipy.fftpack
import numpy as np
from scipy.ndimage import median_filter

# 标准JPEG量化表（亮度）
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def post_process(img, size=1):
    #不好用，会导致图像模糊
    return median_filter(img, size=size)  # 中值滤波去块效应[1,7](@ref)


def adaptive_quantization(dct_block, scaled_Q, qf):
    """基于图像块频率特性的自适应量化"""
    # 计算高频能量（右下角4x4区域）
    high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))

    # 计算频谱重心（频率分布特征）[1](@ref)
    u, v = np.meshgrid(range(8), range(8))
    total_energy = np.sum(np.abs(dct_block))
    if total_energy > 1e-5:  # 避免除零
        freq_center = np.sum(u * np.abs(dct_block)) / total_energy
    else:
        freq_center = 3.5  # 默认中频

    # 动态调整量化步长 [1,2,6](@ref)
    if high_freq_energy > 1000 or freq_center > 5:  # 高频丰富区域
        return np.round(dct_block / (scaled_Q * 0.7))  # 精细量化
    elif high_freq_energy < 300 and freq_center < 3:  # 低频平滑区域
        return np.round(dct_block / (scaled_Q * 1.3))  # 粗量化
    else:  # 中频区域
        return np.round(dct_block / scaled_Q)

def image_to_dct(img, use_quantization=True, qf=70):
    """图像转DCT系数"""
    height, width = img.shape   #获取图像大小
    coeffs = np.zeros((height, width), dtype=np.float64)   #根据图像大小生成一个全零np array

    # 根据质量因子缩放量化表
    scale = 5000 / qf if qf < 50 else 200 - 2 * qf
    scaled_Q = np.clip(np.floor((Q * scale + 50) / 100), 1, 255)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = img[i:i + 8, j:j + 8].astype(np.float64) - 128
            dct_block = dct2(block)

            if use_quantization:

                quant_block = adaptive_quantization(dct_block, scaled_Q, qf)

                coeffs[i:i + 8, j:j + 8] = quant_block
            else:
                coeffs[i:i + 8, j:j + 8] = dct_block

    return coeffs, scaled_Q if use_quantization else None

def dct_to_image(coeffs, use_quantization=True,scaled_Q=None):
    """DCT系数转图像"""
    height, width = coeffs.shape
    img = np.zeros((height, width), dtype=np.uint8)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = coeffs[i:i+8, j:j+8]

            if use_quantization:
                if scaled_Q is not None:
                    block = block * scaled_Q
                else:
                    block = block * Q  # 反量化

            img[i:i+8, j:j+8] = np.clip(idct2(block) + 128, 0, 255).astype(np.uint8)

    #img = post_process(img)

    return img