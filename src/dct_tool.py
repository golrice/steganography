import scipy.fftpack
import numpy as np



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

def image_to_dct(img, use_quantization=True, qf=70):
    """图像转DCT系数"""
    height, width = img.shape
    coeffs = np.zeros((height, width), dtype=np.float64)
    
    # 根据质量因子缩放量化表
    scale = 5000 / qf if qf < 50 else 200 - 2 * qf
    scaled_Q = np.clip(np.floor((Q * scale + 50) / 100), 1, None)
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = img[i:i+8, j:j+8].astype(np.float64) - 128  # JPEG中心化
            dct_block = dct2(block)
            
            if use_quantization:
                coeffs[i:i+8, j:j+8] = np.round(dct_block / scaled_Q)  # 量化
            else:
                coeffs[i:i+8, j:j+8] = dct_block
    
    return coeffs, scaled_Q if use_quantization else None

def dct_to_image(coeffs, use_quantization=True):
    """DCT系数转图像"""
    height, width = coeffs.shape
    img = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = coeffs[i:i+8, j:j+8]
            
            if use_quantization:
                block = block * Q  # 反量化
            
            img[i:i+8, j:j+8] = np.clip(idct2(block) + 128, 0, 255).astype(np.uint8)
    
    return img