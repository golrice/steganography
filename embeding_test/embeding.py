import numpy as np
import imageio
import random
import pywt
import os
import uuid
import tempfile
from STC import STC
from dct_tool import image_to_dct, dct_to_image

def load_image(path):
    """加载图像并转为灰度"""
    img = imageio.v2.imread(path)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    return img

def generate_message(nzAC, payload):
    """生成随机消息"""
    message_length = min(int(round(payload * nzAC)), nzAC)
    random.seed(123)
    return bytes([random.randint(0, 255) for _ in range((message_length + 7) // 8)])

def compute_robustness_penalty(orig_coeffs, comp_coeffs):
    """
    计算鲁棒性惩罚项
    """
    penalty = np.zeros_like(orig_coeffs)
    delta = np.abs(orig_coeffs - comp_coeffs)
    
    # 频率加权矩阵
    freq_weight = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9,10],
        [4, 5, 6, 7, 8, 9,10,11],
        [5, 6, 7, 8, 9,10,11,12],
        [6, 7, 8, 9,10,11,12,13],
        [7, 8, 9,10,11,12,13,14]
    ])
    h, w = orig_coeffs.shape
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = delta[i:i+8, j:j+8]
            penalty[i:i+8, j:j+8] = block * (1 + freq_weight/14.0)
            penalty[i,j] *= 5.0  # DC系数
                
    return penalty

def j_uniward_distortion(spatial_img):
    """
    J-UNIWARD失真计算
    输入空域图像 spatial_img
    输出与DCT域的失真 distortion
    """
    # 输入验证
    if spatial_img.ndim != 2:
        raise ValueError("Input image must be 2D grayscale")
    
    h, w = spatial_img.shape
    distortion = np.zeros((h, w), dtype=np.float64)
    
    # 小波分解
    try:
        wavelet = 'db8'
        levels = 3
        W = pywt.wavedec2(spatial_img, wavelet, level=levels)
        
        # 计算各子带权重
        for level in range(levels + 1):
            if level == 0:
                # 低频子带
                subband = W[0]
                weights = np.ones_like(subband) * 0.1
                
                # 计算上采样尺寸
                upsampled = np.kron(weights, np.ones((2**levels, 2**levels)))
                
                # 安全地叠加到失真矩阵
                up_h, up_w = upsampled.shape
                y_start = (h - up_h) // 2
                x_start = (w - up_w) // 2
                
                if y_start >= 0 and x_start >= 0:
                    distortion[y_start:y_start+up_h, x_start:x_start+up_w] += upsampled
                
            else:
                # 高频子带
                for orientation in range(3):
                    subband = W[level][orientation]
                    sigma = np.std(subband)
                    weights = 1.0 / (sigma + 1e-6)
                    
                    # 计算上采样尺寸
                    upsampled = np.kron(weights, np.ones((2**(levels-level+1), 2**(levels-level+1))))
                    
                    # 安全地叠加到失真矩阵
                    up_h, up_w = upsampled.shape
                    y_start = (h - up_h) // 2
                    x_start = (w - up_w) // 2
                    
                    if y_start >= 0 and x_start >= 0:
                        distortion[y_start:y_start+up_h, x_start:x_start+up_w] += upsampled
        
        # 归一化
        if distortion.max() > distortion.min():
            distortion = (distortion - distortion.min()) / (distortion.max() - distortion.min())
            
    except Exception as e:
        print(f"Wavelet transform warning: {str(e)}")
        return np.zeros((h, w))
    
    return distortion

def compute_channel_distortion_j_uniward(img, robustness_qfs):
    """
    基于jpeg_toolbox的鲁棒性损失计算函数
    
    参数:
        img_path: 输入JPEG图像路径
        qf: 初始嵌入质量因子
        robustness_qfs: 需要抵抗的JPEG压缩质量因子列表
        
    返回:
        distortion: 每个DCT系数的修改损失矩阵 (形状同DCT系数数组)
    """
    y_coeffs,_ = image_to_dct(img,use_quantization=False)
    
    distortion = j_uniward_distortion(img)
    
    for robustness_qf in robustness_qfs:
        # 保存为临时JPEG文件（模拟压缩）
        coe,_ = image_to_dct(img,qf=robustness_qf)
        
        # 重新加载压缩后的系数
        compressed_img = dct_to_image(coe)
        compressed_coeffs,_ = image_to_dct(compressed_img,use_quantization=False)
        
        # 计算鲁棒性惩罚项
        delta_distortion = compute_robustness_penalty(y_coeffs, compressed_coeffs)
        distortion += delta_distortion
    
    return distortion

def compute_channel_distortion(spatial, robustness_qfs):
    """
    计算信道失真代价（改进版）
    :param spatial: 空域图像数据 (uint8)
    :param target_qf: 目标质量因子
    :param robustness_qfs: 用于鲁棒性测试的质量因子范围
    :return: 信道失真代价矩阵
    """
    height, width = spatial.shape
    spatial_float = spatial.astype(np.float64) - 128  # JPEG标准中心化
    
    # 标准JPEG亮度量化表（根据QF缩放）
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float64)

    # 计算原始DCT系数（未量化）
    original_dct,_ = image_to_dct(spatial_float, False)
    q_original_dct,_ = image_to_dct(spatial_float)
    
    # 计算Δ(i)：通过模拟重压缩的系数变化
    delta = np.zeros_like(original_dct)
    c_abs = np.abs(original_dct)
    
    for qf in robustness_qfs:
        # 根据QF缩放量化表（标准JPEG缩放公式）
        scale = 5000 / qf if qf < 50 else 200 - 2 * qf
        scaled_Q = np.floor((Q * scale + 50) / 100)
        scaled_Q[scaled_Q < 1] = 1
        
        # 模拟量化/反量化过程（避免临时文件）
        quantized_dct = np.zeros_like(original_dct)
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = q_original_dct[i:i+8, j:j+8]
                quantized = np.round(block / scaled_Q) * scaled_Q  # 量化+反量化
                quantized_dct[i:i+8, j:j+8] = quantized
        
        # 计算系数绝对值变化
        delta += np.abs(quantized_dct - q_original_dct)
    
    # 计算平均值并加上稳定性项
    delta /= len(robustness_qfs)
    epsilon = 1e-10
    channel_distortion = delta + 1.0 / (c_abs + epsilon)
    
    # 标记所有DC系数（左上角系数）为高代价
    channel_distortion[::8, ::8] = 1e13
    
    return channel_distortion

def compute_channel_distortion_fixed(spatial, robustness_qfs):
    # 使用固定代价
    cost = np.array([
        [100, 100,  99,  50,  50,  50,  50,  50],
        [100, 100,  99,  50,  40,  40,  40,  40],
        [100, 100,  99,  60,  50,  50,  50,  50],
        [100, 100,  95,  60,  50,  60,  55,  55],
        [100, 100,  80,  70,  70,  70,  70,  75],
        [100, 100,  80,  70,  70,  70,  70,  75],
        [100, 100,  80,  70,  70,  70,  70,  75],
        [100, 100,  60,  60,  65,  60,  55, 999],
    ])
    height, width = img.shape
    distortion = np.zeros_like(img)
    for i in range(0, height, 8):
            for j in range(0, width, 8):
                 distortion[i:i+8, j:j+8] = cost
    return cost

def calculate_ber(original_msg, extracted_msg):
    """
    计算比特错误率
    :param original_msg: 原始消息
    :param extracted_msg: 提取的消息
    :return: 比特错误率
    """
    error_bits = 0
    total_bits = len(original_msg) * 8
    
    for i in range(len(extracted_msg)):
        if i < len(original_msg):
            xor_result = original_msg[i] ^ extracted_msg[i]
            error_bits += bin(xor_result).count('1')
        else:
            error_bits += 8
    
    if len(original_msg) > len(extracted_msg):
        error_bits += 8 * (len(original_msg) - len(extracted_msg))
    
    print(f"error_bits = {error_bits}, total_bits = {total_bits}")
    return error_bits / total_bits if total_bits > 0 else 1.0

if __name__ == "__main__":
    # 1. 加载图像
    input_path = "test_img\img.jpg"
    output_path = "test_img/stego_img.jpg"
    img = load_image(input_path)
    
    # 2. 计算信道失真代价
    # 使用JPEG平均代价
    robustness_qfs = range(71, 96)
    
    distortion = compute_channel_distortion(img, robustness_qfs)
    # distortion = compute_channel_distortion_j_uniward(img, robustness_qfs)
    # distortion = compute_channel_distortion_fixed(img, robustness_qfs)

    # 3. 转换为DCT系数
    coeffs, q_table = image_to_dct(img,use_quantization=True)
    
    # 4. 准备STC编码
    stcode = [7,11]
    nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
    original_message = generate_message(nzAC, payload=0.1)
    
    # 5. STC嵌入
    stc = STC(stcode, coeffs.shape[0] // 8)
    stego_coeffs = stc.embed(coeffs, distortion.copy(), original_message)
    
    # 6. 生成载密图像
    stego_img = dct_to_image(stego_coeffs,use_quantization=True)
    imageio.imwrite(output_path, stego_img)
    
    # 7. 提取测试
    # 用jpeg压缩保存并重新提取
    stego_img = load_image(output_path)
    stego_coeffs1, _ = image_to_dct(stego_img)
    extracted_message = stc.extract(stego_coeffs1)[:(len(original_message))]
    ber = calculate_ber(original_message, extracted_message)
    print(f"jpeg压缩后的误码率(BER): {ber:.6f} \n")

    # 测试仅仅使用dct和idct
    stego_img = dct_to_image(stego_coeffs)
    stego_coeffs2, _ = image_to_dct(stego_img)
    extracted_message2 = stc.extract(stego_coeffs2)[:(len(original_message))]
    ber = calculate_ber(original_message, extracted_message2)
    print(f"dct量化并逆变换后的误码率(BER): {ber:.6f}\n")

    # 测试不做任何操作直接提取
    extracted_message3 = stc.extract(stego_coeffs)[:(len(original_message))]
    ber = calculate_ber(original_message, extracted_message3)
    print(f"不做变换直接提取的误码率(BER): {ber:.6f}\n")


