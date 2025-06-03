import numpy as np
import imageio
import random
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

def compute_channel_distortion(spatial, target_qf, robustness_qfs):
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
    input_path = "test_img/img.jpg"
    output_path = "test_img/stego_img.jpg"
    img = load_image(input_path)
    
    # 2. 计算信道失真代价
    robustness_qfs = range(71, 96)
    distortion = compute_channel_distortion(img, 70, robustness_qfs)
    
    # 3. 转换为DCT系数
    coeffs, q_table = image_to_dct(img,use_quantization=False)
    
    # 4. 准备STC编码
    stcode = [7,11]
    nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
    original_message = generate_message(nzAC, payload=0.1)
    
    # 5. STC嵌入
    stc = STC(stcode, coeffs.shape[0] // 8)
    stego_coeffs = stc.embed(coeffs, distortion.copy(), original_message)
    
    # 6. 生成载密图像
    stego_img = dct_to_image(stego_coeffs,use_quantization=False)
    imageio.imwrite(output_path, stego_img)
    
    # 7. 提取测试
    stego_img = load_image(output_path)
    stego_coeffs, _ = image_to_dct(stego_img)
    extracted_message = stc.extract(stego_coeffs)[:(len(original_message))]
    
    # 8. 计算误码率
    ber = calculate_ber(original_message, extracted_message)
    print(f"误码率(BER): {ber:.6f}")

    # 测试仅仅使用dct和idct
    stego_img = dct_to_image(stego_coeffs)
    stego_coeffs, _ = image_to_dct(stego_img)
    extracted_message2 = stc.extract(stego_coeffs)[:(len(original_message))]
    
    # 8. 计算误码率
    ber = calculate_ber(extracted_message, extracted_message2)
    print(f"误码率(BER): {ber:.6f}")


