import numpy as np
import imageio
import random
import scipy.signal
from STC import STC
from dct_tool import image_to_dct, dct_to_image,idct2
import jpeg_toolbox as jt


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

def j_uniward_distortion(spatial, jpg):

    hpdf = np.array([
        -0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,  
         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,  
         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,  
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ])        

    sign = np.array([-1 if i%2 else 1 for i in range(len(hpdf))])
    lpdf = hpdf[::-1] * sign

    F = []
    F.append(np.outer(lpdf.T, hpdf))
    F.append(np.outer(hpdf.T, lpdf))
    F.append(np.outer(hpdf.T, hpdf))


    # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
    spatial_impact = {}
    for i in range(8):
        for j in range(8):
            test_coeffs = np.zeros((8, 8))
            test_coeffs[i, j] = 1
            spatial_impact[i, j] = idct2(test_coeffs) * jpg["quant_tables"][0][i, j]

    # Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
    wavelet_impact = {}
    for f_index in range(len(F)):
        for i in range(8):
            for j in range(8):
                wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.) # XXX


    # Create reference cover wavelet coefficients (LH, HL, HH)
    pad_size = 16 # XXX
    spatial_padded = np.pad(spatial, (pad_size, pad_size), 'symmetric')
    #print(spatial_padded.shape)

    RC = []
    for i in range(len(F)):
        f = scipy.signal.correlate2d(spatial_padded, F[i], mode='same', boundary='fill')
        RC.append(f)


    coeffs = jpg["coef_arrays"][0]
    k, l = coeffs.shape
    nzAC = np.count_nonzero(jpg["coef_arrays"][0]) - np.count_nonzero(jpg["coef_arrays"][0][::8, ::8])

    rho = np.zeros((k, l))
    tempXi = [0.]*3
    sgm = 2**(-6)

    # Computation of costs
    for row in range(k):
        for col in range(l):
            mod_row = row % 8
            mod_col = col % 8
            sub_rows = list(range(row-mod_row-6+pad_size-1, row-mod_row+16+pad_size))
            sub_cols = list(range(col-mod_col-6+pad_size-1, col-mod_col+16+pad_size))

            for f_index in range(3):
                RC_sub = RC[f_index][sub_rows][:,sub_cols]
                wav_cover_stego_diff = wavelet_impact[f_index, mod_row, mod_col]
                tempXi[f_index] = abs(wav_cover_stego_diff) / (abs(RC_sub)+sgm)

            rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
            rho[row, col] = np.sum(rho_temp)


    wet_cost = 10**13
    rho_m1 = rho.copy()
    rho_p1 = rho.copy()

    rho_p1[rho_p1>wet_cost] = wet_cost
    rho_p1[np.isnan(rho_p1)] = wet_cost
    rho_p1[coeffs>1023] = wet_cost

    rho_m1[rho_m1>wet_cost] = wet_cost
    rho_m1[np.isnan(rho_m1)] = wet_cost
    rho_m1[coeffs<-1023] = wet_cost

    return np.minimum(rho_p1, rho_m1)  # 选择代价较小的修改方式
    # return (rho_p1 + rho_m1) / 2 
    # return np.maximum(rho_p1, rho_m1)

def compute_channel_distortion(spatial, robustness_qfs):
    """
    计算信道失真代价
    :param spatial: 空域图像数据 (uint8)
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

    # 计算原始DCT系数
    original_dct,_ = image_to_dct(spatial_float, False)
    q_original_dct,_ = image_to_dct(spatial_float, True, qf=95)
    
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
    
    # # 标记所有DC系数（左上角系数）为高代价
    # channel_distortion[::8, ::8] = 1e13

    # 增强代价控制：按频率分级
    for i in range(8):
        for j in range(8):
            if i + j >= 4:  # 抑制中高频（AC4及以上）
                channel_distortion[i::8, j::8] *= 10  # 权重提高
            if i + j >= 6:  # 完全禁止高频（AC6及以上）
                channel_distortion[i::8, j::8] = 1e13

    return channel_distortion

def coefficient_adjustment(original_coeffs, stego_coeffs, Qo, Qc):
    """
    系数调整方案 - 核心算法
    :param original_coeffs: 原始图像的DCT系数
    :param stego_coeffs: 隐写图像的DCT系数
    :param Qo: 原始质量因子
    :param Qc: 信道质量因子
    :return: 中间图像的DCT系数
    """
    # 获取量化表（这里简化处理，实际应使用标准JPEG量化表）
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
    
    # 根据质量因子缩放量化表
    scale_o = 5000 / Qo if Qo < 50 else 200 - 2 * Qo
    scale_c = 5000 / Qc if Qc < 50 else 200 - 2 * Qc
    Qo_scaled = np.floor((Q * scale_o + 50) / 100)
    Qc_scaled = np.floor((Q * scale_c + 50) / 100)
    Qo_scaled[Qo_scaled < 1] = 1
    Qc_scaled[Qc_scaled < 1] = 1
    
    intermediate_coeffs = np.zeros_like(original_coeffs)
    h, w = original_coeffs.shape
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            # 对每个8x8块进行处理
            O_block = original_coeffs[i:i+8, j:j+8]
            S_block = stego_coeffs[i:i+8, j:j+8]
            I_block = np.zeros((8, 8))
            
            for x in range(8):
                for y in range(8):
                    O_val = O_block[x, y]
                    S_val = S_block[x, y]
                    
                    if Qo == Qc:
                        I_block[x, y] = S_val
                    else:
                        # 计算调整量α
                        alpha = 0
                        min_diff = float('inf')
                        
                        # 搜索最优整数α（论文引理1）
                        for a in range(-5, 6):  # 限制搜索范围提高效率
                            temp = O_val + a
                            compressed_val = round(temp * Qo_scaled[x, y] / Qc_scaled[x, y])
                            diff = abs(compressed_val - S_val)
                            
                            if diff < min_diff:
                                min_diff = diff
                                alpha = a
                                if diff == 0:
                                    break
                        
                        I_block[x, y] = O_val + alpha
            
            intermediate_coeffs[i:i+8, j:j+8] = I_block
    
    return intermediate_coeffs

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
            error_bits += 0
    
    if len(original_msg) > len(extracted_msg):
        error_bits += 8 * (len(original_msg) - len(extracted_msg))
    
    print(f"error_bits = {error_bits}, total_bits = {total_bits}")
    return error_bits / total_bits if total_bits > 0 else 1.0

def embeding(input_path,output_path,payload):
    img = load_image(input_path)
    jpg = jt.load(input_path)
    
    # 2. 计算信道失真代价
    # 使用JPEG平均代价
    robustness_qfs = range(71,96)
    
    # distortion = compute_channel_distortion(img, robustness_qfs)
    distortion = j_uniward_distortion(img,jpg)

    # 3. 转换为DCT系数
    coeffs, q_table = image_to_dct(img)
    
    # 4. 准备STC编码
    stcode = [7,11,15]
    nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
    original_message = generate_message(nzAC, payload=payload)
    # 嵌入encoded_msg而非original_msg
    
    # 5. STC嵌入
    stc = STC(stcode, coeffs.shape[0] // 8)
    stego_coeffs = stc.embed(coeffs, distortion.copy(), original_message)

    
    # 6. 生成载密图像
    stego_img = dct_to_image(stego_coeffs ,use_quantization=True,scaled_Q=q_table)
    imageio.imwrite(output_path, stego_img,quality=70)
    elapse = img-stego_img
    imageio.imwrite("test_img/elapse.jpg",elapse)
    return original_message,stc

if __name__ == "__main__":
    # 1. 加载图像
    input_path = "test_img/img.jpg"
    stego_path = "test_img/stego_img.jpg"
    payload = 0.1
    
    original_message, stc = embeding(input_path, stego_path, payload)


    # 7. 提取测试
    # 用jpeg压缩保存并重新提取
    stego_img = load_image(stego_path)
    stego_coeffs1, _ = image_to_dct(stego_img)
    extracted_message = stc.extract(stego_coeffs1)[:(len(original_message)-5)]
    ber = calculate_ber(original_message[:(len(original_message)-5)], extracted_message)
    print(f"jpeg压缩后的误码率(BER): {ber:.6f} \n")

    # 测试仅仅使用dct和idct
    stego_coeffs2, q_table = image_to_dct(stego_img)
    extracted_message2 = stc.extract(stego_coeffs2)[:(len(original_message)-5)]
    ber = calculate_ber(original_message[:(len(original_message)-5)], extracted_message2)
    print(f"dct量化并逆变换后的误码率(BER): {ber:.6f}\n")

    # # 测试不做任何操作直接提取
    # extracted_message3 = stc.extract(stego_coeffs)[:(len(original_message)-5)]
    # ber = calculate_ber(original_message[:(len(original_message)-5)], extracted_message3)
    # print(f"不做变换直接提取的误码率(BER): {ber:.6f}\n")