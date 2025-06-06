#!/usr/bin/env python3
import os
import glob
import imageio
import numpy as np
import jpeg_toolbox as jt
import scipy.signal
from multiprocessing import Pool
from STC import STC
import random
import tempfile
import uuid
from tqdm import tqdm
from dct_tool import image_to_dct,dct_to_image,idct2

stcode = [7,11,15]
k = 64

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
    epsilon = 1e-13
    channel_distortion = delta + 1.0 / (c_abs + epsilon)
    
    # 标记所有DC系数（左上角系数）为高代价
    channel_distortion[::8, ::8] = 1e13
    
    return channel_distortion

def embed_message(spatial, payload, target_qf, attack_qf):
    """
    在图像中嵌入消息
    :param spatial: 空域图像数据
    :param payload: 嵌入率(bits per non-zero AC coefficient)
    :param target_qf: 目标JPEG质量因子
    :return: 载密图像, 原始消息
    """
    
    temp_path = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4().hex}.jpg")
    try:
        imageio.imwrite(temp_path, spatial)
        jpg = jt.load(temp_path)
    finally:
        # 确保临时文件被删除
        if os.path.exists(temp_path):
            os.remove(temp_path)
    channel_distortion = j_uniward_distortion(spatial,jpg)


    # 1. 计算信道失真代价
    # robustness_qfs = range(target_qf, 96) if target_qf < 95 else [95]
    # channel_distortion = compute_channel_distortion(spatial, robustness_qfs)
    

    
    # 2. 将图像转换为JPEG系数
    coeffs, Q_table = image_to_dct(spatial,use_quantization=True,qf=target_qf)
    
    # 3. 准备STC编码
    nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
    message_length = int(round(payload * nzAC))
    
    # 生成随机消息(固定种子以便测试)
    random.seed(123)
    original_message = bytes([random.randint(0, 255) for _ in range((message_length + 7) // 8)])
    
    # 4. 使用STC进行嵌入
    # stc = STC(stcode, coeffs.shape[0] // 8)
    stc = STC(stcode, k)
    stego_coeffs = stc.embed(coeffs, channel_distortion.copy(), original_message)
    
    # 5. 将系数转换回空域图像
    stego_spatial = dct_to_image(stego_coeffs,use_quantization=True,scaled_Q=Q_table)
    
    return stego_spatial, original_message

def extract_message(attacked_spatial, payload, target_qf):
    """
    从攻击后的图像中提取消息
    :param attacked_spatial: 攻击后的空域图像
    :param payload: 嵌入率
    :param target_qf: 目标JPEG质量因子
    :return: 提取的消息
    """
    height, width = attacked_spatial.shape
    
    # 将图像转换为JPEG系数
    coeffs,_ = image_to_dct(attacked_spatial,qf=target_qf)
    
    # 准备STC解码
    nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
    message_length = int(round(payload * nzAC))
    
    # stc = STC(stcode, coeffs.shape[0] // 8)
    stc = STC(stcode, k)
    extracted_message = stc.extract(coeffs)
    
    # 截取正确长度的消息
    extracted_message = extracted_message[:(message_length + 7) // 8]
    
    return extracted_message

def jpeg_attack(image, qf):
    """
    模拟JPEG压缩攻击
    :param image: 输入图像
    :param qf: JPEG质量因子
    :return: 压缩后的图像
    """
    # 使用唯一临时文件名
    temp_path = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4().hex}.jpg")
    try:
        imageio.imwrite(temp_path, image, quality=qf)
        attacked = imageio.imread(temp_path)
        
        if len(attacked.shape) == 3:
            attacked = np.mean(attacked, axis=2)
        return attacked.astype(np.uint8)
    finally:
        # 确保临时文件被删除
        if os.path.exists(temp_path):
            os.remove(temp_path)

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
    
    return error_bits / total_bits if total_bits > 0 else 1.0

def process_image(args):
    """
    处理单个图像的工作函数
    :param args: (image_path, payload, target_qf, attack_qf)
    :return: (图像文件名, BER)
    """
    image_path, payload, target_qf, attack_qf = args
    
    try:
        # 1. 读取PGM图像
        image = imageio.imread(image_path)
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)
        
        # 2. 嵌入消息
        stego_image, original_msg = embed_message(image, payload, target_qf, attack_qf)
        
        # 3. JPEG攻击
        for qf in range(96,target_qf+1,-1) if target_qf < 95 else [95]:
            attacked_image = jpeg_attack(stego_image, qf)
        # attacked_image = jpeg_attack(stego_image, attack_qf)
        
        # 4. 提取消息
        extracted_msg = extract_message(attacked_image, payload, target_qf)
        
        # 5. 计算BER
        ber = calculate_ber(original_msg, extracted_msg)

        print(f"嵌入率 {payload} {image_path}: 误码率={ber}")
        
        return os.path.basename(image_path), ber
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return os.path.basename(image_path), 1.0  # 出错时返回最大BER

def batch_process(pgm_dir, output_csv, payloads, target_qf, attack_qf, num_workers=4):
    """
    批量处理PGM图像
    :param pgm_dir: PGM图像目录
    :param output_csv: 结果输出CSV文件
    :param payloads: 嵌入率列表
    :param target_qf: 目标质量因子(单个值)
    :param attack_qf: 攻击质量因子(单个值)
    :param num_workers: 并行工作进程数
    """
    # 获取所有PGM文件
    pgm_files = glob.glob(os.path.join(pgm_dir, "*.pgm"))
    if not pgm_files:
        print(f"No PGM files found in {pgm_dir}")
        return
    
    # 准备结果文件
    with open(output_csv, 'w') as f:
        header = "Filename,Payload,TargetQF,AttackQF,BER\n"
        f.write(header)
    
    # 为每个组合创建任务
    tasks = []
    for payload in payloads:
        for pgm_file in pgm_files:
            tasks.append((pgm_file, payload, target_qf, attack_qf))
    
    # 使用多进程并行处理
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))
    
    # 修正的结果写入部分
    with open(output_csv, 'a') as f:
        for i, (filename, ber) in enumerate(results):
            # 直接从tasks中获取对应的参数
            payload = tasks[i][1]  # 每个任务的第二个元素是payload
            line = f"{filename},{payload},{target_qf},{attack_qf},{ber:.6f}\n"
            f.write(line)
    
    print(f"Processing completed. Results saved to {output_csv}")

def analyze_results(csv_path):
    """
    分析结果并生成统计报告
    :param csv_path: 结果CSV文件路径
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # 计算平均BER和完全正确提取率
    stats = df.groupby(['Payload', 'TargetQF', 'AttackQF']).agg({
        'BER': ['mean', 'std'],
        'Filename': lambda x: sum(x == 0) / len(x)  # 完全正确提取率
    })
    
    print("\nStatistical Results:")
    print(stats)
    
    # 保存统计结果
    stats_path = os.path.splitext(csv_path)[0] + "_stats.csv"
    stats.to_csv(stats_path)
    print(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python robust_stc_pgm.py <pgm_directory> <output_csv>")
    #     print("Example: python robust_stc_pgm.py ./pgm_images results.csv")
    #     sys.exit(1)
    
    # pgm_dir = sys.argv[1]
    # output_csv = sys.argv[2]
    
    pgm_dir = "BossBase-1.01"
    output_csv = "res.csv"
    
    # 实验参数设置
    payloads = [0.1, 0.2, 0.3]  # 嵌入率(bits per non-zero AC coefficient)
    target_qf = 70      # 目标JPEG质量因子,即用于计算失真代价函数的JPEG质量因子，若选70则会计算71到95的jpeg压缩的信道失真
    attack_qf = 80      # 攻击JPEG质量因子，同上，选70则会使用71到95的jpeg压缩攻击
    
    # 批量处理图像
    batch_process(pgm_dir, output_csv, payloads, target_qf, attack_qf)
    
    # 分析结果
    analyze_results(output_csv)