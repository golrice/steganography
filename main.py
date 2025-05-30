#!/usr/bin/env python3
import os
import sys
import glob
import imageio
import scipy.fftpack
import scipy.signal
import numpy as np
import jpeg_toolbox as jt
from multiprocessing import Pool
from STC import STC
import random
import tempfile
import uuid
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging():
    """设置日志记录"""
    log_filename = f"embed_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def compute_channel_distortion(spatial, target_qf, robustness_qfs):
    """
    计算信道失真代价
    :param spatial: 空域图像数据
    :param target_qf: 目标质量因子
    :param robustness_qfs: 用于鲁棒性测试的质量因子范围
    :return: 信道失真代价矩阵
    """
    # 将空域图像转换为JPEG系数
    height, width = spatial.shape
    coeffs = np.zeros((height, width), dtype=np.float64)
    
    # 计算原始DCT系数(未量化)
    original_dct_coeffs = np.zeros_like(coeffs, dtype=np.float64)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = spatial[i:i+8, j:j+8]
            original_dct_coeffs[i:i+8, j:j+8] = dct2(block - 128)
    
    # 计算Δ(i) - 通过多次重压缩计算系数的平均变化
    delta = np.zeros_like(original_dct_coeffs, dtype=np.float64)
    c_abs = np.abs(original_dct_coeffs)
    
    for qf in robustness_qfs:
        # 使用唯一临时文件名
        temp_path = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4().hex}.jpg")
        try:
            imageio.imwrite(temp_path, spatial.astype(np.uint8), quality=qf)
            
            # 重新加载并计算DCT系数
            recompressed = imageio.imread(temp_path)
            if len(recompressed.shape) == 3:
                recompressed = np.mean(recompressed, axis=2)
            recompressed = recompressed.astype(np.float64)
            
            # 计算重压缩后的DCT系数
            recompressed_dct_coeffs = np.zeros_like(original_dct_coeffs, dtype=np.float64)
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    block = recompressed[i:i+8, j:j+8]
                    recompressed_dct_coeffs[i:i+8, j:j+8] = dct2(block - 128)
            
            # 计算绝对值变化并累加
            delta += np.abs(np.abs(recompressed_dct_coeffs) - c_abs)
        finally:
            # 确保临时文件被删除
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # 计算平均值
    delta /= len(robustness_qfs)
    
    # 计算最终信道失真代价 d(i) = Δ(i) + 1/abs(c(i))
    epsilon = 1e-10
    channel_distortion = delta + 1.0 / (c_abs + epsilon)
    
    # 对于DC系数设置极高的代价
    channel_distortion[::8, ::8] = 1e13
    
    return channel_distortion

def embed_message(spatial, payload, target_qf):
    """
    在图像中嵌入消息
    :param spatial: 空域图像数据
    :param payload: 嵌入率(bits per non-zero AC coefficient)
    :param target_qf: 目标JPEG质量因子
    :return: 载密图像, 原始消息
    """
    height, width = spatial.shape
    
    # 1. 计算信道失真代价
    robustness_qfs = range(target_qf+1, 96) if target_qf < 95 else [95]
    channel_distortion = compute_channel_distortion(spatial, target_qf, robustness_qfs)
    
    # 2. 将图像转换为JPEG系数
    coeffs = np.zeros((height, width), dtype=np.int16)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = spatial[i:i+8, j:j+8]
            coeffs[i:i+8, j:j+8] = np.round(dct2(block - 128))
    
    # 3. 准备STC编码
    stcode = [71, 109]
    nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
    message_length = int(round(payload * nzAC))
    
    # 生成随机消息(固定种子以便测试)
    random.seed(123)
    original_message = bytes([random.randint(0, 255) for _ in range((message_length + 7) // 8)])
    
    # 4. 使用STC进行嵌入
    stc = STC(stcode, coeffs.shape[0] // 8)
    rho_p1 = channel_distortion.copy()
    rho_m1 = channel_distortion.copy()
    stego_coeffs = stc.embed(coeffs, rho_p1, original_message)
    
    # 5. 将系数转换回空域图像
    stego_spatial = np.zeros_like(spatial, dtype=np.uint8)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = stego_coeffs[i:i+8, j:j+8]
            stego_spatial[i:i+8, j:j+8] = np.clip(idct2(block) + 128, 0, 255).astype(np.uint8)
    
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
    coeffs = np.zeros((height, width), dtype=np.int16)
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = attacked_spatial[i:i+8, j:j+8]
            coeffs[i:i+8, j:j+8] = np.round(dct2(block - 128))
    
    # 准备STC解码
    stcode = [71, 109]
    nzAC = np.count_nonzero(coeffs) - np.count_nonzero(coeffs[::8, ::8])
    message_length = int(round(payload * nzAC))
    
    stc = STC(stcode, coeffs.shape[0] // 8)
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
            error_bits += 8
    
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
        stego_image, original_msg = embed_message(image, payload, target_qf)
        
        # 3. JPEG攻击
        for qf in range(target_qf+1, 96) if target_qf < 95 else [95]:
            attacked_image = jpeg_attack(stego_image, qf)
        
        # 4. 提取消息
        extracted_msg = extract_message(attacked_image, payload, target_qf)
        
        # 5. 计算BER
        ber = calculate_ber(original_msg, extracted_msg)

        print(f"嵌入率 {payload} {image_path}: 误码率={ber}")
        logging.info(f"嵌入率 {payload} {image_path}: 误码率={ber}")
        
        return os.path.basename(image_path), ber
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        logging.error(f"Error processing {image_path}: {str(e)}")
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

    setup_logging()
    
    pgm_dir = "BossBase-1.01"
    output_csv = "res.csv"
    
    # 实验参数设置
    payloads = [0.1, 0.2, 0.3]  # 嵌入率(bits per non-zero AC coefficient)
    target_qf = 70   # 目标JPEG质量因子,即用于计算失真代价函数的JPEG质量因子，若选70则会计算71到95的jpeg压缩的信道失真
    attack_qf = 70    # 攻击JPEG质量因子，同上，选70则会使用71到95的jpeg压缩攻击
    
    # 批量处理图像
    batch_process(pgm_dir, output_csv, payloads, target_qf, attack_qf)
    
    # 分析结果
    analyze_results(output_csv)