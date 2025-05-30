from dct_tool import dct_to_image,image_to_dct
import imageio
import numpy as np

def load_image(path):
    """加载图像并转为灰度"""
    img = imageio.v2.imread(path)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    return img

input_path = "test_img/img.jpg"
output_path = "test_img/elapse.jpg"
img = load_image(input_path)
coeffs, q_table = image_to_dct(img)
stego_img = dct_to_image(coeffs)
elapse = img - stego_img
imageio.imwrite(output_path, elapse)
