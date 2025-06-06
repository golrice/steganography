import imageio
import numpy as np

input_path = "test_img/12.pgm"
output_path = "test_img/img.jpg"
img = imageio.v2.imread(input_path)
if len(img.shape) == 3:
    img = np.mean(img, axis=2).astype(np.uint8)
imageio.imwrite(output_path,img)
    