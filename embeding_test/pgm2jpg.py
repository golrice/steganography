import imageio
import numpy as np

input_path = "1.pgm"
output_path = "img.jpg"
img = imageio.v2.imread(input_path)
if len(img.shape) == 3:
    img = np.mean(img, axis=2).astype(np.uint8)
imageio.imwrite(output_path,img)
    