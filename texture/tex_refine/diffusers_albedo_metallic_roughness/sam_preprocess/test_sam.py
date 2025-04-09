import os
import time
from PIL import Image
import numpy as np
import shutil
import cv2
from utils import remove_backgroud, remove_backgroud_whitebg

from run_sam import process_image, process_image_path

# image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation/test10.png"
# image_sam = process_image_path(image_path)
# image_sam.save("sam_test_result.png")


# npy_path = "/aigc_cfs_gdp/neoshang/data/test_results/selected_highres_may_zero123plus_v4.7/npy/2012.npy"
# test_save_dir = "test_out"
# os.makedirs(test_save_dir, exist_ok=True)

# image_array = np.load(npy_path)

# for i, image in enumerate(image_array):
#     image = np.transpose(image, (1,2,0))
#     print(image.shape)
#     image_rgb = Image.fromarray(image.astype(np.uint8)).convert("RGB")
#     image_rgb.save(os.path.join(test_save_dir, str(i)+"_origin.jpg"))

# start_time = time.time()
# output = remove_backgroud_whitebg(image_array)
# end_time = time.time()
# print(end_time - start_time)

# for i, image in enumerate(output):
#     image = np.transpose(image, (1,2,0))
#     print(image.shape)
#     image_rgb = Image.fromarray(image.astype(np.uint8)).convert("RGB")
#     image_rgb.save(os.path.join(test_save_dir, str(i)+".jpg"))
# # breakpoint()


# image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation/test10.png"
# image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation/mario.png"
# image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation/helmet.png"
image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/penguins_mv/b1.png"
# img_dir = "/aigc_cfs_2/neoshang/code/tdmq_everything/data/data/test_quality"
save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/penguins_mv"
os.makedirs(save_dir, exist_ok=True)
# for filename in os.listdir(img_dir):
    # image_path = os.path.join(img_dir, filename)
result, result_mask = process_image_path(image_path, bg_color=255, wh_ratio=0.9, save_samout=False, use_sam=False)
print(result.size)
# save_path = os.path.join(save_dir, os.path.basename(image_path))
result.save("rgb.png")
result_mask.save("mask.png")