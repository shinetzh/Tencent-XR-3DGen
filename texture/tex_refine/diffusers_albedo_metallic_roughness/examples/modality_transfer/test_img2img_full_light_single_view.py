import os
import torch
import numpy as np
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler
import sys
import json

from torchvision.utils import make_grid, save_image

current_file_path = os.path.dirname(os.path.abspath(__file__))
two_levels_up = os.path.abspath(os.path.join(current_file_path, "../../"))
sys.path.append(two_levels_up)

from src.diffusers.pipelines.img2img.pipeline_img2img_single import Img2ImgPipeline
# from sam_preprocess.run_sam import process_image, process_image_path

def to_rgb_image(maybe_rgba: Image.Image, bg_color=127, edge_aug_threshold=0, bright_scale=None):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        # img = np.random.randint(random_grey_low, random_grey_high, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = np.ones([rgba.size[1], rgba.size[0], 3], dtype=np.uint8) * bg_color
        img = Image.fromarray(img, 'RGB')

        #### bright adapt
        if bright_scale is not None:
            rgba_array = np.array(rgba)
            rgb = cv2.convertScaleAbs(rgba_array[..., :3], alpha=bright_scale, beta=0)
            rgb = Image.fromarray(rgb)
            img.paste(rgb, mask=rgba.getchannel('A'))
        else:
            img.paste(rgba, mask=rgba.getchannel('A'))

        #### edge augmentation
        if edge_aug_threshold > 0 and (random.random() < edge_aug_threshold):
            mask_img = np.array(rgba.getchannel('A'))
            mask_img[mask_img > 0] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            iterration_num = random.randint(1, 2)
            mask_img_small = cv2.erode(mask_img, kernel, iterations=iterration_num)
            mask_img_edge = mask_img - mask_img_small
            mask_img_edge = np.concatenate([mask_img_edge[..., None]]*3, axis=-1) / 255.0
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img_array = np.array(img) * (1 - mask_img_edge) + rand_color * mask_img_edge
            img = Image.fromarray(img_array.astype(np.uint8))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

device = torch.device("cuda:0")

img2img_pipeline = Img2ImgPipeline.from_pretrained(
                    "/aigc_cfs_4/xibin/code/diffusers_triplane_models/delight_models_full_light",
                    torch_dtype=torch.float16,
                    local_files_only=True
                )

# Feel free to tune the scheduler
img2img_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    img2img_pipeline.scheduler.config,
    timestep_spacing='trailing'
)
img2img_pipeline.to(device)


# Run the pipeline
root_save_dir = "/aigc_cfs_4/xibin/code/diffusers_triplane_models/test_images_delight_res_50000"
os.makedirs(root_save_dir, exist_ok=True)

input_path = "/aigc_cfs_4/xibin/code/diffusers_triplane_models/test_images_delight"
img_name_list = os.listdir(input_path)

for name in img_name_list:
    img_name = os.path.join(input_path, name)

    if True:
        # for idx in cond_list:
        if True:
            image = Image.open(img_name)
            image.save(os.path.join(root_save_dir, name+"_input.png"))
            image = image.resize((512, 512))

            image = np.array(image)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
            # image = image.unsqueeze(0)
            print(image.shape)
            mv_imgs = img2img_pipeline(image, num_inference_steps=30, guidance_scale=0.5, width=512, height=512).images[0]
            mv_imgs.save(os.path.join(root_save_dir, name + "_res.png"))

