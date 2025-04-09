import os
import torch
import numpy as np
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler
import sys
import json
import random

from torchvision.utils import make_grid, save_image

current_file_path = os.path.dirname(os.path.abspath(__file__))
two_levels_up = os.path.abspath(os.path.join(current_file_path, "../../"))
sys.path.append(two_levels_up)

from src.diffusers.pipelines.img2img.pipeline_img2img_albedo_metallic_roughness_bumpmap_sample import Img2ImgPipeline

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
                    # "/aigc_cfs_4/xibin/code/diffusers_triplane_models/img2img_albedo_6views_512_250k",
                    "/aigc_cfs_4/xibin/code/diffusers_triplane_models/img2img_albedo_metallic_roughness_bumpmap_6views",
                    torch_dtype=torch.float16,
                    local_files_only=True
                )

# Feel free to tune the scheduler
# img2img_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
#     img2img_pipeline.scheduler.config,
#     timestep_spacing='trailing'
# )

img2img_pipeline.scheduler = DDIMScheduler.from_config(
    img2img_pipeline.scheduler.config,
    timestep_spacing='trailing'
)

img2img_pipeline.to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

# Run the pipeline
root_save_dir = "/aigc_cfs_4/xibin/code/diffusers_triplane_models/6views_albedo_real_img_res_512_250k_41000_gs_pbr_all_sample"
os.makedirs(root_save_dir, exist_ok=True)
test_path = "/aigc_cfs/xibinsong/code/zero123plus_control/results/6views_90_15000_180rotate_75_1e_5_guidance_scale_5.5_conditioning_scale_2.0"

folder_list = os.listdir(test_path)
for folder in folder_list:
    # folder_each = os.path.join(test_path, folder)
    folder_each = os.path.join(test_path, folder)

    img_size = 512
    n_row = 3
    n_col = 2

    img_list = []
    img_name = os.path.join(folder_each, "res.png")
    image = Image.open(img_name)
    image_input = image
    image = image.resize((img_size*2, img_size*3))
    image = np.array(image)

    for m in range(n_row):
        for n in range(n_col):
            sub_img = image[m*img_size: (m+1)*img_size, n*img_size: (n+1)*img_size, :]
            sub_img = torch.from_numpy(sub_img)
            sub_img = sub_img.permute(2, 0, 1)
            sub_img = sub_img.unsqueeze(0)
            img_list.append(sub_img)

    images = torch.cat(img_list, dim=0)
    images = images.unsqueeze(0)
    
    # mv_imgs = img2img_pipeline(images, num_inference_steps=30, guidance_scale=0.5, width=img_size*2, height=img_size*3).images[0]
    mv_imgs = img2img_pipeline(images, num_inference_steps=30, guidance_scale=0.5, width=img_size*2, height=img_size*3)
    mv_imgs[0].images[0].save(os.path.join(root_save_dir, folder + "_albeo.png"))
    mv_imgs[1].images[0].save(os.path.join(root_save_dir, folder + "_metallic_roughness.png"))
    mv_imgs[2].images[0].save(os.path.join(root_save_dir, folder + "_bumpmap.png"))
    image_input.save(os.path.join(root_save_dir, folder + "_input.png"))