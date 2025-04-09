from numpy import dtype
import torch
import requests
from PIL import Image
import sys
import os
import numpy as np
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane/src")
from diffusers.pipelines.zero123_diffusion.pipeline_zero123plus_img2img_v2 import Zero123PlusImg2ImgPipeline
from diffusers import EulerAncestralDiscreteScheduler

from sam_preprocess.run_sam import process_image, process_image_path
import torchvision.utils as vutils


# Load the pipeline
pipeline_img2img = Zero123PlusImg2ImgPipeline.from_pretrained(
    "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus_img2img/v2.0",
    # "/aigc_cfs_2/neoshang/models/zero123plus-v1.2", 
    # "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v1",
    torch_dtype=torch.float32, 
    local_files_only=True
)
# Feel free to tune the scheduler
pipeline_img2img.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline_img2img.scheduler.config,
    timestep_spacing='trailing'
)

device = torch.device("cuda:0") # "cuda:0" | "npu:0"
pipeline_img2img.to(device)


# Run the pipeline
condition_img_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation"
shading_concat_img_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation_output_zero123_v4.5"
save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation_output_zero123plus_img2img_v2.0_10000"
# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face_out"

# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti_output_zero123plus_v4.3"

seed = 99
torch.manual_seed(seed)
generator = torch.Generator(device=device).manual_seed(seed)

def traverse_image_dir(condition_img_dir, shading_concat_img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(condition_img_dir):
        cond_image_path = os.path.join(condition_img_dir, filename)
        shading_img_cat_path = os.path.join(shading_concat_img_dir, filename)
        if os.path.isdir(cond_image_path):
            sub_save_dir = os.path.join(save_dir, filename)
            traverse_image_dir(cond_image_path, shading_img_cat_path, sub_save_dir)
            continue
        if not os.path.splitext(filename)[-1] in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            continue
        print(cond_image_path)
        try:
            cond = process_image_path(cond_image_path)
            shading_image_cat = np.array(Image.open(shading_img_cat_path))
            images_in_list = [Image.fromarray(shading_image_cat[:256, :256, :]), Image.fromarray(shading_image_cat[:256, 256:, :]),
                            Image.fromarray(shading_image_cat[256:512, :256, :]), Image.fromarray(shading_image_cat[256:512, 256:, :]),
                            Image.fromarray(shading_image_cat[512:768, :256, :]), Image.fromarray(shading_image_cat[512:768, 256:, :]),
                            Image.fromarray(shading_image_cat[768:, :256, :]), Image.fromarray(shading_image_cat[768:, 256:, :])]
            result = pipeline_img2img(cond, images_in_list, num_inference_steps=75, guidance_scale=3.0, generator=generator).images[0]
            result.save(os.path.join(save_dir, filename))
        except KeyboardInterrupt:
            break
        except:
            continue

traverse_image_dir(condition_img_dir, shading_concat_img_dir, save_dir)