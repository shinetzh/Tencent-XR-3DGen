from numpy import dtype
import torch
import requests
from PIL import Image
import sys
import os

sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane/src")
from diffusers.pipelines.zero123_diffusion.pipeline_zero123plus import Zero123PlusPipeline
from diffusers.pipelines.zero123_diffusion.pipeline_zero123plus_img2img import Zero123PlusImg2ImgPipeline
from diffusers import EulerAncestralDiscreteScheduler

from sam_preprocess.run_sam import process_image, process_image_path
import torchvision.utils as vutils


# Load the pipeline
pipeline_img2img = Zero123PlusImg2ImgPipeline.from_pretrained(
    "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus_img2img/v1.0",
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

# Load the pipeline
pipeline = Zero123PlusPipeline.from_pretrained(
    "/aigc_cfs_2/neoshang/code/diffusers_triplane/release/zero23plus_v4.4",
    # "/aigc_cfs_2/neoshang/models/zero123plus-v1.2", 
    # "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v1",
    torch_dtype=torch.float32, 
    local_files_only=True
)
# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config,
    timestep_spacing='trailing'
)

device = torch.device("cuda:0") # "cuda:0" | "npu:0"
pipeline.to(device)

# Run the pipeline
root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation"
root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation_output_zero123plus_img2img_v1.0"
# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face_out"

# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti_output_zero123plus_v4.3"

seed = 99
torch.manual_seed(seed)
generator = torch.Generator(device=device).manual_seed(seed)

def traverse_image_dir(image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        if os.path.isdir(image_path):
            sub_save_dir = os.path.join(save_dir, filename)
            traverse_image_dir(image_path, sub_save_dir)
            continue
        if not os.path.splitext(filename)[-1] in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            continue
        print(image_path)
        # try:
        # cond = Image.open(image_path)
        cond = process_image_path(image_path)
        print("zero123++ running...")
        image1 =  pipeline(cond, num_inference_steps=75, guidance_scale=3.0, generator=generator, output_type="pt").images[0]
        vutils.save_image(image1, "test.png")
        breakpoint()
        print("zero123++ image to image running...")
        result = pipeline_img2img(cond, image1, num_inference_steps=75, guidance_scale=3.0, generator=generator).images[0]
        result.save(os.path.join(save_dir, filename))
        # except KeyboardInterrupt:
        #     break
        # except:
        #     continue

traverse_image_dir(root_image_dir, root_save_dir)