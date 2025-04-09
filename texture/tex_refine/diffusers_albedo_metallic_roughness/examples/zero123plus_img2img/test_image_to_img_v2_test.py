from numpy import dtype
import torch
import requests
from PIL import Image
import sys
import os
import numpy as np
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane/src")
from diffusers.pipelines.zero123_diffusion.pipeline_zero123plus_img2img_v2_test import Zero123PlusImg2ImgPipeline
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
save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation_output_zero123plus_img2img_v2.0_base"
# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face_out"

# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti_output_zero123plus_v4.3"

seed = 99
torch.manual_seed(seed)
generator = torch.Generator(device=device).manual_seed(seed)

latents_in_grid = torch.load("latents_in_grid.pt")
class_embeddings = torch.load("class_embeddings.pt")
cond_encoded_clip = torch.load("cond_encoded_clip.pt")
encoder_hidden_states_prompt = torch.load("encoder_hidden_states_prompt.pt")

result = pipeline_img2img(latents_in_grid=latents_in_grid,
                            class_embeddings=class_embeddings,
                            cond_encoded_clip=cond_encoded_clip,
                            encoder_hidden_states_prompt=encoder_hidden_states_prompt, 
        num_inference_steps=75, guidance_scale=3.0, generator=generator).images[0]
result.save("test.png")
