from numpy import dtype
import torch
import os
import torch
import numpy as np
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler
from src.diffusers.pipeline.pipeline_zero123plus import Zero123PlusPipeline
from sam_preprocess.run_sam import process_image, process_image_path

# Load the pipeline
pipeline = Zero123PlusPipeline.from_pretrained(
    "/aigc_cfs_2/neoshang/code/diffusers_triplane/release/zero23plus_v10",
    # "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v4.4",
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
# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_quality"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_quality_output_zero123plus_v4.4_40000"

# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_face_out2"

# root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation"
# root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation_output_zero123plus_v10_15000"

root_image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti"
root_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_wuti_zero123plus_v10"

# seed = 99
# torch.manual_seed(seed)
# generator = torch.Generator(device=device).manual_seed(seed)


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
        try:
            # cond = Image.open(image_path)
            cond = process_image_path(image_path, bg_color=255, wh_ratio=0.8)
            result = pipeline(cond, num_inference_steps=75, guidance_scale=3.0).images[0]
            result.save(os.path.join(save_dir, filename))
        except KeyboardInterrupt:
            break
        except:
            continue

traverse_image_dir(root_image_dir, root_save_dir)