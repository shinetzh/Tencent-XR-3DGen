import time
import numpy as np
import json
import os
import sys
import torch
import open_clip
from PIL import Image
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, AutoModel
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler, DDIMScheduler

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.vae.vae import get_vae_model
from models.vae.extract_geometry import extract_geometry
from utils.utils_obj import export_obj
from utils.rmbg1_4 import process_image_path, process_image_path_list
from models.diffusion.transformer_vector import SD3Transformer2DModel
from pipelines.pipeline_mmdit_flow import MMDitFlowPipeline

import subprocess

device = torch.device('cuda:0')

# exp_dir = "/root/autodl-tmp/xibin/checkpoint/geometry_dit"
exp_dir = "/root/autodl-tmp/xibin/code/Tencent-XR-3DGen/geometry_dit"
# image_dir = "./sample_images/"
image_dir = "./sample_images/typical_creature_robot_crab.png"
# image_dir = "/root/autodl-tmp/xibin/code/PandoraX/geometry/main_pipeline/diffusion/sample_images/typical_creature_robot_crab.png"
    
save_dir = "../../../outputs/"
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.join(save_dir, "test_out_" + time.strftime('%Y-%m-%d-%H:%M:%S'))
os.makedirs(save_dir, exist_ok=True)
shape_dir = os.path.join(save_dir, "shape_dir")
os.makedirs(shape_dir, exist_ok=True)

# ## test 1 view cond diffusion
cmd = f"python scripts/test_mmdit_image23D_flow_1view_gradio.py \
    --exp_dir '{exp_dir}' \
    --save_dir '{shape_dir}' \
    --image_dir '{image_dir}'"
subprocess.run(cmd, shell=True)

basename = os.path.basename(image_dir).split(".")[0]
mesh_path = os.path.join(shape_dir, f"{basename}.obj")
ref_img_path = os.path.join(shape_dir, f"{basename}.png")

print(ref_img_path)
print(mesh_path)
print(basename)
# breakpoint()

# Texture Generation
cmd = f"python ../../../texture/tex_refine/inference_consistent_d2rgb_6views_sdxl_sr_v5_pbr_gradio.py \
--obj_path '{mesh_path}' \
--ref_img_path '{ref_img_path}' \
--output_path '{save_dir}'"

subprocess.run(cmd, shell=True)
