import os
import sys
current_path = os.getcwd()
# 将当前路径添加到系统路径中
sys.path.append(current_path)

import time
from pathlib import Path
from dotmap import DotMap
from datetime import datetime
import torch
import torch.nn.functional as F
import requests
from PIL import Image, ImageEnhance 
import random
import numpy as np
import argparse
from pdb import set_trace as st
import nvdiffrast.torch as dr
import gc

# VAE & ControlNet
from diffusers import AutoencoderKL, ControlNetModel
# Pipeline
# from d2rgb_pipeline_6views import Zero123PlusPipeline
# from d2rgb_pipeline_6views_3views import Zero123PlusPipeline as Zero123PlusPipeline_3views
# from sdxl_pipeline import StableDiffusionXLControlNetImg2ImgPipeline
# Scheduler
from consistent_scheduler_eular_ancestral_discrete_newbaking_v2_2stage_90s import ConsistentEulerAncestralDiscreteScheduler

import math
from math import sin, cos
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

# Manifold constrain
from torch.autograd import grad

from pdb import set_trace as st
from matplotlib import pyplot as plt

# Baking
# import os, sys
# sys.path.append("/aigc_cfs_2/jiayuuyang/texture_refine/src/")
# from nv_diff_bake import NvdiffRender

# New Baking
import os, sys
# sys.path.append("/aigc_cfs_2/zacheng/demo_render/render_bake/")
# from render_bake_utils_v2_2stage_90s import dilate_masks, Renderer
from render_bake_utils_v5_pbr import dilate_masks, Renderer
import cv2, torch, numpy as np

# Captioning
# from gpt_caption import gpt_caption

def init_renderer(
    mesh=None, 
    render_size=512, 
    scale_factor=1.0, 
    renderer_extra_scale_up_factor=1.0, 
    d2rgb_special_scale=False, 
    texture_resolution=2048, 
    init_texture=None, 
    num_inference_steps=None,
    apply_consistent_interval=0,
    disable_consistency=False,
    rotate_input_mesh=False,
    ctx=None,
    unwrap_uv=False
):
        
    # New baking
    texture_resolution = texture_resolution
    obj_path = mesh
    obj_bound = scale_factor

    # New 6 views settings
    cam_azimuths = [0, 90, 180, 270, 0, 0] # len is n_views
    cam_elevations = [0,0,0,0, -89.9, 89.9] # len is n_views
    cam_distances = [5,5,5,5,5,5] # len is n_views
    camera_type = "ortho"

    # Original 4 views
    # cam_azimuths = [0, 90, 180, 270] # len is n_views
    # cam_elevations = [0,0,0,0] # len is n_views
    # cam_distances = [5,5,5,5] # len is n_views
    # camera_type = "ortho"

    bake_weight_exp = 3.0
    bake_front_view_weight = 10.0
    bake_erode_boundary = 5

    image_resolution = render_size

    # set up renderer
    renderer = Renderer(image_resolution, texture_resolution, world_orientation="y-up",ctx=ctx)
    renderer.set_object(obj_path, bound=obj_bound, orientation="y-up")
    if unwrap_uv:
        renderer.unwrap_uv()
    if rotate_input_mesh:
        deg = 30
        rad = math.radians(deg)
        transform = np.array([
            [cos(rad),  0,   sin(rad),  0],
            [0,         1,   0,         0],
            [-sin(rad), 0,   cos(rad),  0],
            [0,         0,          0,  1]
        ])
        renderer.transform_obj(transform)
    renderer.set_cameras(azimuths=cam_azimuths, elevations=cam_elevations, dists=cam_distances, camera_type=camera_type, zooms=1.0, near=1e-1, far=1e1)

    # render normal, depth, xyz
    depth, mask = renderer.render_depth("absolute", normalize=(255,50), bg=0) # (n_views, img_res, img_res, 1)
    normal, mask = renderer.render_normal("camera") # (n_views, img_res, img_res, 3)
    view_cos = -normal[...,-1:] # (n_views, img_res, img_res, 3)
    # xyz, _ = renderer.render_xyz(system="world", antialias=True, ssaa=True, cameras=None)

    # detect depth discontinuities, i.e. occlusion boundaries
    depth_map_uint8 = depth.cpu().numpy().astype(np.uint8) # (n_views, img_res, img_res, 1)
    depth_edge = [(cv2.Canny(d, 10, 40) > 0) for d in depth_map_uint8]
    depth_edge = dilate_masks(*depth_edge, iterations=bake_erode_boundary)
    depth_edge = (torch.from_numpy(depth_edge).cuda() > 0).float().unsqueeze(-1) # binary (n_views, img_res, img_res, 1)

    weights = view_cos * (1-depth_edge) # remove pixels on occlusion boundaries
    # apply weights
    weights = weights ** bake_weight_exp
    weights[0] *= bake_front_view_weight

    # self.renderer = renderer
    # self.depth = depth
    # self.normal = normal
    # self.xyz = xyz
    # self.mask = mask
    # self.weights = weights
    # self.renderer_extra_scale_up_factor = renderer_extra_scale_up_factor
    # self.d2rgb_special_scale = d2rgb_special_scale

    # self.apply_consistent_interval = apply_consistent_interval
    # self.disable_consistency = disable_consistency

    # Render rgb initial views if texture map is provided.
    if init_texture is not None:
        # Interpoate texture map to desired resolution.
        init_texture = torch.nn.functional.interpolate(init_texture, size=(texture_resolution,texture_resolution), mode='nearest', align_corners=False)

        # render
        color, alpha = self.renderer.sample_texture(init_texture, max_mip_level=4)
        # self.init_rgb_views = torch.clamp(torch.cat([color,alpha],dim=-1).permute(0,3,1,2),0,1)
        init_rgb_views = torch.clamp(torch.cat([color,alpha],dim=-1).permute(0,3,1,2),0,1)

    print("Renderer initialized.")

    return {"weights": weights, 
        "renderer": renderer, 
        "mask": mask, 
        "renderer_extra_scale_up_factor": renderer_extra_scale_up_factor, 
        "d2rgb_special_scale": d2rgb_special_scale, 
        "apply_consistent_interval": apply_consistent_interval}

def split_image_tensor(image, rows, cols):
    B, C, width, height = image.shape
    assert B == 1
    block_width = width // cols
    block_height = height // rows

    images = []
    for i in range(rows):
        for j in range(cols):
            left = j * block_width
            upper = i * block_height
            right = (j + 1) * block_width
            lower = (i + 1) * block_height
            sub_image = image[0, :,left:right,upper:lower]
            images.append(sub_image)

    return images

def bake_pbr_mr(
        in_images,
        renderer,
        weights,
        renderer_extra_scale_up_factor,
        save_intermediate_results=True,
):
    # in_images = torch.nn.functional.interpolate(in_images.float(), size=None, scale_factor=renderer_extra_scale_up_factor, mode='bilinear', align_corners=None, recompute_scale_factor=None, antialias=False)

    # image_list = split_image_tensor(in_images, 2, 3)
    # in_images = torch.stack([image_list[0],image_list[3],image_list[1],image_list[4],image_list[2],image_list[5]],dim=0)

    in_images = torch.stack([in_images[0],in_images[3],in_images[1],in_images[4],in_images[2],in_images[5]],dim=0)

    print(in_images.shape)
    print(weights.shape)

    # New baking
    # bake
    # image_weights = torch.cat((in_images.permute(0,2,3,1), weights), dim=-1)
    image_weights = torch.cat((in_images, weights), dim=-1)
    # texture_weights = renderer.bake_textures_raycast(image_weights, interpolation="nearest", inpaint=False)
    # texture_weights = renderer.bake_textures(image_weights, inpaint=False, max_mpimap_level=None)
    texture_weights = renderer.bake_textures(image_weights)
    textures, weights = torch.split(texture_weights, (3,1), dim=-1)
    # blend textures by weights
    total_weights = torch.sum(weights, dim=0, keepdim=True) # (1, img_res, img_res, 1)
    texture = torch.sum(textures*weights, dim=0, keepdim=True) / (total_weights + 1e-10) # (1, img_res, img_res, 3)
    # inpaint missing regions, optional
    texture_mr = renderer.inpaint_textures(texture, (total_weights<=1e-5), inpaint_method="laplace") # (1, img_res, img_res, 3)

    ### Render ###
    # re-render image from textures
    color, alpha = renderer.sample_texture(texture_mr, max_mip_level=4)

    metallic_roughness_z_rendered = torch.clamp(torch.cat([color,alpha],dim=-1).permute(0,3,1,2),0,1) # range [0,1]

    # Encode rendered views into latents and use as new pred_x0
    render_size = metallic_roughness_z_rendered.shape[-1]
    if metallic_roughness_z_rendered.shape[0] == 4: # 4 views
        metallic_roughness_z_rendered_grid = torch.zeros((1,3,render_size*2,render_size*2),device="cuda",dtype=torch.float32)

        metallic_roughness_z_rendered_grid[:,:,:render_size,:render_size] = metallic_roughness_z_rendered[0][:3]*metallic_roughness_z_rendered[0][-1:] + ((1-metallic_roughness_z_rendered[0][-1:])*0.0)# Black background
        metallic_roughness_z_rendered_grid[:,:,:render_size,render_size:] = metallic_roughness_z_rendered[1][:3]*metallic_roughness_z_rendered[1][-1:] + ((1-metallic_roughness_z_rendered[1][-1:])*0.0)
        metallic_roughness_z_rendered_grid[:,:,render_size:,:render_size] = metallic_roughness_z_rendered[2][:3]*metallic_roughness_z_rendered[2][-1:] + ((1-metallic_roughness_z_rendered[2][-1:])*0.0)
        metallic_roughness_z_rendered_grid[:,:,render_size:,render_size:] = metallic_roughness_z_rendered[3][:3]*metallic_roughness_z_rendered[3][-1:] + ((1-metallic_roughness_z_rendered[3][-1:])*0.0)

    elif metallic_roughness_z_rendered.shape[0] == 6: # 6 views
        metallic_roughness_z_rendered_grid = torch.zeros((1,3,render_size*3,render_size*2),device="cuda",dtype=torch.float32)

        metallic_roughness_z_rendered_grid[:,:,:render_size,:render_size] = metallic_roughness_z_rendered[0][:3]*metallic_roughness_z_rendered[0][-1:] + ((1-metallic_roughness_z_rendered[0][-1:])*0.0)# Black background
        metallic_roughness_z_rendered_grid[:,:,:render_size,render_size:] = metallic_roughness_z_rendered[1][:3]*metallic_roughness_z_rendered[1][-1:] + ((1-metallic_roughness_z_rendered[1][-1:])*0.0)
        metallic_roughness_z_rendered_grid[:,:,render_size:2*render_size,:render_size] = metallic_roughness_z_rendered[2][:3]*metallic_roughness_z_rendered[2][-1:] + ((1-metallic_roughness_z_rendered[2][-1:])*0.0)
        metallic_roughness_z_rendered_grid[:,:,render_size:2*render_size,render_size:] = metallic_roughness_z_rendered[3][:3]*metallic_roughness_z_rendered[3][-1:] + ((1-metallic_roughness_z_rendered[3][-1:])*0.0)
        metallic_roughness_z_rendered_grid[:,:,2*render_size:,:render_size] = metallic_roughness_z_rendered[4][:3]*metallic_roughness_z_rendered[4][-1:] + ((1-metallic_roughness_z_rendered[4][-1:])*0.0)
        metallic_roughness_z_rendered_grid[:,:,2*render_size:,render_size:] = metallic_roughness_z_rendered[5][:3]*metallic_roughness_z_rendered[5][-1:] + ((1-metallic_roughness_z_rendered[5][-1:])*0.0)

    print(metallic_roughness_z_rendered_grid.shape)
    breakpoint()

    if save_intermediate_results:
        plt.imsave(os.path.join("./",f"metallic_roughness_rendered.png"),(torch.clamp(metallic_roughness_z_rendered_grid[0].to(torch.float32),0,1).permute(1,2,0).contiguous().cpu().numpy()))
    # plt.imsave("debug.png",(torch.clamp(new_pred_x0_rgb[0].to(torch.float32),0,1).permute(1,2,0).contiguous().cpu().numpy()))
    # plt.imsave("debug2.png",textured_views_rgb[2][:3].permute(1,2,0).float().data.cpu().numpy())

    # Downsample back to original size of diffusion model
    # metallic_roughness_z_rendered_grid = torch.nn.functional.interpolate(metallic_roughness_z_rendered_grid, size=None, scale_factor=1/self.renderer_extra_scale_up_factor, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)

    return texture_mr, metallic_roughness_z_rendered_grid

in_mesh_path = "/aigc_cfs_10/Asset/objaverse/meshes/new_art/conflatio/standard_part4/mesh/pod_22/objaverse/00a271e787014ba9bd3106502d9c8c31/texture/texture.obj"
metallic_roughness_path = "/apdcephfs_cq10/share_1615605/Assets/objaverse/render/material/new_pbr/pbr_20241208/debug/render_data/objaverse/00a271e787014ba9bd3106502d9c8c31/render_512_Valour/PBR/color"
color_path = "/apdcephfs_cq10/share_1615605/Assets/objaverse/render/material/new_pbr/pbr_20241208/debug/render_data/objaverse/00a271e787014ba9bd3106502d9c8c31/render_512_Valour/color"

# Initialize global_ctx
global_ctx = dr.RasterizeCudaContext()

# scheduler = ConsistentEulerAncestralDiscreteScheduler.from_config()

d2rgb_per_view_image_size = 512
render_resolution = 512
scale_factor = 0.9
renderer_extra_scale_up_factor = 1
d2rgb_texture_map_resolution = 1024

params = init_renderer(mesh=in_mesh_path, 
    render_size=render_resolution, 
    scale_factor=scale_factor, 
    renderer_extra_scale_up_factor=renderer_extra_scale_up_factor,
    d2rgb_special_scale=True,
    texture_resolution=d2rgb_texture_map_resolution,
    ctx=global_ctx,
    unwrap_uv=True)

# breakpoint()


img_name_list = os.listdir(metallic_roughness_path)
img_name_list.sort()

img_list = []
for name in img_name_list:
    full_name = os.path.join(metallic_roughness_path, name)
    img = Image.open(full_name)
    img = torch.from_numpy(np.array(img)).to("cuda")
    img = img[:,:,:3]
    # breakpoint()
    # img = img.permute(2, 0, 1)
    print(img.shape)
    # img = img[:3, :, :]
    img_list.append(img)


res1, res2 = bake_pbr_mr(
        img_list,
        params["renderer"],
        params["weights"],
        params["renderer_extra_scale_up_factor"],
        save_intermediate_results=True,)

# metallic_roughness_z_grid = torch.tensor(np.array(metallic_roughness_imgs)).to('cuda').permute(2,0,1).unsqueeze(0).float()/255.0

