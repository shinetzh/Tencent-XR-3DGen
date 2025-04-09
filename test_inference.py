import os
import shutil
import time
import spaces
from glob import glob
from pathlib import Path

import gradio as gr
from gradio_litmodel3d import LitModel3D
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import subprocess

import argparse

from geometry.main_pipeline.diffusion.scripts.shape_inference import init_models_shape, inference_shape
from texture.tex_refine.texture_inference import init_models_texture, inference_texture

def _gen_shape(image_path, save_dir, weights_path):

    shape_models = init_models_shape(weights_path=weights_path)

    basename = os.path.basename(image_path)
    mesh_name = basename.split('.')[0] + ".obj"
    
    os.makedirs(save_dir, exist_ok=True)
    save_shape_dir = os.path.join(save_dir, "test_out_" + time.strftime('%Y-%m-%d-%H:%M:%S'))
    os.makedirs(save_dir, exist_ok=True)
    shape_dir = os.path.join(save_shape_dir, "shape_dir")
    os.makedirs(shape_dir, exist_ok=True)

    inference_shape(shape_models, image_path, shape_dir)

    mesh_path = os.path.join(shape_dir, mesh_name)
    ref_img_path = os.path.join(shape_dir, basename)

    return mesh_path, save_shape_dir, ref_img_path

def _gen_texture(
    mesh_path,
    save_path,
    ref_image_path,
    weights_path,
):
    texture_models = init_models_texture(weights_path=weights_path)

    output_path = os.path.join(save_path, "texture_dir")
    os.makedirs(output_path, exist_ok=True)

    job_id = 0
    inference_texture(texture_models, mesh_path, ref_image_path, output_path, job_id)

    basename = os.path.basename(ref_image_path).split(".")[0]
    mesh_path = os.path.join(output_path, f"{basename}.obj")
    return mesh_path

def main():
    parser = argparse.ArgumentParser(description='image based 3D Gen!')
    parser.add_argument('--image_path', type=str, default='./geometry/main_pipeline/diffusion/sample_images/typical_creature_robot_crab.png', help='input image path')
    parser.add_argument('--save_dir', type=str, default='./res_outputs', help='mesh save path')
    parser.add_argument('--weights_path', type=str, default='../Tencent-XR-3DGen', help='model path')
    args = parser.parse_args()

    image_dir = args.image_path
    save_dir = args.save_dir
    weights_path = args.weights_path
    os.makedirs(save_dir, exist_ok=True)

    mesh_path_shape, save_shape_dir, ref_img_path = _gen_shape(image_dir, save_dir, weights_path)
    print("generated shape path: ")
    print(mesh_path_shape)
    print(save_dir)
    print(ref_img_path)
    # breakpoint()

    mesh_path = _gen_texture(mesh_path_shape, save_shape_dir, ref_img_path, weights_path)
    print("generated shape and texture path: ")
    print(mesh_path)

if __name__ == "__main__":
    main()