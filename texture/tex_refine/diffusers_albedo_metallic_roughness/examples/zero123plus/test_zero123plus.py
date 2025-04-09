import time
import numpy as np
import json
import os
import sys
import torch
import random
# import torch_npu
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane/src")
from datasets_diffusion import get_dataset
# from diffusers.pipelines.zero123_diffusion.pipeline_zero123plus_test import Zero123PlusPipeline, RefOnlyNoisedUNet, scale_image, scale_latents, unscale_image, unscale_latents
from diffusers.pipelines.zero123_diffusion.pipeline_zero123plus import Zero123PlusPipeline, RefOnlyNoisedUNet, scale_image, scale_latents, unscale_image, unscale_latents

from diffusers import EulerAncestralDiscreteScheduler

device = torch.device('cuda:0')

def test(args):
    exp_dir = args.exp_dir
    num_samples = args.num_samples
    batch_size = args.batch_size
    if not args.save_dir:
        save_test_dir = os.path.join(exp_dir, "test_out_" + time.strftime('%Y-%m-%d-%H:%M:%S'))
    else:
        save_test_dir = os.path.join(args.save_dir, "test_out_" + time.strftime('%Y-%m-%d-%H:%M:%S'))
    print(f"save_test_dir: {save_test_dir}")
    os.makedirs(save_test_dir, exist_ok=True)

    configs_path = os.path.join(exp_dir, "train_configs.json")
    with open(configs_path, 'r') as fr:
        configs = json.load(fr)
    configs["exp_dir"] = exp_dir

    test_dataset = get_dataset(configs, data_type="test", resample=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    # "/aigc_cfs_2/neoshang/models/zero123plus-v1.2", 
    pipeline = Zero123PlusPipeline.from_pretrained(
        exp_dir, 
        torch_dtype=torch.float32,
        local_files_only=True
    )
    # Feel free to tune the scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing='trailing'
    )
    pipeline.to('cuda:0')

    num = 0
    for i, batch in enumerate(test_dataloader):
        # cond_image = batch["cond_image"]
        # cond_image_clip = batch["cond_image_clip"]
        # cond_image_vae = batch["cond_image_vae"]
        # images_out = batch["images_out"]
        # result = pipeline(cond_image_vae, cond_image_clip, num_inference_steps=75, guidance_scale=4.0).images

        # for c_image, gt_image, pred_image in zip(cond_image, images_out, result):
        #     save_sub_dir = os.path.join(save_test_dir, str(num).zfill(4))
        #     os.makedirs(save_sub_dir, exist_ok=True)
        #     c_image = Image.fromarray(c_image.numpy().astype('uint8')).convert('RGB')
        #     c_image.save(os.path.join(save_sub_dir, "cond_image.jpg"))
        #     image_gt = make_grid(gt_image, nrow=2) * 0.5 + 0.5
        #     save_image(image_gt, os.path.join(save_sub_dir, "gt_image.jpg"))
        #     pred_image.save(os.path.join(save_sub_dir, "pred_image.jpg"))
        #     num += 1
        #     print(f"generated {num}")
        
        cond_image_path = batch["cond_image_path"]
        for image_path in cond_image_path:
            save_sub_dir = os.path.join(save_test_dir, str(num).zfill(4))
            os.makedirs(save_sub_dir, exist_ok=True)
            cond = Image.open(image_path)
            result = pipeline(cond, num_inference_steps=75).images[0]
            result.save(os.path.join(save_sub_dir, "pred_image.jpg"))
            num += 1



if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'train_configs.json' and 'test_configs.json'",
    )
    arg_parser.add_argument(
        "--save_dir", default=None,
        help="specify the save_dir",
    )
    arg_parser.add_argument("--num_samples", "-n", default=999999, type=int, help='number of samples to generate and reconstruct')
    arg_parser.add_argument("--batch_size", "-b", default=2, type=int, help='number of samples to generate and reconstruct')
    arg_parser.add_argument("--num_images_per_prompt", default=1, type=int, help='number of samples to generate per prompt')
    arg_parser.add_argument("--test_prob", default=1.0, type=float, help='the probility of test a batch')
    

    args = arg_parser.parse_args()

    test(args)