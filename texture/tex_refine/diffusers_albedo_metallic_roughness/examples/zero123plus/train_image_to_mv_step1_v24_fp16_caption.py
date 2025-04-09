#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import json
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from transformers.utils import ContextManagers
from PIL import Image, ImageColor, ImageDraw, ImageFont
import pathlib
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
# from sam_preprocess.run_sam import process_image, process_image_path

import sys
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")

sys.path.append("/aigc_cfs_2/neoshang/code/ModelZoo-PyTorch/PyTorch/built-in/diffusion/diffusers0.21.0")
sys.path.append("/aigc_cfs_2/neoshang/code/ModelZoo-PyTorch/PyTorch/built-in/diffusion/diffusers0.21.0/src")
from src.diffusers.pipelines.zero123_diffusion.pipeline_zero123plus_v3 import Zero123PlusPipeline, RefOnlyNoisedUNet, scale_image, scale_latents, unscale_image, unscale_latents
from src.diffusers.models.unet_2d_condition import UNet2DConditionModel

import diffusers
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from datasets_diffusion import get_dataset
from torchvision.utils import save_image
if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


# @torch.no_grad()
# def save_image(
#     tensor: Union[torch.Tensor, List[torch.Tensor]],
#     fp: Union[str, pathlib.Path, BinaryIO],
#     format: Optional[str] = None,
#     **kwargs,
# ) -> None:
#     """
#     Save a given Tensor into an image file.

#     Args:
#         tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
#             saves the tensor as a grid of images by calling ``make_grid``.
#         fp (string or file object): A filename or a file object
#         format(Optional):  If omitted, the format to use is determined from the filename extension.
#             If a file object was used instead of a filename, this parameter should always be used.
#         **kwargs: Other arguments are documented in ``make_grid``.
#     """

#     grid = make_grid(tensor, **kwargs)
#     # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
#     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
#     im = Image.fromarray(ndarr)
#     im.save(fp, format=format)



# def log_validation(vae, text_encoder, vision_encoder, tokenizer, unet_cond, unet, args, accelerator, weight_dtype, epoch, max_num=8):
#     logger.info("Running validation... ")

#     pipeline = Zero123PlusPipeline.from_pretrained(
#         args.pretrained_model_name_or_path,
#         vae=accelerator.unwrap_model(vae),
#         text_encoder=accelerator.unwrap_model(text_encoder),
#         tokenizer=tokenizer,
#         unet_cond=accelerator.unwrap_model(unet_cond).unet,
#         unet=accelerator.unwrap_model(unet).unet,
#         vision_encoder=accelerator.unwrap_model(vision_encoder),
#         safety_checker=None,
#         revision=args.revision,
#         torch_dtype=torch.float16,
#     )

#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     seed = 99
#     torch.manual_seed(seed)
#     generator = torch.Generator(device=accelerator.device).manual_seed(seed)

#     images = []
#     validation_image_path_list = []
#     for filename in os.listdir(args.validation_images_dir):
#         if not os.path.splitext(filename)[-1] in [".png", ".jpg", ".PNG", ".JPEG", ".JPG", ".jpeg"]:
#             continue
#         filepath = os.path.join(args.validation_images_dir, filename)
#         validation_image_path_list.append(filepath)

#     for i in range(len(validation_image_path_list)):
#         if len(images) >= max_num:
#             break
#         print(f"\rvalidate: {i}/{max_num}", end="", flush=True)

#         cond = process_image_path(validation_image_path_list[i], bg_color=255, wh_ratio=0.8)
#         result = pipeline(cond, num_inference_steps=75, 
#                           guidance_scale=3.0, width=512,
#                           height=1024, output_type="pt", 
#                           generator=generator)
#         # result = pipeline(cond, num_inference_steps=75, guidance_scale=3.0)

#         if result is None:
#             continue
#         image = result.images[0]

#         # image.save(f"test{i}.jpg")

#         images.append(image)

#     images_pred_all = torch.stack(images, dim=0)
#     nrow = 4
#     images_cond_grid = make_grid(images_pred_all, nrow=nrow, padding=1)
#     validation_save_dir = os.path.join(args.output_dir, "validation_output")
#     os.makedirs(validation_save_dir, exist_ok=True)
#     save_image(images_cond_grid, os.path.join(validation_save_dir, f"{epoch}-pred.jpg"))

#     del pipeline
#     torch.cuda.empty_cache()

#     return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/mnt/aigc_bucket_2/neoshang/code/diffusers",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="v_prediction",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--do_classifier_free_guidance",
        action="store_true",
        help="if use classifier free guidance"
    )

    parser.add_argument(
        "--validation_images_dir",
        type=str,
        default=None,
        help=(
            "validation images direction"),
    )
    parser.add_argument(
        "--drop_condition_prob",
        type=float,
        default=0.1,
        help=(
            "drop condition probility"),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


# class TrainModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()


# def register_params_for_train(train_module, params_dict):
#     for key, param in params_dict.items():
#         train_module.register_parameter(key.replace(".", "-"), param)


def make_validation_steps(checkpointing_steps, min_val_step=300, max_val_step=10000):
    validation_steps = []
    base_checkpointing_step = checkpointing_steps // 100
    for i in range(3):
        validation_steps.append(base_checkpointing_step)
        base_checkpointing_step = base_checkpointing_step * 10
    
    validation_steps = [max(min_val_step, min(x, max_val_step)) for x in validation_steps]
    print(f"validation_steps:{validation_steps}")
    return validation_steps

def get_validation_step(global_step, validation_steps):
    if global_step <= validation_steps[0] * 10:
        return validation_steps[0]
    elif global_step < validation_steps[0] * 10 + validation_steps[1] * 10:
        return validation_steps[1]
    else:
        return validation_steps[2]


def main():
    args = parse_args()

    config_path = os.path.join(args.output_dir, "train_configs.json")
    with open(config_path, 'r') as fr:
        configs = json.load(fr)
    configs["exp_dir"] = args.output_dir

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id


    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    
    
    
    # Load scheduler, tokenizer and models.
    val_noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_config(val_noise_scheduler.config)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vision_encoder", revision=args.revision
        )

    _unet_ = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, torch_dtype=torch.float16)
    _unet_.set_attention_slice(slice_size="auto")
    unet = RefOnlyNoisedUNet(unet=_unet_, train_sched=noise_scheduler, val_sched=val_noise_scheduler)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    vision_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.unet.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    params_require_grad = []
    params_require_nograd = []
    ### only training kv weight.
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if "transformer_blocks" in name:
            if "attn1" in name:
                param.requires_grad = True
                params_require_grad.append(name)
                param.data = param.data.float()
                continue
            elif "attn2.to_k.weight" in name or "attn2.to_v.weight" in name:
                param.requires_grad = True
                params_require_grad.append(name)
                param.data = param.data.float()
                continue
        ##### add ramping_coefficients for optimize
        elif "ramping_coefficients" in name:
            param.requires_grad = True
            params_require_grad.append(name)
            param.data = param.data.float()
            continue
        params_require_nograd.append(name)

    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset_train = get_dataset(configs, data_type="train", resample=False)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    print(f"args.num_train_epochs: {args.num_train_epochs}") ### 6
    print(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}") ### 61w
    print(f"args.lr_warmup_steps: {args.lr_warmup_steps}")  #### 1000
    print(f"accelerator.num_processes: {accelerator.num_processes}") ### 48
    print(f"args.max_train_steps: {args.max_train_steps}") ### 360w
    print(f"args.drop_condition_prob: {args.drop_condition_prob}")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vision_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # ### reduce memory
    # vae.enable_tiling()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_train.__len__()}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    validation_steps_list = make_validation_steps(args.checkpointing_steps)


    # if accelerator.is_main_process:
    #     log_validation(
    #         vae,
    #         text_encoder,
    #         vision_encoder,
    #         tokenizer,
    #         unet_cond,
    #         unet,
    #         args,
    #         accelerator,
    #         weight_dtype,
    #         global_step,
    #     )
    
    pre_param_grad = None
    pre_param_nograd = None
    for epoch in range(first_epoch, args.num_train_epochs):
        loss_list = []
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                cond_image_clip = batch["cond_image_clip"]
                cond_image_vae = batch["cond_images_vae"].to(weight_dtype)
                images_out = batch["images_out"]
                cond_caption = batch["cond_caption"]
                images_out = scale_image(images_out)
                b, vio, c, h, w = images_out.shape
                # ### get latents out
                # ncol = v // 2
                # nrow = v // ncol
                # images_out_grid = torch.zeros((b, c, h*ncol, w*nrow)).to(images_out.device, dtype=weight_dtype)
                # for i in range(b):
                #     images_out_grid[i] = make_grid(images_out[i], nrow=nrow, padding=0)

                # latents_out_grid = vae.encode(images_out_grid.to(weight_dtype)).latent_dist.sample()
                # latents_out_grid = latents_out_grid * vae.config.scaling_factor
                # latents_out_grid = scale_latents(latents_out_grid)

                ### get latents out
                if vio != 1:
                    latents_out = vae.encode(images_out.view(b*vio, c, h, w).to(weight_dtype)).latent_dist.sample()
                    _, cl, hl, wl = latents_out.shape
                    latents_out = latents_out.view(b, vio, cl, hl, wl)
                    ncol = vio // 2
                    nrow = vio // ncol
                    latents_out_grid = torch.zeros((b, cl, hl*ncol, wl*nrow)).to(latents_out.device, dtype=weight_dtype)
                    for i in range(b):
                        latents_out_grid[i] = make_grid(latents_out[i], nrow=nrow, padding=0)
                    # assert torch.sum(latents_out_grid[2, :, 40:, :40] - latents_out[2, 2, ...]) == 0
                else:
                    latents_out_grid = vae.encode(images_out.view(b*vio, c, h, w).to(weight_dtype)).latent_dist.sample()
                latents_out_grid = latents_out_grid * vae.config.scaling_factor
                latents_out_grid = scale_latents(latents_out_grid)

                if accelerator.is_main_process and (global_step == 0):
                    print(latents_out_grid.shape)

                if args.do_classifier_free_guidance:
                    drop_idx = np.where(np.random.rand(cond_image_vae.shape[0]) < args.drop_condition_prob)
                    cond_image_vae[drop_idx] = 0.0
                    for idx in drop_idx[0]:
                        cond_caption[int(idx)] = ""

                prompt_embedding = tokenize_captions(cond_caption).to(cond_image_vae.device)
                encoder_hidden_states_prompt = text_encoder(prompt_embedding)[0]

                ## get latents cond clip
                cond_encoded_clip = vision_encoder(cond_image_clip.to(weight_dtype), output_hidden_states=False) ### clip with projection
                cond_encoded_clip = cond_encoded_clip.image_embeds
                cond_encoded_clip = cond_encoded_clip.unsqueeze(-2)

                # get latents_cond_vae
                latents_cond_vae_list = []
                for i in range(b):
                    latents_cond_vae = vae.encode(cond_image_vae[i].to(weight_dtype)).latent_dist.sample()
                latents_cond_vae_list.append(latents_cond_vae)
                latents_cond_vae = torch.stack(latents_cond_vae_list, dim=0)
                cross_attention_kwargs = dict(cond_lat=latents_cond_vae)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents_out_grid)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents_out_grid.shape[0], latents_out_grid.shape[1], 1, 1), device=latents_out_grid.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents_out_grid.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_out_grid.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents_out_grid, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents_out_grid, noise, timesteps)


                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise.to(torch.float32)
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents_out_grid.to(torch.float32), noise.to(torch.float32), timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states_prompt=encoder_hidden_states_prompt,
                    cond_encoded_clip=cond_encoded_clip,
                    drop_idx=drop_idx,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample

                if args.snr_gamma <= 0:
                    if accelerator.is_main_process and (global_step == 0):
                        print("without snr")
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    if accelerator.is_main_process and (global_step == 0):
                        print("with snr")
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                    
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                # for name, param in unet.named_parameters():
                #     if "ramping_coefficients_weight" in name:
                #         print(name)
                #         if pre_param_grad is None:
                #             pre_param_grad = param.clone()
                #             break
                #         else:
                #             print(f"grad: {torch.sum(param - pre_param_grad)}")
                #         pre_param_grad = param.clone()
                #         break
                # for name, param in unet.named_parameters():
                #     if name == params_require_nograd[10]:
                #         if pre_param_nograd is None:
                #             pre_param_nograd = param.clone()
                #             break
                #         else:
                #             print(f"nograd: {torch.sum(param - pre_param_nograd)}")
                #         pre_param_nograd = param.clone()
                #         break
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # validation_step = get_validation_step(global_step, validation_steps_list)

                # if global_step % validation_step == 0:
                #     if accelerator.is_main_process:
                #         if args.use_ema:
                #             # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                #             ema_unet.store(unet.parameters())
                #             ema_unet.copy_to(unet.parameters())
                #         log_validation(
                #             vae,
                #             text_encoder,
                #             vision_encoder,
                #             tokenizer,
                #             unet,
                #             args,
                #             accelerator,
                #             weight_dtype,
                #             global_step,
                #         )
                #         if args.use_ema:
                #             # Switch back to the original UNet parameters.
                #             ema_unet.restore(unet.parameters())


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = Zero123PlusPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet).unet,
            vision_encoder=accelerator.unwrap_model(vision_encoder),
            safety_checker=None,
            revision=args.revision,
            torch_dtype=torch.float16,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()
