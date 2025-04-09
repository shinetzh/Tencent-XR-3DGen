#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import trimesh
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils import mesh, evaluate
from utils.reconstruct import *
from diff_utils.helpers import * 
#from metrics.evaluation_metrics import *#compute_all_metrics
#from metrics import evaluation_metrics

from dataloader.pc_loader import PCloader
from dataloader.sdf_nogt_loader import SdfNogtLoader
from dataloader.vae_data_loader import TriplaneDataLoader

@torch.no_grad()
def test_vae():
    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    for i in tqdm(range(args.num_samples)):
        with torch.no_grad():
          triplane = model.vae_model.sample(1)[0]
        # save modulation vectors for training diffusion model for next stage
        # filter based on the chamfer distance so that all training data for diffusion model is clean 
        # would recommend visualizing some reconstructed meshes and manually determining what chamfer distance threshold to use

        torch.save(triplane.cpu(), os.path.join(latent_dir, "{}.pt".format(str(i).zfill(5))))


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--num_samples", "-n", default=5, type=int, help='number of samples to generate and reconstruct')

    arg_parser.add_argument("--filter", default=False, help='whether to filter when sampling conditionally')

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])

    
    latent_dir = os.path.join(args.exp_dir, "modulations")
    os.makedirs(latent_dir, exist_ok=True)
    test_vae()


  
