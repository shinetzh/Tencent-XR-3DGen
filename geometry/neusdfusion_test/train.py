import warnings

warnings.filterwarnings("ignore")

import os
import json
import time
import warnings

import torch
import torch.utils.data
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from models import *
from utils.reconstruct import *
from diff_utils.helpers import *
from dataloader.dataset_sdf_yh import DatasetSdfYh
from dataloader.dataset_sdfcolor import DatasetSdfColor
from dataloader.dataset_sdf_yh_coloronly import DatasetSdfYhColoronly
from dataloader.dataset_diffusion import DatasetDiffusion
from dataloader.dataset_diffusion_concat import DatasetDiffusionCat
from dataloader.dataset_dmtet import DatasetDmtet
from dataloader.dataset_diffusion_cloud_condition import DatasetCloudDiffusion
from dataloader.dataset_diffusion_unconditioned import DatasetDiffusionUnconditioned
from dataloader.dataset_diffusion_unconditioned_plane import (
    DatasetDiffusionUnconditionedPlane,
)
from dataloader.dataset_img_sdfcolor import DatasetImgSdfColor
from dataloader.dataset_sdfgeo import DatasetSdfGeo
from dataloader.dataset_diffusion_text_conditioned import (
    DatasetDiffusionTextConditioned,
)
from dataloader.dataset_pix3d_conditioned import DatasetDiffusionPix3D
from dataloader.dataset_diffusion_partial_conditioned import (
    DatasetDiffusionPartialConditioned,
)


def get_dataset(specs, resample=True):
    data_config = specs["data_config"]
    dataset_type = data_config["dataset_type"]

    if dataset_type == "diffusion_cloud_condition":
        return DatasetCloudDiffusion(data_config, data_type="train")
    elif dataset_type == "diffusion":
        return DatasetDiffusion(data_config, data_type="train")
    elif dataset_type == "diffusion_cond_cat":
        return DatasetDiffusionCat(data_config, data_type="train")
    elif dataset_type == "diffusion_uncond":
        return DatasetDiffusionUnconditioned(data_config, data_type="train")
    elif dataset_type == "diffusion_triplane":
        return DatasetDiffusionUnconditionedPlane(data_config, data_type="train")
    elif dataset_type == "sdf_sdfyh":
        return DatasetSdfYh(data_config, data_type="train")
    elif dataset_type == "dmtet":
        return DatasetDmtet(specs, data_type="train")
    elif dataset_type == "sdf_coloronly":
        return DatasetSdfYhColoronly(data_config, data_type="train")
    elif dataset_type == "sdf_sdfcolor":
        return DatasetSdfColor(data_config, data_type="train", resample=resample)
    elif dataset_type == "img_sdfcolor":
        return DatasetImgSdfColor(data_config, data_type="train", resample=resample)
    elif dataset_type == "sdf_sdfgeo":
        return DatasetSdfGeo(data_config, data_type="train")
    elif dataset_type == "diffusion_text_cond":
        return DatasetDiffusionTextConditioned(data_config, data_type="train")
    elif dataset_type == "diffusion_pix3d_cond":
        return DatasetDiffusionPix3D(data_config, data_type="train")
    elif dataset_type == "diffusion_partial_cond":
        return DatasetDiffusionPartialConditioned(data_config, data_type="train")
    else:
        print("dataset_type not recogonized: {}".format(dataset_type))
        exit(1)


def train(specs, args):
    train_dataset = get_dataset(specs, resample=True)

    if specs["training_task"] == ["vae_dmtet", "combined_dmtet_diffusion"]:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
            shuffle=True,
            pin_memory=False,
            persistent_workers=True,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    # creates a copy of current code / files in the config folder
    # save_code_to_conf(args.exp_dir)

    # pytorch lightning callbacks
    save_last = specs["save_last"] if "save_last" in specs.keys() else True
    svae_path = (
        os.path.join(specs["save_path"], args.exp_dir.split("/")[-2])
        if "save_path" in specs.keys()
        else args.exp_dir
    )
    print("save path: {}".format(svae_path))
    callback = ModelCheckpoint(
        dirpath=svae_path,
        filename="{epoch}",
        save_top_k=-1,
        save_last=save_last,
        every_n_epochs=specs["log_freq"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [callback, lr_monitor]
    model = CombinedModel(specs, args)

    if args.resume_params is not None and args.resume is None:
        ckpt = (
            "{}.ckpt".format(args.resume_params)
            if args.resume_params == "last"
            else "epoch={}.ckpt".format(args.resume_params)
        )
        resume_params_path = os.path.join(args.exp_dir, ckpt)
        model.load_state_dict(
            torch.load(resume_params_path, map_location="cpu")["state_dict"],
            strict=False,
        )
        resume = None
        print("load only state_dict from checkpoint: {}".format(resume_params_path))
    elif args.resume == "finetune":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if specs["training_task"] in ["vae_dmtet", "combined_dmtet_diffusion"]:
                model = model.load_from_checkpoint(
                    specs["modulation_ckpt_path"], specs=specs, args=args, strict=False
                )
            else:
                model = model.load_from_checkpoint(
                    specs["modulation_ckpt_path"], specs=specs, args=args, strict=False
                )
            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k, v in ckpt["state_dict"].items():
                new_key = k.replace(
                    "diffusion_model.", ""
                )  # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)
        resume = None
    elif args.resume is not None:
        ckpt = (
            "{}.ckpt".format(args.resume)
            if args.resume == "last"
            else "epoch={}.ckpt".format(args.resume)
        )
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None

    print(model)

    if args.multi_host:
        trainer = pl.Trainer(
            accelerator="gpu",
            num_nodes=5,
            devices=8,
            precision=specs["precision"],
            max_epochs=specs["num_epochs"],
            callbacks=callbacks,
            log_every_n_steps=1,
            default_root_dir=os.path.join("tensorboard_logs", args.exp_dir),
            strategy="ddp",
            accumulate_grad_batches=2,
        )
    else:
        log_every_n_steps = (
            specs["log_every_n_steps"] if "log_every_n_steps" in specs.keys() else 1
        )
        accumulate_grad_batches = (
            specs["accumulate_grad_batches"]
            if "accumulate_grad_batches" in specs.keys()
            else 2
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=-1,
            precision=32,
            max_epochs=specs["num_epochs"],
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            default_root_dir=os.path.join("tensorboard_logs", args.exp_dir),
            strategy=DDPStrategy(find_unused_parameters=False),
            accumulate_grad_batches=accumulate_grad_batches,
        )
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir",
        "-e",
        required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume",
        default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )
    arg_parser.add_argument(
        "--multi_host",
        action="store_true",
        help="if infer_latent, then, with out grad, and save all latent",
    )
    arg_parser.add_argument("--gen", type=bool, default=False)

    arg_parser.add_argument(
        "--resume_params",
        default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    arg_parser.add_argument("--batch_size", "-b", default=32, type=int)
    arg_parser.add_argument("--workers", "-w", default=1, type=int)

    arg_parser.add_argument(
        "--infer_latent",
        action="store_true",
        help="if infer_latent, then, with out grad, and save all latent",
    )

    args = arg_parser.parse_args()

    if args.gen:
        specs = json.load(open(os.path.join(args.exp_dir, "specs_gen.json")))
    else:
        specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    config_exp_dir = args.exp_dir.replace("store/", "")
    os.makedirs(config_exp_dir, exist_ok=True)
    json_path_list = glob(args.exp_dir + "/*.json")
    for json_path in json_path_list:
        json_name = os.path.basename(json_path)
        json_save2config_path = os.path.join(config_exp_dir, json_name)
        os.system("cp {} {}".format(json_path, json_save2config_path))
    print(specs["Description"])

    print(specs)

    torch.multiprocessing.set_start_method("spawn")
    print(
        "***********************************************************************************************"
    )
    print(
        "start time: {}, experiment: {}".format(
            time.strftime("%Y-%m-%d-%H:%M:%S"), args.exp_dir
        )
    )
    print(
        "***********************************************************************************************"
    )
    train(specs, args)
