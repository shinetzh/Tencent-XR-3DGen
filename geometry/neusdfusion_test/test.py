#!/usr/bin/env python3

import torch
import imageio
import torch.utils.data
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from easydict import EasyDict as edict
import nvdiffrast.torch as dr
from trisdf.model.utils_tex import plot_texmesh, plot_texmesh_split, plot_geomesh_split
import json
from triplane_nvdiffrec.render import obj
import os
from glob import glob
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
from dataloader.dataset_sdf_yh import DatasetSdfYh
from dataloader.dataset_sdfcolor import DatasetSdfColor
from dataloader.dataset_sdf_yh_coloronly import DatasetSdfYhColoronly
from dataloader.vae_data_loader import TriplaneDataLoader
from dataloader.dataset_diffusion import DatasetDiffusion
from dataloader.dataset_diffusion_concat import DatasetDiffusionCat
from dataloader.dataset_dmtet import DatasetDmtet
from dataloader.dataset_diffusion_cloud_condition import DatasetCloudDiffusion
from dataloader.dataset_diffusion_unconditioned import DatasetDiffusionUnconditioned
from dataloader.dataset_img_sdfcolor import DatasetImgSdfColor
from dataloader.dataset_sdfgeo import DatasetSdfGeo
from dataloader.dataset_diffusion_text_conditioned import DatasetDiffusionTextConditioned
from dataloader.dataset_pix3d_conditioned import DatasetDiffusionPix3D
from dataloader.dataset_diffusion_partial_conditioned import DatasetDiffusionPartialConditioned

from models.utils_model import get_decoder_model, get_diffusion_model, get_vae_model
from models.loss import L2Loss

from dataloader.triplane_stats import normalize, unnormalize


def get_dataset(specs, data_type="train", resample=False):
    data_config = specs["data_config"]
    dataset_type = data_config["dataset_type"]

    if dataset_type == "diffusion_cloud_condition":
        return DatasetCloudDiffusion(data_config, data_type=data_type, resample=resample)
    elif dataset_type == "diffusion":
        return DatasetDiffusion(data_config, data_type=data_type, resample=resample)
    elif dataset_type == "diffusion_cond_cat":
        return DatasetDiffusionCat(data_config, data_type=data_type, resample=resample)
    elif dataset_type == "diffusion_uncond":
        return DatasetDiffusionUnconditioned(data_config, data_type="train")
    elif dataset_type == "sdf_sdfyh":
        return DatasetSdfYh(data_config, data_type=data_type, resample=resample)
    elif dataset_type == "dmtet":
        return DatasetDmtet(specs, data_type=data_type, resample=resample)
    elif dataset_type == "sdf_coloronly":
        return DatasetSdfYhColoronly(data_config, data_type=data_type, resample=resample)
    elif dataset_type == "sdf_sdfcolor":
        return DatasetSdfColor(data_config, data_type=data_type, resample=resample)
    elif dataset_type == "sdf_sdfgeo":
        return DatasetSdfGeo(data_config, data_type=data_type, resample=resample)
    else:
        print("dataset_type not recogonized: {}".format(dataset_type))
        raise NotImplementedError


@torch.no_grad()
def test_vae():
    
    # load dataset, dataloader, model checkpoint
    if specs["training_task"] == "vae":
        test_dataset = TriplaneDataLoader(specs["data_config"], return_filename=True, sort=True)
    else:
        test_split = json.load(open(specs["TestSplit"]))
        test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
            if specs["training_task"] == "vae":
                triplane, filename = data
            else:
                point_cloud, filename = data # filename = path to the csv file of sdf data
            filename = filename[0] # filename is a tuple

            cls_name = filename.split("/")[-3]
            mesh_name = filename.split("/")[-1].split('.')[0]
            
            triplane = triplane.cuda()
            out = model.vae_model(triplane)
            reconstructed_plane_feature, latent = out[0], out[-1]
            print(torch.mean(torch.abs(reconstructed_plane_feature - triplane)))
            # save modulation vectors for training diffusion model for next stage
            # filter based on the chamfer distance so that all training data for diffusion model is clean 
            # would recommend visualizing some reconstructed meshes and manually determining what chamfer distance threshold to use

            outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            print(latent.shape)
            np.save(os.path.join(outdir, "latent.npy"), latent.cpu().numpy())
            torch.save(reconstructed_plane_feature.cpu(), os.path.join(outdir, "plane_feature.pt"))

@torch.no_grad()
def test_vae_sdfcolor():
    # load dataset, dataloader, model checkpoint
    stats_dir = None
    if os.path.exists(os.path.join(args.exp_dir, "stats")):
        stats_dir = os.path.join(args.exp_dir, "stats")
        min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
        max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
        _range = max_values - min_values
        middle = (min_values + max_values) / 2
    test_dataset = DatasetSdfColor(specs["data_config"], data_type=specs["data_config"]["test_type"], resample=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()
    else:
        model = CombinedModel(specs=specs, args=args).cuda().eval()
    

    # json_path = '/apdcephfs_cq3/share_1615605/neoshang/code/rendering_free_onetri/savedir/test_128_v13/alldata_1113_right.json'
    # with open(json_path, 'r') as fr:
    #     dataset_dict = json.load(fr)

    print(model)
    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            cls_name = data["class_name"][0]
            mesh_name = data["obj_name"][0]
            image100_path = data["image100_path"][0]
            plane_features = data["triplane"].cuda()
            # print("plane", plane_features.shape) # ([1, 3, 32, 128, 128])
            
            # generate vector save dir
            # latent_save_dir = os.path.join(latent_dir, cls_name)
            # # os.makedirs(latent_save_dir, exist_ok=True)
            # latent_save_path = os.path.join(latent_save_dir, mesh_name + ".npy")
            # # np.save(latent_save_path, latent[i].detach().unsqueeze(0).cpu().numpy())
            # dataset_dict['data'][cls_name][mesh_name]['latent'] = latent_save_path
            # continue

            if stats_dir is not None:
                plane_features = normalize(plane_features, stats_dir, middle, _range)
            else:
                plane_features = plane_features.clamp(-1.0, 1.0)

            if not random_sample:
                out = model.vae_model(plane_features)
            else:
                out = model.vae_model.sample(1)
            reconstructed_plane_feature, latent_list = out[0], out[-1]


            # ##### test decoder model
            # reconstructed_plane_feature = plane_features.clone()

            # ## add noise
            # # noise_gaussian = torch.randn_like(reconstructed_plane_feature) * 0.02
            # noise_gaussian = torch.rand_like(reconstructed_plane_feature) * 0.1
            # reconstructed_plane_feature += noise_gaussian

            loss_l1 = (reconstructed_plane_feature - plane_features).abs().mean()
            loss_l2 = L2Loss(reconstructed_plane_feature, plane_features)
            print("l1 loss: {}".format(loss_l1), "l2 loss: {}".format(loss_l2))

            if stats_dir is not None:
                reconstructed_plane_feature = unnormalize(reconstructed_plane_feature, stats_dir, middle, _range)
            else:
                reconstructed_plane_feature = reconstructed_plane_feature.clamp(-1.0, 1.0)
            ###### color loss
            color_points = data["color_points"].cuda()
            color_colors = data["color_colors"].cuda()
            color_points_normal = data['color_points_normal'].cuda()
            if "get_expand_color" not in globals():
                from dataloader.dataset_sdfcolor import get_expand_color, single_points_sampler
            expand_points, t = single_points_sampler(color_points, color_points_normal)
            color_points = torch.cat((expand_points, color_points), dim=1)
            expand_colors = get_expand_color(color_colors, t)
            color_colors = torch.cat((expand_colors, color_colors), dim=1)

            pred_rgb_surface = model.decoder_model.forward_rgb(reconstructed_plane_feature, color_points)
            loss_color = (pred_rgb_surface - color_colors).abs().sum(-1).mean()
            print("loss color: {}".format(loss_color))
            ####### save mesh
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            try:
                os.system("cp {} {}".format(image100_path, outdir))
            except:
                pass
            mesh_filename = os.path.join(outdir, "reconstruct.ply")

            # print("reconstructed_plane_feature", reconstructed_plane_feature.shape) # ([1, 3, 32, 128, 128])
            reconstructed_plane_feature_list = [reconstructed_plane_feature, reconstructed_plane_feature]
            plot_texmesh_split(model.decoder_model, reconstructed_plane_feature_list, 512, mesh_filename)
            
            # save triplane:
            # breakpoint()
            # triplane_filename = os.path.join(outdir, "triplane.tar")
            # torch.save(reconstructed_plane_feature.detach().cpu(), triplane_filename)

            # # ###### save latent

            # try:
            #     outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
            #     os.makedirs(outdir, exist_ok=True)
            #     np.save(os.path.join(outdir, "latent.npy"), latent.cpu().numpy())
            #     np.save(os.path.join(outdir, "triplane_recon.npy"), reconstructed_plane_feature.cpu().numpy())
            # except Exception as e:
            #     print(e)
    
    # with open('/apdcephfs_cq3/share_1615605/weixuansun/code/DiffusionSDF/config/stage1_vae_sdfcolor_vector_transformer_v10/alldata_1113_right_with_latent.json', 'w') as fp:
        # json.dump(dataset_dict, fp, indent=2)

@torch.no_grad()
def test_vae_sdfyh_color():
    # load dataset, dataloader, model checkpoint

    test_dataset = DatasetSdfYhColoronly(specs["data_config"], data_type="train")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)

    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()
    else:
        model =CombinedModel(specs=specs, args=args).cuda().eval()

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))



            cls_name = data["class_name"][0]
            mesh_name = data["obj_name"][0]
            image100_path = data["image100_path"][0]
            sdf_plane_features = data["sdf_triplane"].clamp(-1.0, 1.0)
            color_plane_features = data["color_triplane"].clamp(-1.0, 1.0)

            # triplane_geo_path = "/apdcephfs_data_cq2_1/share_1615605/weizheliu/code/git_moa/trisdf/logs/vroid_test/geo/epoch_last/vroid_obj_100757673924144965.tar"
            # triplane_tex_path = "/apdcephfs_data_cq2_1/share_1615605/weizheliu/code/git_moa/trisdf/logs/vroid_test/epoch_last/vroid_obj_100757673924144965.tar"
            # sdf_plane_features = torch.load(triplane_geo_path, map_location=torch.device('cuda:0')).clamp(-1.0, 1.0).unsqueeze(0)
            # color_plane_features = torch.load(triplane_tex_path, map_location=torch.device('cuda:0')).clamp(-1.0, 1.0).unsqueeze(0)

            if not random_sample:
                out = model.vae_model([sdf_plane_features.cuda(), color_plane_features.cuda()])
            else:
                out = model.vae_model.sample(1)
            reconstructed_plane_feature_list, latent_list = out[0], out[-1]

            ####### save mesh
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            try:
                os.system("cp {} {}".format(image100_path, outdir))
            except:
                pass
            mesh_filename = os.path.join(outdir, "reconstruct.ply")
            # reconstructed_plane_feature_list = [sdf_plane_features, color_plane_features]
            plot_texmesh_split(model.decoder_model, reconstructed_plane_feature_list, 512, mesh_filename)
            # # ###### save latent
            # try:
            #     outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
            #     os.makedirs(outdir, exist_ok=True)
            #     np.save(os.path.join(outdir, "latent.npy"), latent.cpu().numpy())
            #     np.save(os.path.join(outdir, "triplane_recon.npy"), reconstructed_plane_feature.cpu().numpy())
            # except Exception as e:
            #     print(e)



@torch.no_grad()
def test_vae_sdf():
    test_dataset = get_dataset(specs, data_type="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    if args.resume is None:
        model = CombinedModel(specs, args)
        # ckpt = "{}.ckpt".format(args.resume_params) if args.resume_params=='last' else "epoch={}.ckpt".format(args.resume_params)
        # resume_params_path = os.path.join(args.exp_dir, ckpt)
        # model.load_state_dict(torch.load(resume_params_path)["state_dict"], strict=False)
        # resume = None
        model = model.cuda().eval()
        # print("load only state_dict from checkpoint: {}".format(resume_params_path))
    elif args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()
    else:
        print("must have a  from a checkpoint")

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
            plane_features = data["sdf_triplane"]
            plane_features = plane_features.clamp(-1.0, 1.0).cuda()
            cls_name = data["class_name"][0]
            mesh_name = data["obj_name"][0]
            image100_path = data["image100_path"][0]

            if not random_sample:
                out = model.vae_model(plane_features)
            else:
                out = model.vae_model.sample(1)
            recon = out[0]
            latent = out[-1]

            triplane = recon

            # ##### save triplane reconsturct
            # os.makedirs(os.path.join(recon_dir, cls_name), exist_ok=True)
            # recon_path = os.path.join(recon_dir, cls_name, mesh_name + ".npy")
            # np.save(recon_path, recon.cpu().numpy())

            #### marching cube and save mesh
            #print("mesh filename: ", mesh_filename)
            # N is the grid resolution for marching cubes; set max_batch to largest number gpu can hold
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            mesh_filename = os.path.join(outdir, "reconstruct")
            mesh.create_mesh(model.decoder_model, triplane, mesh_filename, N=256, max_batch=2**21, from_plane_features=True)

            ###### save front image
            if os.path.exists(image100_path):
                os.system("cp {} {}".format(image100_path, outdir))
            else:
                print("image100_path not exists: {}".format(image100_path))
          
            # ###### save triplane encoded latent         
            # try:
            #     outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
            #     os.makedirs(outdir, exist_ok=True)
            #     np.save(os.path.join(outdir, "latent.npy"), latent.cpu().numpy())
            # except Exception as e:
            #     print(e)

@torch.no_grad()
def test_vae_dmtet():
    # load dataset, dataloader, model checkpoint
    test_dataset = get_dataset(specs=specs, data_type=specs["data_config"]["test_type"], resample=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    # ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    # resume = os.path.join(args.exp_dir, ckpt)
    model = CombinedModel(specs, args)
    # ckpt_dict = torch.load(resume)
    # model.load_state_dict(ckpt_dict["state_dict"])
    model = model.cuda().eval()
    glctx = dr.RasterizeCudaContext()

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            for key, value in data.items():
                if key in ["mv", "mvp", "campos", "img", "triplane"]:
                    data[key] = value.cuda()

            dataclass = data["dataclass"][0]
            mesh_name = data["obj_save_name"][0]
            shaded_dir = data["shaded_dir"][0]

            ##### test 
            plane_features = data["triplane"].clamp(-1.0, 1.0)

            if not random_sample:
                out = model.vae_model(plane_features.cuda())
            else:
                out = model.vae_model.sample(1)
            recon = out[0]
            latent = out[-1]

            # ######## get encoder latent
            # try:
            #     modulation_save_dir = os.path.join(latent_dir, dataclass, mesh_name)
            #     os.makedirs(modulation_save_dir, exist_ok=True)
            #     save_path = os.path.join(modulation_save_dir, "latent.npy")
            #     np.save(save_path, latent.cpu().numpy())
            # except Exception as e:
            #     print(e)

            #########  render reconstruct and save gif
            outdir_recon = os.path.join(recon_dir, "{}/{}".format(dataclass, mesh_name))
            mesh_filename = os.path.join(outdir_recon, "reconstruct")
            os.makedirs(mesh_filename, exist_ok=True)
            try:
                image_path = os.path.join(shaded_dir, "color/cam-0100.png")
                os.system("cp {} {}".format(image_path, outdir_recon))
            except:
                pass

            ## decode and get mesh
            print("mesh filename: ", mesh_filename)
            ##### N is the grid resolution for marching cubes; set max_batch to largest number gpu can hold
            mesh_list = model.decoder_model.xatlas_uvmap_batch(glctx, recon)

            # # ###### test gt
            # recon = data["triplane"].clamp(-1.0, 1.0)
            # # mesh_list = model.decoder_model.xatlas_uvmap_batch(glctx, recon)

            # save mesh
            for idx, base_mesh in enumerate(mesh_list):
                obj.write_obj(mesh_filename, base_mesh)
            
            ##### make gif
            itr_all = 30
            np_img_list = []
            gif_path = os.path.join(outdir_recon, "rotate.gif")
            for itr in range(itr_all):
                target = test_dataset.rotate_scene(itr, itr_all=itr_all)
                np_img = model.decoder_model.render_image(glctx, recon, mesh_list, target)[0][0]
                np_img_list.append(np_img)
            imageio.mimsave(gif_path, np_img_list, 'GIF', duration=0.1)
            


@torch.no_grad()
def test_modulations():
    
    # load dataset, dataloader, model checkpoint
    if specs["training_task"] == "modulation_nosdfgt":
        test_dataset = SdfNogtLoader(specs["data_config"], return_filename=True)
    else:
        test_split = json.load(open(specs["TestSplit"]))
        test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)

    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    # filename for logging chamfer distances of reconstructed meshes
    cd_file = os.path.join(recon_dir, "cd.csv")

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
            if specs["training_task"] == "modulation_nosdfgt":
                point_cloud = data[0]
                filename = data[-1]
                # point_cloud, _, _, _, _, _, filename = data
            else:
                point_cloud, filename = data # filename = path to the csv file of sdf data
            filename = filename[0] # filename is a tuple

            cls_name = filename.split("/")[-3]
            mesh_name = filename.split("/")[-2]
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            mesh_filename = os.path.join(outdir, "reconstruct")
            
            # given point cloud, create modulations (e.g. 1D latent vectors)
            plane_features = model.sdf_model.pointnet.get_plane_features(point_cloud.cuda())  # tuple, 3 items with ([1, D, resolution, resolution])
            plane_features = torch.cat(plane_features, dim=1) # ([1, D*3, resolution, resolution])
            recon = model.vae_model.generate(plane_features) # ([1, D*3, resolution, resolution])
            print("recon feature mean: {}, std: {}".format(torch.mean(recon), torch.std(recon)))
            #print("mesh filename: ", mesh_filename)
            # N is the grid resolution for marching cubes; set max_batch to largest number gpu can hold
            mesh.create_mesh(model.sdf_model, recon, mesh_filename, N=256, max_batch=2**21, from_plane_features=True)

            # load the created mesh (mesh_filename), and compare with input point cloud
            # to calculate and log chamfer distance 
            mesh_log_name = cls_name+"/"+mesh_name
            try:
                evaluate.main(point_cloud, mesh_filename, cd_file, mesh_log_name)
            except Exception as e:
                print(e)


            # save modulation vectors for training diffusion model for next stage
            # filter based on the chamfer distance so that all training data for diffusion model is clean 
            # would recommend visualizing some reconstructed meshes and manually determining what chamfer distance threshold to use
            try:
                # skips modulations that have chamfer distance > 0.0018
                # the filter also weighs gaps / empty space higher
                if not filter_threshold(mesh_filename, point_cloud, 0.0018): 
                    continue
                outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)
                features = model.sdf_model.pointnet.get_plane_features(point_cloud.cuda())
                features = torch.cat(features, dim=1) # ([1, D*3, resolution, resolution])
                latent = model.vae_model.get_latent(features) # (1, D*3)
                np.save(os.path.join(outdir, "latent.npy"), latent.cpu().numpy())
            except Exception as e:
                print(e)

           
@torch.no_grad()
def test_generation():

    # load model 
    if args.resume == 'finetune': # after second stage of training 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # loads the sdf and vae models
            model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, args=args, strict=False) 

            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)

            model = model.cuda().eval()
    else:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    conditional = specs["diffusion_model_specs"]["cond"] 

    if not conditional:
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            mesh.create_mesh(model.sdf_model, plane_feature, recon_dir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            
    else:
        # load dataset, dataloader, model checkpoint
        test_split = json.load(open(specs["TestSplit"]))
        test_dataset = PCloader(specs["DataSource"], test_split, pc_size=specs.get("PCsize",1024), return_filename=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

                point_cloud, filename = data # filename = path to the csv file of sdf data
                filename = filename[0] # filename is a tuple

                cls_name = filename.split("/")[-3]
                mesh_name = filename.split("/")[-2]
                outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)

                # filter, set threshold manually after a few visualizations
                if args.filter:
                    threshold = 0.08
                    tmp_lst = []
                    count = 0
                    while len(tmp_lst)<args.num_samples:
                        count+=1
                        samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True) # batch should be set to max number GPU can hold
                        plane_features = model.vae_model.decode(samples)
                        # predicting the sdf values of the point cloud
                        perturbed_pc_pred = model.sdf_model.forward_with_plane_features(plane_features, perturbed_pc.repeat(args.num_samples, 1, 1))
                        consistency = F.l1_loss(perturbed_pc_pred, torch.zeros_like(perturbed_pc_pred), reduction='none')
                        loss = reduce(consistency, 'b ... -> b', 'mean', b = consistency.shape[0]) # one value per generated sample 
                        #print("consistency shape: ", consistency.shape, loss.shape, consistency[0].mean(), consistency[1].mean(), loss) # cons: [B,N]; loss: [B]
                        thresh_idx = loss<=threshold
                        tmp_lst.extend(plane_features[thresh_idx])

                        if count > 5: # repeat this filtering process as needed 
                            break
                    # skip the point cloud if cannot produce consistent samples or 
                    # just use the samples that are produced if comparing to other methods
                    if len(tmp_lst)<1: 
                        continue
                    plane_features = tmp_lst[0:min(10,len(tmp_lst))]

                else:
                    # for each point cloud, the partial pc and its conditional generations are all saved in the same directory 
                    samples, perturbed_pc = model.diffusion_model.generate_from_pc(point_cloud.cuda(), batch=args.num_samples, save_pc=outdir, return_pc=True)
                    plane_features = model.vae_model.decode(samples)
                
                for i in range(len(plane_features)):
                    plane_feature = plane_features[i].unsqueeze(0)
                    mesh.create_mesh(model.sdf_model, plane_feature, outdir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            


@torch.no_grad()
def test_generation_from_image_sdf():
    # load model 
    if args.resume == 'finetune': # after second stage of training 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # loads the sdf and vae models
            model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, args=args, strict=False) 

            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)

            model = model.cuda().eval()
    else:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    conditional = specs["diffusion_model_specs"]["cond"] 

    if not conditional:
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            mesh.create_mesh(model.sdf_model, plane_feature, recon_dir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            
    else:
        # load dataset, dataloader, model checkpoint
        test_dataset = get_dataset(specs, data_type="test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                if idx > args.num_samples:
                    break
                pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

                latent_image = data["latent_image"]
                filename_path = data["filename"]
                filename = filename_path[0] # filename is a tuple

                cls_name = filename.split("/")[-5]
                mesh_name = filename.split("/")[-4]
                outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)

                #### test text control
                # import pdb;pdb.set_trace()
                # latent_text_tensor = torch.from_numpy(np.load("/apdcephfs_cq3/share_1615605/neoshang/code/open_clip/output/00001.npy")).float().cuda()
                # latent_image = latent_text_tensor
                if not random_sample:
                    samples = model.diffusion_model.generate_from_latent(latent_image.cuda(), batch=5)
                else:
                    samples = model.diffusion_model.generate_unconditional(num_samples=5)
                    
                plane_features = model.vae_model.decode(samples)
                
                try:
                    os.system("cp {} {}".format(filename, outdir))
                except:
                    pass

                for i in range(len(plane_features)):
                    plane_feature = plane_features[i].unsqueeze(0)
                    # plane_feature = (plane_feature / 2 + 0.5) * 1 + (-0.5)
                    mesh.create_mesh(model.sdf_model, plane_feature, outdir+"/{}_recon".format(i), N=512, max_batch=2**21, from_plane_features=True)

@torch.no_grad()
def test_generation_unconditioned_sdf():

    # load model checkpoint
    model = CombinedModel(specs=specs, args=args).cuda().eval()
    
    # tqdm
    for i in tqdm(range(args.num_samples), desc="Generating", total=args.num_samples):
        sample = model.diffusion_model.generate_unconditional(1)

        plane_features = model.vae_model.decode(sample)
            
        plane_features = [plane_features, plane_features]

        ####### save mesh
        mesh_filename = os.path.join(recon_dir, "random_{}.ply".format(i))
        plot_texmesh_split(model.decoder_model, plane_features, 512, mesh_filename)
        save_latent = True
        # save_latent = False
        if save_latent:
            torch.save(sample.squeeze(), os.path.join(recon_dir, "random_{}.pt".format(i)))
            
            
@torch.no_grad()
def test_generation_unconditioned_geosdf():

    # load model checkpoint
    model = CombinedModel(specs=specs, args=args).cuda().eval()
    
    # tqdm
    for i in tqdm(range(args.num_samples), desc="Generating", total=args.num_samples):
        sample = model.diffusion_model.generate_unconditional(1)
        spatial = True
        if spatial:
            sample = sample.reshape(1, 4, 8, 24)
        plane_features = model.vae_model.decode(sample)

        ####### save mesh
        mesh_filename = os.path.join(recon_dir, "random_{}.ply".format(i))
        plot_geomesh_split(model.decoder_model, plane_features, 512, mesh_filename)
        save_latent = True
        # save_latent = False
        if save_latent:
            torch.save(sample.squeeze(), os.path.join(recon_dir, "random_{}.pt".format(i)))
            

@torch.no_grad()
def test_generation_one_stage_unconditioned_sdf():

    # load model checkpoint
    model = CombinedModel(specs=specs, args=args).cuda().eval()
    
    # tqdm
    for i in tqdm(range(args.num_samples), desc="Generating", total=args.num_samples):
        plane_features = model.diffusion_model.generate_unconditional(1)
        plane_features = [plane_features, plane_features]

        ####### save mesh
        mesh_filename = os.path.join(recon_dir, "random_{}.ply".format(i))
        plot_texmesh_split(model.decoder_model, plane_features, 512, mesh_filename)
        # # save_latent = True
        # save_latent = False
        # if save_latent:
        #     torch.save(sample.squeeze(), os.path.join(recon_dir, "random_{}.pt".format(i)))

@torch.no_grad()
def test_generation_from_image_sdf_cat():
    # load dataset, dataloader, model checkpoint

    test_dataset = get_dataset(specs=specs, data_type="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)

    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()
    else:
        model =CombinedModel(specs=specs, args=args).cuda().eval()

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            cls_name = data["class_name"][0]
            mesh_name = data["obj_name"][0]
            image100_path = data["image100_path"][0]
            latent_image = data["latent_image"]

            #### test text control
            # import pdb;pdb.set_trace()
            # latent_text_tensor = torch.from_numpy(np.load("/apdcephfs_cq3/share_1615605/neoshang/code/open_clip/output/00001.npy")).float().cuda()
            # latent_image = latent_text_tensor
            if not random_sample:
                samples = model.diffusion_model.generate_from_latent(latent_image.cuda(), batch=1)
            else:
                samples = model.diffusion_model.generate_unconditional(num_samples=5)
            
            dim = samples.shape[1]
            z_geo = samples[:, :dim//2]
            z_tex = samples[:, dim//2:]

            [sdf_plane_features, color_plane_features] = model.vae_model.decode([z_geo, z_tex])

            ####### save mesh
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            try:
                os.system("cp {} {}".format(image100_path, outdir))
            except:
                pass

            for i in range(len(sdf_plane_features)):
                mesh_filename = os.path.join(outdir, "reconstruct" + str(i) + ".ply")
                plot_texmesh_split(model.decoder_model, [sdf_plane_features, color_plane_features], 512, mesh_filename)


@torch.no_grad()
def test_generation_from_image_dmtet():
    glctx = dr.RasterizeCudaContext()
    # load model 
    if args.resume == 'finetune': # after second stage of training 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # loads the sdf and vae models
            model = CombinedModel(specs=specs, args=args)
            model = model.cuda().eval()
    else:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    conditional = specs["diffusion_config"]["diffusion_model_specs"]["cond"]

    if not conditional:
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            mesh.create_mesh(model.sdf_model, plane_feature, recon_dir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            
    else:
        # load dataset, dataloader, model checkpoint
        test_dataset = get_dataset(specs, data_type="test")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                if idx > args.num_samples:
                    break
                pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

                latent_image = data["latent_image"]
                filename_path = data["image100_path"]
                filename = filename_path[0] # filename is a tuple

                cls_name = filename.split("/")[-5]
                mesh_name = filename.split("/")[-4]
                mesh_name_list = [mesh_name]
                latent_list = [latent_image]

                # ### test custom latent
                # import pdb;pdb.set_trace()
                # print("test custom latent, if not, comment this block")
                # latent_dir = "/apdcephfs_cq3/share_1615605/neoshang/code/open_clip/latents_output/texts_3"
                # latent_list = []
                # mesh_name_list = []
                # for latent_name in os.listdir(latent_dir):
                #     if not latent_name.endswith(".npy"):
                #         continue
                #     latent_path = os.path.join(latent_dir, latent_name)
                #     latent_tensor = torch.from_numpy(np.load(latent_path)).float().cuda()
                #     latent_list.append(latent_tensor)
                #     mesh_name_list.append(latent_name.split('.')[0])

                #### infer and save mesh
                for mesh_name, latent_image in zip(mesh_name_list, latent_list):
                    outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
                    os.makedirs(outdir, exist_ok=True)
                    if not random_sample:
                        samples = model.diffusion_model.generate_from_latent(latent_image.cuda(), batch=5)
                    else:
                        samples = model.diffusion_model.generate_unconditional(num_samples=5)
                    
                    plane_features = model.vae_model.decode(samples)
                    
                    try:
                        os.system("cp {} {}".format(filename, outdir))
                    except:
                        pass

                    mesh_list = model.decoder_model.xatlas_uvmap_batch(glctx, plane_features)

                    # save mesh
                    for idx, base_mesh in enumerate(mesh_list):
                        save_dir = os.path.join(outdir, str(idx))
                        os.makedirs(save_dir, exist_ok=True)
                        obj.write_obj(save_dir, base_mesh)

@torch.no_grad()
def test_generation_from_image_stage3():
    # load model 
    if args.resume == 'finetune': # after second stage of training 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # loads the sdf and vae models
            model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, args=args, strict=False) 

            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)

            model = model.cuda().eval()
    else:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()

    conditional = specs["diffusion_model_specs"]["cond"] 

    if not conditional:
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            mesh.create_mesh(model.sdf_model, plane_feature, recon_dir+"/{}_recon".format(i), N=128, max_batch=2**21, from_plane_features=True)
            
    else:
        # load dataset, dataloader, model checkpoint
        test_dataset = DatasetSdfYh(specs["data_config"], return_triplane=True, return_image=True, return_filename=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)

        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                if idx > args.num_samples:
                    break
                pbar.set_description("Files generated: {}/{}".format(idx, len(test_dataloader)))

                _, psdxyz, points, masks, normal, psdsdfs_gt, plane_features, latent_image = [x.cuda() for x in data[:-1]]
                filename_path = data[-1]
                filename = filename_path[0] # filename is a tuple

                cls_name = filename.split("/")[-3]
                mesh_name = filename.split("/")[-2]
                outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
                os.makedirs(outdir, exist_ok=True)
                # for each point cloud, the partial pc and its conditional generations are all saved in the same directory 
                if not random_sample:
                    samples = model.diffusion_model.generate_from_latent(latent_image.cuda(), batch=5)
                else:
                    samples = model.diffusion_model.generate_unconditional(num_samples=5)

                triplane_features = model.vae_model.decode(samples)

                # triplane_features, latent = model.vae_model.sample(5)
              
                for i in range(len(triplane_features)):
                    plane_features = triplane_features[i].unsqueeze(0)
                    plane_features = (plane_features / 2 + 0.5) * 1 + (-0.5)
                    mesh.create_mesh(model.sdf_model, plane_features, outdir+"/{}_recon".format(i), N=256, max_batch=2**21, from_plane_features=True)



@torch.no_grad()
def test_vae_img_sdfcolor():
    # load dataset, dataloader, model checkpoint
    test_dataset = DatasetImgSdfColor(specs["data_config"], data_type=specs["data_config"]["test_type"], resample=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    if args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()
    else:
        model = CombinedModel(specs=specs, args=args).cuda().eval()
    

    # json_path = '/apdcephfs_cq3/share_1615605/neoshang/code/rendering_free_onetri/savedir/test_128_v13/alldata_1113_right.json'
    # with open(json_path, 'r') as fr:
    #     dataset_dict = json.load(fr)

    # print(model)
    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            cls_name = data["class_name"][0]
            mesh_name = data["obj_name"][0]
            image100_path = data["image100_path"][0]
            image = data["image"].cuda()
            
            if not random_sample:
                 predicted_planes, _ = model.vae_model(image)
            else:
                 predicted_planes, _ = model.vae_model.sample(1)

            # print("plane shape: ", predicted_planes.shape)

            # ###### color loss
            # color_points = data["color_points"].cuda()
            # color_colors = data["color_colors"].cuda()
            # color_points_normal = data['color_points_normal'].cuda()
            # if "get_expand_color" not in globals():
            #     from dataloader.dataset_sdfcolor import get_expand_color, single_points_sampler
            # expand_points, t = single_points_sampler(color_points, color_points_normal)
            # color_points = torch.cat((expand_points, color_points), dim=1)
            # expand_colors = get_expand_color(color_colors, t)
            # color_colors = torch.cat((expand_colors, color_colors), dim=1)

            # pred_rgb_surface = model.decoder_model.forward_rgb(predicted_planes, color_points)
            # loss_color = (pred_rgb_surface - color_colors).abs().sum(-1).mean()
            # print("loss color: {}".format(loss_color))
            ####### save mesh
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            try:
                os.system("cp {} {}".format(image100_path, outdir))
            except:
                pass
            mesh_filename = os.path.join(outdir, "reconstruct.ply")

            # predicted_planes = predicted_planes.squeeze()
            predicted_planes_list = [predicted_planes, predicted_planes]
            plot_texmesh_split(model.decoder_model, predicted_planes_list, 512, mesh_filename)
            
            # save triplane:
            # breakpoint()
            # triplane_filename = os.path.join(outdir, "triplane.tar")
            # torch.save(reconstructed_plane_feature.detach().cpu(), triplane_filename)

            # # ###### save latent

            # try:
            #     outdir = os.path.join(latent_dir, "{}/{}".format(cls_name, mesh_name))
            #     os.makedirs(outdir, exist_ok=True)
            #     np.save(os.path.join(outdir, "latent.npy"), latent.cpu().numpy())
            #     np.save(os.path.join(outdir, "triplane_recon.npy"), reconstructed_plane_feature.cpu().numpy())
            # except Exception as e:
            #     print(e)
    
    # with open('/apdcephfs_cq3/share_1615605/weixuansun/code/DiffusionSDF/config/stage1_vae_sdfcolor_vector_transformer_v10/alldata_1113_right_with_latent.json', 'w') as fp:
        # json.dump(dataset_dict, fp, indent=2)



@torch.no_grad()
def test_vae_sdfgeo():
    # load dataset, dataloader, model checkpoint
    stats_dir = None
    if os.path.exists(os.path.join(args.exp_dir, "stats")):
        stats_dir = os.path.join(args.exp_dir, "stats")
        min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
        max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
        _range = max_values - min_values
        middle = (min_values + max_values) / 2
    test_dataset = DatasetSdfGeo(specs["data_config"], data_type=specs["data_config"]["test_type"], resample=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    # if args.resume is not None:
    #     ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    #     resume = os.path.join(args.exp_dir, ckpt)
    #     model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args).cuda().eval()
    # else:
    model = CombinedModel(specs=specs, args=args).cuda().eval()
    
    obj_count = {}
    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx >= args.num_samples:
                break
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            cls_name = data["class_name"][0]
            mesh_name = data["obj_name"][0]
            image100_path = data["obj_path"][0]

            # if cls_name in obj_count.keys():
            #     obj_count[cls_name] += 1
            # else:
            #     obj_count[cls_name] = 1
            # if obj_count[cls_name] < 50:
            #     print("skip ", obj_count[cls_name], cls_name, mesh_name)
            #     continue

            plane_features = data["triplane"].cuda()


            if stats_dir is not None:
                plane_features = normalize(plane_features, stats_dir, middle, _range)
            else:
                plane_features = plane_features.clamp(-1.0, 1.0)

            if not random_sample:
                out = model.vae_model(plane_features)
            else:
                out = model.vae_model.sample(1)
            reconstructed_plane_feature, latent_list = out[0], out[-1]

            loss_l1 = (reconstructed_plane_feature - plane_features).abs().mean()
            loss_l2 = L2Loss(reconstructed_plane_feature, plane_features)
            print("{}/{} , l1 loss: {}".format(cls_name, mesh_name, loss_l1), "l2 loss: {}".format(loss_l2))

            if stats_dir is not None:
                reconstructed_plane_feature = unnormalize(reconstructed_plane_feature, stats_dir, middle, _range)
            else:
                reconstructed_plane_feature = reconstructed_plane_feature.clamp(-1.0, 1.0)
            ####### save mesh
            outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, mesh_name))
            os.makedirs(outdir, exist_ok=True)
            try:
                os.system("ln -s {} {}".format(image100_path, outdir))
            except:
                pass

            mesh_filename = os.path.join(outdir, "reconstruct.ply")
            plot_geomesh_split(model.decoder_model, reconstructed_plane_feature, 512, mesh_filename)
            # print(outdir)
            # reconstructed_plane_feature = plane_features
            # mesh_filename = os.path.join(outdir, "reconstruct_512.ply")
            # plot_geomesh_split(model.decoder_model, reconstructed_plane_feature, 512, mesh_filename)
            # mesh_filename = os.path.join(outdir, "reconstruct_128.ply")
            # plot_geomesh_split(model.decoder_model, reconstructed_plane_feature, 128, mesh_filename)
            # mesh_filename = os.path.join(outdir, "reconstruct_64.ply")
            # plot_geomesh_split(model.decoder_model, reconstructed_plane_feature, 64, mesh_filename)


            # reconstructed_plane_feature = torch.nn.functional.interpolate(reconstructed_plane_feature.squeeze(0), scale_factor=0.5, mode='bilinear').unsqueeze(0)
            # import time
            # start = time.time()
            # for _ in range(10):
            # plot_geomesh_split(model.decoder_model, reconstructed_plane_feature, 128, mesh_filename)
            # end = time.time()
            # print("time: ", end - start) # 64 time:  0.5808906555175781, 256 14.026119232177734, 128 1.8966727256774902
            # breakpoint()
            # save_latent = True
            # if save_latent:
            #     torch.save(latent_list, os.path.join(outdir, "latent.pt"))


@torch.no_grad()
def test_generation_text_conditioned_sdf():

    if "eval_metrics" in specs.keys():
        print("Generating sample for evaluating metrics")
        from utils.mesh import extract_surface_points, load_and_downsample_ply
        n_sample = specs["eval_metrics"]["n_sample"]
        n_points = specs["eval_metrics"]["n_points"]
        dataset = DatasetDiffusionTextConditioned(specs["data_config"], data_type=specs["data_config"]["test_type"], fix_index=True)
        model = CombinedModel(specs=specs, args=args).cuda().eval()
        
        # tqdm iterate over dataset
        # num_samples = min(args.num_samples, len(dataset)) 
        num_samples = len(dataset)
        for i in tqdm(range(num_samples), desc="Generating", total=num_samples):
            data = dataset[i]
            text_prompt = data["text"]
            text_feature = torch.from_numpy(data["text_feature"])
            obj_name = data["obj_name"]
            obj_path = data["obj_path"]
            generated_samples = []
            for j in range(n_sample):
                sample = model.diffusion_model.generate_from_latent(text_feature.cuda(), batch=1)
                spatial = True
                if spatial:
                    sample = sample.reshape(1, 4, 8, 24)
                plane_features = model.vae_model.decode(sample)

                ####### save mesh
                os.makedirs(os.path.join(recon_dir, obj_name), exist_ok=True)
                mesh_filename = os.path.join(recon_dir, obj_name, "generation_{}.ply".format(j))
                plot_geomesh_split(model.decoder_model, plane_features, 256, mesh_filename)
                points = load_and_downsample_ply(mesh_filename, n_points)
                generated_samples.append(points)
            generated_samples = np.stack(generated_samples, axis=0)
            np.save(os.path.join(recon_dir, obj_name, "generated_samples.npy"), generated_samples)

            # move the original object to the folder
            os.system("cp {} {}".format(obj_path, os.path.join(recon_dir, obj_name, "original.obj")))
            # save text
            with open(os.path.join(recon_dir, obj_name, "text_prompt.txt"), "w") as f:
                f.write(text_prompt)
            # load original object and downsample points
            original_points = extract_surface_points(os.path.join(recon_dir, obj_name, "original.obj"), n_points)
            original_points = torch.from_numpy(np.array(original_points.points))
            np.save(os.path.join(recon_dir, obj_name, "original_points.npy"), original_points)

    else:
        # load model checkpoint
        dataset = DatasetDiffusionTextConditioned(specs["data_config"], data_type=specs["data_config"]["test_type"], fix_index=0)
        model = CombinedModel(specs=specs, args=args).cuda().eval()
        
        num_samples = min(args.num_samples, len(dataset))
        # tqdm iterate over datase
        for i in tqdm(range(num_samples), desc="Generating", total=num_samples):
            data = dataset[i]
            text_prompt = data["text"]
            text_feature = torch.from_numpy(data["text_feature"])
            obj_name = data["obj_name"]
            obj_path = data["obj_path"]
            sample = model.diffusion_model.generate_from_latent(text_feature.cuda(), batch=1)
            spatial = True
            if spatial:
                sample = sample.reshape(1, 4, 8, 24)
            plane_features = model.vae_model.decode(sample)

            ####### save mesh
            os.makedirs(os.path.join(recon_dir, obj_name), exist_ok=True)
            mesh_filename = os.path.join(recon_dir, obj_name, "generation.ply")
            plot_geomesh_split(model.decoder_model, plane_features, 256, mesh_filename)
            os.system("cp {} {}".format(obj_path, os.path.join(recon_dir, obj_name, "original.obj")))
            # save text
            with open(os.path.join(recon_dir, obj_name, "text_prompt.txt"), "w") as f:
                f.write(text_prompt)
            save_latent = False
            # save_latent = False
            if save_latent:
                torch.save(sample.squeeze(), os.path.join(recon_dir, obj_name, "latent.pt"))
            

@torch.no_grad()
def test_generation_partial_conditioned_sdf():

    # load model checkpoint
    dataset = DatasetDiffusionPartialConditioned(specs["data_config"], data_type=specs["data_config"]["test_type"])
    model = CombinedModel(specs=specs, args=args).cuda().eval()
    
    num_samples = min(args.num_samples, len(dataset))
    # tqdm iterate over datase
    for i in tqdm(range(num_samples), desc="Generating", total=num_samples):
        data = dataset[i]
        text_feature = torch.from_numpy(data["text_feature"])
        obj_name = data["obj_name"]
        obj_path = data["obj_path"]
        sample = model.diffusion_model.generate_from_latent(text_feature.cuda(), batch=1)
        spatial = True
        if spatial:
            sample = sample.reshape(1, 16, 8, 24)
        plane_features = model.vae_model.decode(sample)

        ####### save mesh
        os.makedirs(os.path.join(recon_dir, obj_name), exist_ok=True)
        mesh_filename = os.path.join(recon_dir, obj_name, "generation.ply")
        plot_geomesh_split(model.decoder_model, plane_features, 256, mesh_filename)
        os.system("cp {} {}".format(obj_path, os.path.join(recon_dir, obj_name, "original.obj")))
        save_latent = False
        # save_latent = False
        if save_latent:
            torch.save(sample.squeeze(), os.path.join(recon_dir, obj_name, "latent.pt"))
            
            
@torch.no_grad()
def test_generation_pix3d_conditioned_sdf():
    # load model checkpoint
    dataset = DatasetDiffusionPix3D(specs["data_config"], data_type=specs["data_config"]["test_type"])
    model = CombinedModel(specs=specs, args=args).cuda().eval()


    
    obj_count = {}
    num_samples = min(args.num_samples, len(dataset))
    arr = np.arange(2887)
    arr1 = arr[253: 253+530]
    arr2 = arr[783: 783+530]
    arr3 = arr[1313: 1313+530]
    arr4 = arr[1843: 1843+530]
    arr5 = arr[2373: 2373+530]
    # for i in tqdm(range(num_samples), desc="Generating", total=num_samples):
    # for i in tqdm(arr1, desc="Generating 253 to 783", total=len(arr1)):
    # for i in tqdm(arr2, desc="Generating 783 to 1313", total=len(arr2)):
    # for i in tqdm(arr3, desc="Generating 1313 to 1843", total=len(arr3)):  98/530 
    # for i in tqdm(arr4, desc="Generating 1843 to 2373", total=len(arr4)):
    for i in tqdm(arr5, desc="Generating 2373 to 2887", total=len(arr5)):
        data = dataset[i]
        category = data["category"]
        img_idx = data["obj_name"].split(";")[0].split("/")[-1].split(".")[0]
        image_path = data["image_path"]
        obj_path = data["obj_path"]
        img_feature = data["latent_image"].cuda()
        # if category in obj_count.keys():
        #     obj_count[category] += 1
        # else:
        #     obj_count[category] = 1
        # if obj_count[category] > 5:
        #     continue
            
        # for i in range(1):
        sample = model.diffusion_model.generate_from_latent(img_feature, batch=1)
        spatial = True
        if spatial:
            sample = sample.reshape(1, 4, 8, 24)
        plane_features = model.vae_model.decode(sample)

        ####### save mesh
        os.makedirs(os.path.join(recon_dir, category, img_idx), exist_ok=True)
        mesh_filename = os.path.join(recon_dir, category, img_idx, f"reconstructio_{0}.ply")
        plot_geomesh_split(model.decoder_model, plane_features, 512, mesh_filename)
        # copy image and mesh
        os.system("ln -s {} {}".format(obj_path, os.path.join(recon_dir, category, img_idx, "original.obj")))
        os.system("ln -s {} {}".format(image_path, os.path.join(recon_dir, category, img_idx, "image.png")))
        # torch.save(sample.squeeze(), os.path.join(recon_dir, category, img_idx, "latent.pt"))
                

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default='last',
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )
    arg_parser.add_argument(
        "--resume_params", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )
    arg_parser.add_argument("--num_samples", "-n", default=99999, type=int, help='number of samples to generate and reconstruct')
    arg_parser.add_argument("--random_sample", action="store_true", default=False, help='whether to random sample from noise')
    arg_parser.add_argument("--filter", default=False, help='whether to filter when sampling conditionally')

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs_test.json")))
    # config_exp_dir = args.exp_dir.replace("store/", "")
    # os.makedirs(config_exp_dir, exist_ok=True)
    # json_path_list = glob(args.exp_dir + "/*.json")
    # for json_path in json_path_list:
    #     json_name = os.path.basename(json_path)
    #     json_save2config_path = os.path.join(config_exp_dir, json_name)
    #     os.system("cp {} {}".format(json_path, json_save2config_path))
    #     os.system("cp {} {}".format(json_path, json_save2config_path))
    print(specs["Description"])

    svae_path = os.path.join(specs['save_path'], args.exp_dir.split('/')[-2]) if 'save_path' in specs.keys() else args.exp_dir
    recon_dir = os.path.join(svae_path, "recon" + time.strftime('%Y-%m-%d-%H:%M:%S'))
    print("Saving to {}".format(recon_dir))
    os.makedirs(recon_dir, exist_ok=True)
    json_path_list = glob(args.exp_dir + "/*.json")
    for json_path in json_path_list:
        json_name = os.path.basename(json_path)
        json_save2config_path = os.path.join(recon_dir, json_name)
        os.system("cp {} {}".format(json_path, json_save2config_path))
    
    random_sample = args.random_sample
    # if random_sample:
    #     print("test random")
    # else:
    #     print("test with condition")

    if specs['training_task'] == 'modulation':
        latent_dir = os.path.join(args.exp_dir, "modulations" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        test_modulations()
    elif specs['training_task'] == 'modulation_nosdfgt':
        latent_dir = os.path.join(args.exp_dir, "modulations" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        test_modulations()
        # test_modulations_vae_sample()
    elif specs['training_task'] == 'vae':
        latent_dir = os.path.join(args.exp_dir, "modulations" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        test_vae()
    elif specs["training_task"] in ['vae_sdfcolor', "vae_sdfcolor_L1L2"]:
        latent_dir = os.path.join(args.exp_dir, "latent" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        test_vae_sdfcolor()  # support random_sample option
    elif specs['training_task'] in ['vae_sdf', 'vae_sdfyh', "vae_sdfyh_spacial_rollout"]:
        latent_dir = os.path.join(args.exp_dir, "latent" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        test_vae_sdf()  # support random_sample option
    elif specs['training_task'] in ['vae_sdfyh_coloronly']:
        latent_dir = os.path.join(args.exp_dir, "latent" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        test_vae_sdfyh_color()  # support random_sample option
    elif specs['training_task'] in ['vae_dmtet']:
        latent_dir = os.path.join(args.exp_dir, "latent" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        test_vae_dmtet()  # support random_sample option
    elif specs['training_task'] == 'combined':
        test_generation()
    elif specs["training_task"] == "diffusion_image_cond_cat":
        test_generation_from_image_sdf_cat()
    elif specs["training_task"] == "diffusion_image_cond" and specs["triplane_type"] == "sdf":
        test_generation_from_image_sdf()
    elif specs["training_task"] == "diffusion_uncond" and specs["triplane_type"] == "sdf":
        test_generation_unconditioned_sdf()
    elif specs["training_task"] == "diffusion_uncond_geo" and specs["triplane_type"] == "sdf":
        test_generation_unconditioned_geosdf()
    elif specs["training_task"] == "diffusion_image_cond" and specs["triplane_type"] == "dmtet":
        test_generation_from_image_dmtet()
    elif specs["training_task"] == "diffusion_text_cond" and specs["triplane_type"] == "geosdf":
        test_generation_text_conditioned_sdf()
    elif specs["training_task"] == "diffusion_partial_cond" and specs["triplane_type"] == "geosdf":
        test_generation_partial_conditioned_sdf()
    elif specs["training_task"] == "diffusion_pix3d_cond" and specs["triplane_type"] == "geosdf":
        test_generation_pix3d_conditioned_sdf()
    elif specs["training_task"] == "combined_sdfyh_diffusion":
        test_generation_from_image_stage3()  # support random_sample option
    elif (specs["training_task"] == "one_diffusion_plane_uncond" or specs["training_task"] == "one_diffusion_uncond") and specs["triplane_type"] == "sdf":
        test_generation_one_stage_unconditioned_sdf()
    elif specs['training_task'] == 'svr_sdfcolor':
        test_vae_img_sdfcolor()
    elif specs['training_task'] == 'vae_sdfgeo':
        test_vae_sdfgeo()  # support random_sample option
        
    else:
        raise NotImplementedError
    

  
