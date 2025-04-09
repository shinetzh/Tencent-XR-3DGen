#!/usr/bin/env python3

import torch
import imageio
from glob import glob
import torch.utils.data 
from easydict import EasyDict as edict
import nvdiffrast.torch as dr

from triplane_nvdiffrec.render import obj
import os
import time
import uuid
import json
import time
from einops import rearrange, reduce
import numpy as np
import warnings

# add paths in model/__init__.py for new models
from models import * 
from utils.reconstruct import *
from diff_utils.helpers import * 

import dataloader.util_dmtet as util

obj_cfg_path = "cam_parameters.json"
with open(obj_cfg_path, 'r') as fr:
    obj_cfg = json.load(fr)
fovy = util.focal_length_to_fovy(obj_cfg['cam-0001']['k'][0][0], 512)
proj = util.perspective(fovy, 1.0, 0.01, 100)


def rotate_scene(itr_all=30, batch_size=8, cam_radius=2.):
    """only for eval, render rotate circle imgs
    """
    # Smooth rotation for display.
    mv_list = []
    mvp_list = []
    campos_list = []
    for itr in range(itr_all):
        ang    = (itr / itr_all) * np.pi * 2
        mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        mv_list.append(mv)
        mvp_list.append(mvp)
        campos_list.append(campos)
    # make mvp [batch, n_view, 4, 4]
    # campos [batch, n_view, 3]
    # background [batch, n_view, h, w, 3]
    mvp = torch.concat([x.unsqueeze(0).unsqueeze(0) for x in mvp_list], dim=1).repeat(batch_size, 1, 1, 1)
    campos = torch.concat([x.unsqueeze(0).unsqueeze(0) for x in campos_list], dim=1).repeat(batch_size, 1, 1)

    return {
        'mvp': mvp.cuda(),
        'campos': campos.cuda(),
        'resolution': [512, 512],
        'spp': 1,
        'background': torch.ones((batch_size, itr_all, 512, 512, 3), dtype=torch.float32, device='cuda'),
    }


def init_model_dmtet(args):
    specs = json.load(open(os.path.join(args.exp_dir, "release_specs_test.json")))
    config_exp_dir = args.exp_dir.replace("store/", "")
    os.makedirs(config_exp_dir, exist_ok=True)
    json_path_list = glob(args.exp_dir + "/*.json")
    for json_path in json_path_list:
        json_name = os.path.basename(json_path)
        json_save2config_path = os.path.join(config_exp_dir, json_name)
        os.system("cp {} {}".format(json_path, json_save2config_path))
    print(specs["Description"])

    if args.resume == 'finetune':
        specs['modulation_ckpt_path'] = args.modulation_ckpt_path
        specs['diffusion_ckpt_path'] = args.diffusion_ckpt_path

    recon_dir = os.path.join(args.exp_dir, "recon_webui_tmp")
    os.makedirs(recon_dir, exist_ok=True)
    
    with open(specs["decoder_config"]["config_json"], 'r') as fr:
        flags = edict(json.load(fr))
    glctx = dr.RasterizeCudaContext()
    # load model 
    if args.resume == 'finetune': # after second stage of training 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # loads the sdf and vae models
            print("vae+decoder model path: {}".format(specs["modulation_ckpt_path"]))
            model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, args=args, flags=flags, strict=False) 

            # loads the diffusion model; directly calling diffusion_model.load_state_dict to prevent overwriting sdf and vae params
            print("deffusion model path: {}".format(specs["diffusion_ckpt_path"]))
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            new_state_dict = {}
            for k,v in ckpt['state_dict'].items():
                new_key = k.replace("diffusion_model.", "") # remove "diffusion_model." from keys since directly loading into diffusion model
                new_state_dict[new_key] = v
            model.diffusion_model.load_state_dict(new_state_dict)

            model = model.cuda().eval()
    else:
        print("stage3 model path: {}".format(args.resume))
        ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        model = CombinedModel.load_from_checkpoint(resume, specs=specs, args=args, flags=flags).cuda().eval()
    return model, recon_dir, glctx



class DmtetModel:
  def __init__(self, args):
    args = edict(args)
    self.model, self.recon_dir, self.glctx = init_model_dmtet(args)
    self.class_free_weight = args.class_free_weight

  def render_image(self, plane_features, num=30, view_max=61):
    print("render_image...")
    objs_num_all = plane_features.shape[0]
    objs_num_batch = view_max // num
    images_list = []
    start = 0
    while start < objs_num_all:
      if start + objs_num_batch >= objs_num_all:
         end = objs_num_all
      else:
         end = start + objs_num_batch
      target_cam_positions = rotate_scene(itr_all=num, batch_size=end - start)
      np_imgs = self.model.decoder_model.render_image(self.glctx, plane_features[start:end], target_cam_positions)
      start = end
      images_list.append(np_imgs)
    imgs_cat = np.concatenate(images_list, axis=0)
    return imgs_cat
  
  def get_front_images(self, plane_features):
    print("get_front_image...")
    imgs_cat = self.render_image(plane_features, num=1)
    return imgs_cat
  
  def save_mesh_gif(self, plane_features, mesh_list):
    print("save_mesh_gif...")
    # save mesh and gif
    uid = str(uuid.uuid4())
    suid = ''.join(uid.split('-'))
    result_path_list = []
    date_str = time.strftime('%Y%m%d')
    time_str = time.strftime('%H%M%S')
    imgs_cat = self.render_image(plane_features, num = 30)
    images_list = [x for x in imgs_cat]
    for idx, (image_list, mesh) in enumerate(zip(images_list, mesh_list)):
        image_list = [x for x in image_list]
        save_dir = os.path.join(self.recon_dir, date_str, time_str+suid, str(idx))
        save_tar_path = os.path.join(self.recon_dir, date_str, time_str+suid,  str(idx) + ".tar")
        os.makedirs(save_dir, exist_ok=True)
        obj.write_obj(save_dir, mesh)
        os.system("tar -cvf {} -C {} .".format(save_tar_path, save_dir))
        ##### make gif
        gif_path = os.path.join(self.recon_dir, date_str, time_str+suid,  str(idx) + ".gif")
        imageio.mimsave(gif_path, image_list, 'GIF', duration=0.1)
        result_path_list.append((save_tar_path, gif_path))
    return result_path_list
  
  @torch.no_grad()
  def generate_triplane_mesh(self, latent, batch_size=8):
      print("generate_triplane_mesh...")
      samples = self.model.diffusion_model.generate_from_latent(latent.cuda(), batch=batch_size, class_free_weight=self.class_free_weight)

      # samples = model.diffusion_model.generate_unconditional(num_samples=5)
      
      plane_features = self.model.vae_model.decode(samples)
      mesh_list = self.model.decoder_model.xatlas_uvmap_batch(self.glctx, plane_features)

      return plane_features, mesh_list
  
