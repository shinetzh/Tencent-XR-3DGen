import torch
import time
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
import json

# add paths in model/__init__.py for new models
from models import * 
from models.lr_scheduler.transformer_lr_scheduler import TransformerStayLRScheduler
# from utils import mesh, evaluate
from dataloader.dataset_sdfcolor import get_expand_color, single_points_sampler
from models.utils_model import get_decoder_model, get_diffusion_model, get_vae_model
from models.loss import tvloss, L2Loss
from dataloader.triplane_stats import normalize, unnormalize

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)) + "/triplane_nvdiffrec")
# from triplane_nvdiffrec.get_dmtet_loss import dmtetNetWork

class CombinedModel(pl.LightningModule):
    def __init__(self, specs, args):
        super().__init__()
        self.specs = specs
        self.cur_epoch = 0
        self.args = args
        self.stats_dir = None
        
        if os.path.exists(os.path.join(self.args.exp_dir, "stats")):
            self.stats_dir = os.path.join(self.args.exp_dir, "stats")
            min_values = np.load(f'{self.stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
            max_values = np.load(f'{self.stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
            self._range = max_values - min_values
            self.middle = (min_values + max_values) / 2

        if "warmup" in specs:
            self.warm_up_epoch = specs["warmup"]
        else:
            self.warm_up_epoch = 3
        enable_complie = specs.get("enable_complie", False)
        print("enable_complie: {}".format(enable_complie))

        self.latent_dir = os.path.join(self.args.exp_dir, "modulations" + time.strftime('%Y-%m-%d-%H:%M:%S'))
        self.task = specs['training_task'] # 'combined' or 'modulation' or 'diffusion'
        if self.task in ['combined', 'modulation', "modulation_nosdfgt"]:
            self.decoder_model = SdfModel(specs=specs)
            vae_config = specs["vae_config"]
            self.vae_model = self.get_vae_model(vae_config)
        if self.task in ['combined', 'diffusion']:
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]) 


        if "decoder_config" in specs:
            self.decoder_model = get_decoder_model(specs)
            if enable_complie:
                self.decoder_model = torch.compile(self.decoder_model)
            if specs["decoder_config"]["decoder_type"] == "dmtet":
                with open(specs["decoder_config"]["config_json"], 'r') as fr:
                    self.dmtet_flags = edict(json.load(fr))

        if "vae_config" in specs:
            self.vae_model = get_vae_model(specs)
            if enable_complie:
                self.vae_model = torch.compile(self.vae_model)
            load_from_pretrain = specs["vae_config"].get("load_from_pretrain", None)
            if load_from_pretrain:
                print("load vae from: {}".format(specs["vae_config"]["load_from_pretrain"]))
                ckpt_state_dict = torch.load(specs["vae_config"]["load_from_pretrain"], map_location='cpu')["state_dict"]
                ckpt_state_dict_checked = {}
                for key, value in ckpt_state_dict.items():
                    key = key.replace("dmtetnet", "decoder_model")
                    ckpt_state_dict_checked[key] = value
                self.load_state_dict(ckpt_state_dict_checked, strict=True)

        if "diffusion_config" in specs:
            self.diffusion_model = get_diffusion_model(specs)
            # print(self.diffusion_model)
            load_from_pretrain = specs["diffusion_config"].get("load_from_pretrain", None)
            if load_from_pretrain:
                print("load diffusion from: {}".format(specs["diffusion_config"]["load_from_pretrain"]))
                ckpt_state_dict = torch.load(specs["diffusion_config"]["load_from_pretrain"], map_location='cpu')["state_dict"]
                ckpt_diffusion_state_dict = {}
                for key, value in ckpt_state_dict.items():
                    if "diffusion_model" in key:
                        diffusion_key = '.'.join(key.split(".")[1:])
                        ckpt_diffusion_state_dict[diffusion_key] = value
                self.diffusion_model.load_state_dict(ckpt_diffusion_state_dict, strict=True)

        self.ctx = {}  ## needed by dmtet decoder
        if "loss_config" in specs:
            self.loss_config = edict(specs["loss_config"])

        self.load_from_pratrain = load_from_pretrain

    def training_step(self, x, idx):
        if self.current_epoch > self.cur_epoch:
            self.cur_epoch = self.current_epoch
            torch.cuda.empty_cache()
        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == "combined_sdfyh_diffusion":
            return self.train_combined_vaeyh_diffusion(x)
        elif self.task == "combined_dmtet_diffusion":
            return self.train_combined_dmtet_diffusion(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        elif self.task == "modulation_nosdfgt":
            return self.train_modulation_nosdfgt(x)
        elif self.task == "vae":
            return self.train_vae(x)
        elif self.task == "vae_sdf":
            return self.train_vae_sdf(x)
        elif self.task == "vae_sdfyh":
            return self.train_vae_sdfyh(x)
        elif self.task == "vae_sdfcolor":
            return self.train_vae_sdfcolor(x)
        elif self.task == "vae_sdfgeo":
            return self.train_vae_sdfgeo(x)
        elif self.task == "vae_sdfcolor_gen_latent":
            return self.gen_latent_vae_sdfcolor(x)
        elif self.task == "vae_sdfcolor_L1L2":
            return self.train_vae_sdfcolor_L1L2(x)
        elif self.task == "vae_sdfyh_color":
            return self.train_vae_sdfyh_color(x)
        elif self.task == "vae_sdfyh_coloronly":
            return self.train_vae_sdfyh_coloronly(x)
        elif self.task == "vae_dmtet":
            return self.train_vae_dmtet(x)
        elif self.task == "vae_sdfyh_spacial_rollout":
            return self.train_vae_sdfyh(x)
        elif self.task == "vae_sdf_triplane":
            optimizer, optimizer_triplane = self.optimizers()
            optimizer.zero_grad()
            optimizer_triplane.zero_grad()
            loss, filenames = self.train_vae_sdf_triplane(x)
            self.manual_backward(loss)
            optimizer.step()
            optimizer_triplane.step()
            self.save_triplane(self.triplane.data, filenames, triplane_name="triplane.pt")
        elif self.task == "sdf_triplane":
            optimizer, optimizer_triplane = self.optimizers()
            optimizer.zero_grad()
            optimizer_triplane.zero_grad()
            loss, filenames = self.train_sdf_triplane(x)
            self.manual_backward(loss)
            optimizer.step()
            optimizer_triplane.step()
            self.save_triplane(self.triplane.data, filenames, triplane_name="triplane.pt")
        elif self.task == 'diffusion_image_cond':
            return self.train_diffusion_image_cond(x)
        elif self.task == 'diffusion_image_cond_cat':
            return self.train_diffusion_image_cond(x)
        elif self.task == 'diffusion_image_cond_spacial':
            return self.train_diffusion_image_cond_spacial(x)
        elif self.task == 'diffusion_uncond':
            return self.train_diffusion_unconditioned(x)
        elif self.task == 'diffusion_text_cond':
            return self.train_diffusion_text_cond(x)
        elif self.task == 'diffusion_partial_cond':
            return self.train_diffusion_partial_cond(x)
        elif self.task == 'diffusion_pix3d_cond':
            return self.train_diffusion_pix3d_cond(x)
        elif self.task == 'one_diffusion_uncond':
            return self.train_one_stage_diffusion_unconditioned(x)
        elif self.task == 'one_diffusion_plane_uncond':
            return self.train_one_stage_plane_diffusion_unconditioned(x)
        elif self.task == 'svr_sdfcolor':
            return self.train_svr_sdfcolor(x)
        else:
            NotImplementedError


    def save_triplane(self, triplane, filenames, triplane_name="triplane.pt"):
        for plane, filedir in zip(triplane, filenames):
            save_path = os.path.join(filedir, triplane_name)
            torch.save(plane.cpu(), save_path)


    def configure_optimizers(self):
        if self.task == 'combined':
            params_list = [
                    { 'params': list(self.decoder_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['lr_init'] },
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == "combined_sdfyh_diffusion":
            params_list = [
                          {'params': self.vae_model.parameters(), 'lr':self.specs['lr_init']},
                          { 'params': self.diffusion_model.parameters(), 'lr':self.specs['lr_init'] }
                          ]
        elif self.task == "combined_dmtet_diffusion":
            params_list = [
                          {'params': self.vae_model.parameters(), 'lr':self.specs['lr_init']},
                          { 'params': self.diffusion_model.parameters(), 'lr':self.specs['lr_init'] }
                          ]
        elif self.task == 'modulation':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task in ['modulation_nosdfgt', 'vae_sdf']:
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == 'diffusion':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task in ['diffusion_image_cond',  'diffusion_text_cond', "diffusion_pix3d_cond", "diffusion_partial_cond"]:
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == 'diffusion_image_cond_cat':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task in ['diffusion_uncond']:
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == 'diffusion_image_cond_spacial':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == 'one_diffusion_uncond':
            params_list = [
                    { 'params':self.diffusion_model.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == 'one_diffusion_plane_uncond':
            params_list = [
                    { 'params':self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == 'vae':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                ]
        elif self.task == "vae_sdf_triplane":
            params_list = [
                    { 'params': list(self.decoder_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['lr_init'] }
                ]
            params_triplane = [{'params': self.triplane}]
            optimizer = torch.optim.Adam(params_list)
            optimizer_triplane = torch.optim.SGD(params_triplane, lr=self.specs['triplane_lr'])
            return [optimizer, optimizer_triplane]
        elif self.task == "sdf_triplane":
            params_list = [
                    { 'params': list(self.decoder_model.parameters()), 'lr':self.specs['lr_init'] }
                ]
            params_triplane = [{'params': self.triplane}]
            optimizer = torch.optim.Adam(params_list)
            optimizer_triplane = torch.optim.SGD(params_triplane, lr=self.specs['triplane_lr'])
            return [optimizer, optimizer_triplane]

        elif self.task in [ "vae_dmtet"]:
            if "train_params" in self.specs.keys() and self.specs["train_params"] == "all":
              print("train vae and dmtet.geom_tex_mlp\n")
              params_list = [
                      { 'params': self.vae_model.parameters(), 'lr':self.specs['lr_init'] },
                      { 'params': self.decoder_model.geom_tex_mlp.parameters(), 'lr':self.specs['lr_init'] * 0.01 }
                  ]
            elif "train_params" in self.specs.keys() and self.specs["train_params"] == "vae":
              params_list = [
                      { 'params': self.vae_model.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            else:
                exit("train_params are not specified")
        elif self.task in ["vae_sdfcolor_L1L2", "vae_sdfcolor", "vae_sdfgeo", "vae_sdfyh", "vae_sdfyh_spacial_rollout", "vae_sdfyh_color", "vae_sdfyh_coloronly", 'vae_sdfcolor_gen_latent']:
            if "train_params" in self.specs.keys() and self.specs["train_params"] == "all":
              params_list = [
                      { 'params': self.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            elif "train_params" in self.specs.keys() and self.specs["train_params"] == "vae":
              params_list = [
                      { 'params': self.vae_model.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            else:
                exit("train_params are not specified")
        elif self.task == 'svr_sdfcolor':
            params_list = [
                      { 'params': self.vae_model.parameters(), 'lr':self.specs['lr_init'] }
                  ]
            
        else:
            raise NotImplementedError
        optimizer = torch.optim.Adam(params_list)

        lr_scheduler = TransformerStayLRScheduler(optimizer = optimizer,
                    init_lr=self.specs['lr_init'],
                    peak_lr=self.specs['lr'],
                    final_lr=self.specs['final_lr'],
                    final_lr_scale=self.specs['final_lr_scale'],
                    warmup_steps=self.specs['warmup_steps'],
                    stay_steps=self.specs['stay_steps'],
                    decay_steps=self.specs['decay_steps'])


        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "total_loss",
                    "frequency": 1,
                    "interval": "epoch"
                }
              }


    #-----------different training steps for sdf modulation, diffusion, combined----------
    def train_sdf_triplane(self, data):
        _, psdxyz, points, masks, normal, psdsdfs_gt, plane_features = [x.cuda() for x in data[:-1]]
        filenames = data[-1]
        self.triplane.data = plane_features
        #### STEP 1: obtain reconstructed plane feature and latent code 
        # out = self.vae_model(self.triplane) # out = [self.decode(z), input, mu, log_var, z]
        # reconstructed_plane_feature, latent = out[0], out[-1]

        reconstructed_plane_feature = self.triplane
        # STEP 2: pass recon back to GenSDF pipeline 
        psdsdf_pred = self.decoder_model(reconstructed_plane_feature, psdxyz)
        points_sdf = self.decoder_model(reconstructed_plane_feature, points)
        surface_sdf = points_sdf[masks]
        
        points_norm_pred = self.gradient_tri(self.decoder_model, reconstructed_plane_feature, points)
        surface_norm_pred = points_norm_pred[masks]
        empty_norm_pred = points_norm_pred[~masks]
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            # vae_loss = self.vae_model.loss_function(*out)
            vae_loss = 0
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        eikonal_loss = ((empty_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        surface_sdf_loss = (surface_sdf.abs()).mean()
        normals_loss = ((surface_norm_pred - normal[masks]).abs()).norm(2, dim=-1).mean() 
        psd_loss = (psdsdf_pred.squeeze().abs() - psdsdfs_gt.abs()).abs().mean()
        sdf_loss = eikonal_loss * 0.1 + surface_sdf_loss + normals_loss + psd_loss * 0.5
        # sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        # sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss * 0.5 + vae_loss

        loss_dict =  {"loss": loss, "eikonal_loss": eikonal_loss, "surface_sdf_loss": surface_sdf_loss, "normals_loss": normals_loss, "psd_loss": psd_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

        return loss, filenames


    def train_vae_sdf_triplane(self, data):
        _, psdxyz, points, masks, normal, psdsdfs_gt, plane_features = [x.cuda() for x in data[:-1]]
        filenames = data[-1]
        self.triplane.data = plane_features
        #### STEP 1: obtain reconstructed plane feature and latent code 
        out = self.vae_model(self.triplane) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]
        # STEP 2: pass recon back to GenSDF pipeline 
        psdsdf_pred = self.decoder_model(reconstructed_plane_feature, psdxyz)
        points_sdf = self.decoder_model(reconstructed_plane_feature, points)
        surface_sdf = points_sdf[masks]
        
        points_norm_pred = self.gradient_tri(self.decoder_model, reconstructed_plane_feature, points)
        surface_norm_pred = points_norm_pred[masks]
        empty_norm_pred = points_norm_pred[~masks]
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        eikonal_loss = ((empty_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        surface_sdf_loss = (surface_sdf.abs()).mean()
        normals_loss = ((surface_norm_pred - normal[masks]).abs()).norm(2, dim=-1).mean() 
        psd_loss = (psdsdf_pred.squeeze().abs() - psdsdfs_gt.abs()).abs().mean()
        sdf_loss = eikonal_loss * 0.1 + surface_sdf_loss + normals_loss + psd_loss * 0.5
        # sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        # sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss * 0.5 + vae_loss

        loss_dict =  {"loss": loss, "eikonal_loss": eikonal_loss, "surface_sdf_loss": surface_sdf_loss, "normals_loss": normals_loss, "psd_loss": psd_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

        return loss, filenames


    def train_vae_sdfyh_coloronly(self, data):
        image100_path = data["image100_path"]
        class_name = data["class_name"]
        obj_name = data["obj_name"]
        surface_points = data["surface_points"].cuda()
        surface_colors = data["surface_colors"].cuda()
        color_triplane = data["color_triplane"].cuda()

        color_plane_features = color_triplane.clamp(-1.0, 1.0)

        # STEP 1: obtain reconstructed plane feature and latent code 
        out = self.vae_model(color_plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch
        loss_l1 = (reconstructed_plane_feature - color_plane_features).abs().mean()
        if self.current_epoch < 3:
            loss = loss_l1 + vae_loss
            loss_dict =  {"total_loss": loss,
                      "loss_color_surface": 0,
                      "l1": loss_l1,
                      "vae": vae_loss}
            self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
            return loss


        ################################
        #####    color loss
        ################################
        triplane_rgb = reconstructed_plane_feature
        pred_rgb_surface = self.decoder_model(triplane_rgb, surface_points)

        loss_color_surface = (pred_rgb_surface - surface_colors).abs().mean()

        loss = vae_loss + loss_color_surface * 100 + loss_l1

        loss_dict =  {"total_loss": loss,
                      "loss_color_surface": loss_color_surface,
                      "l1": loss_l1,
                      "vae": vae_loss}
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss


    def train_vae_sdfcolor(self, data):

        points_surface = data["surface_points"].cuda()
        normal_surface = data["surface_normals"].cuda()
        color_points = data["color_points"].cuda()
        color_colors = data["color_colors"].cuda()
        color_points_normal = data['color_points_normal'].cuda()
        points_empty = data["sdf_points"].cuda()
        sdf_empty_gt = data["sdf_sdfs"].cuda()
        plane_features = data["triplane"].cuda()

        # STEP 1: obtain reconstructed plane feature and latent code 
        random_points = torch.from_numpy(
                            np.random.uniform(-1.0, 1.0, size=points_empty.shape).astype(np.float32)
                        ).cuda()
        points_all = torch.cat([points_surface, random_points], dim=1)
        points_surface_num = points_surface.shape[1]
        # # STEP 1: obtain reconstructed plane feature and latent code 
        # plane_features = plane_features.clamp(-1.0, 1.0)
        if self.stats_dir is not None:
            plane_features = normalize(plane_features, self.stats_dir, self.middle, self._range)
        else:
            plane_features = plane_features.clamp(-1.0, 1.0)
        ### training
        
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature = out[0]
        
        try:
            loss_vae = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        loss_l1 = (reconstructed_plane_feature - plane_features).abs().mean()

        if (self.load_from_pratrain is None) and self.current_epoch < self.warm_up_epoch:
            loss = loss_l1 * self.loss_config.loss_l1_weight + loss_vae * self.loss_config.loss_vae_weight
            loss_dict =  {"loss_total": loss,
                      "loss_eikonal": 0,
                      "loss_surface_sdf": 0,
                      "loss_normals": 0,
                      "loss_psd_sdf": 0,
                      "loss_color": 0,
                      "loss_l1": loss_l1,
                      "loss_vae": loss_vae,
                      "loss_tvloss": 0}
            self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
            return loss

        if self.stats_dir is not None:
            reconstructed_plane_feature = unnormalize(reconstructed_plane_feature, self.stats_dir, self.middle, self._range)
        ###### sdf loss
        sdf_empty_pred = self.decoder_model.forward_sdf(reconstructed_plane_feature, points_empty)
        surface_sdf = self.decoder_model.forward_sdf(reconstructed_plane_feature, points_surface)
        
        points_norm_pred = self.gradient_tri(self.decoder_model.forward_sdf, reconstructed_plane_feature, points_all)
        surface_norm_pred = points_norm_pred[:, :points_surface_num]


        loss_eikonal = ((points_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        loss_surface_sdf = (surface_sdf.abs()).mean()
        loss_normals = ((surface_norm_pred - normal_surface).abs()).norm(2, dim=-1).mean()
        loss_psd_sdf = (sdf_empty_pred.squeeze().abs() - sdf_empty_gt.abs()).abs().mean()
        sdf_loss = loss_eikonal * self.loss_config.loss_eikonal_weight + \
                    loss_surface_sdf * self.loss_config.loss_surface_sdf_weight +  \
                    loss_normals * self.loss_config.loss_normals_weight +  \
                    loss_psd_sdf * self.loss_config.loss_psd_sdf_weight


        ###### color loss
        expand_points, t = single_points_sampler(color_points, color_points_normal)
        color_points = torch.cat((expand_points, color_points), dim=1)
        expand_colors = get_expand_color(color_colors, t)
        color_colors = torch.cat((expand_colors, color_colors), dim=1)

        pred_rgb_surface = self.decoder_model.forward_rgb(reconstructed_plane_feature, color_points)
        loss_color = (pred_rgb_surface - color_colors).abs().sum(-1).mean()

        # ###### tv loss
        # loss_tv = tvloss(reconstructed_plane_feature)


        loss = sdf_loss + \
                loss_vae * self.loss_config.loss_vae_weight + \
                loss_color * self.loss_config.loss_color_weight
        
        loss_dict =  {"loss_total": loss,
                      "loss_eikonal": loss_eikonal,
                      "loss_surface_sdf": loss_surface_sdf,
                      "loss_normals": loss_normals,
                      "loss_psd_sdf": loss_psd_sdf,
                      "loss_color": loss_color,
                      "loss_l1": loss_l1,
                      "loss_vae": loss_vae}
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss


    def gen_latent_vae_sdfcolor(self, data):

        class_name = data["class_name"]
        obj_name = data["obj_name"]

        plane_features = data["triplane"].cuda()

        # # STEP 1: obtain reconstructed plane feature and latent code 
        if self.stats_dir is not None:
            plane_features = normalize(plane_features, self.stats_dir, self.middle, self._range)
        else:
            plane_features = plane_features.clamp(-1.0, 1.0)
        
        ########## test and save reconstruct all
        self.vae_model.eval()
        with torch.no_grad():
            latent = self.vae_model.get_latent(plane_features, return_dist=True)
            # print(latent.shape)
        ###########################################
        
        # breakpoint()
        class_name = data["class_name"]
        obj_name = data["obj_name"]
        # Here set the latent dir path of your model
        latent_dir = ''

        for i in range(len(obj_name)):
            ####### 2. save latent
            latent_save_dir = os.path.join(latent_dir, class_name[i])
            os.makedirs(latent_save_dir, exist_ok=True)
            latent_save_path = os.path.join(latent_save_dir, obj_name[i] + ".npy")
            np.save(latent_save_path, latent[i].detach().unsqueeze(0).cpu().numpy())

            if os.path.isfile(latent_save_path):
                self.dataset_dict['data'][class_name[i]][obj_name[i]]['latent'] = latent_save_path
            else:
                self.dataset_dict['data'][class_name[i]][obj_name[i]]['latent'] = None

        return None

    def train_vae_sdfcolor_L1L2(self, data):
        class_name = data["class_name"]
        obj_name = data["obj_name"]
        image100_path = data["image100_path"]

        points_surface = data["surface_points"].cuda()
        normal_surface = data["surface_normals"].cuda()
        color_points = data["color_points"].cuda()
        color_colors = data["color_colors"].cuda()
        color_points_normal = data['color_points_normal'].cuda()
        points_empty = data["sdf_points"].cuda()
        sdf_empty_gt = data["sdf_sdfs"].cuda()
        plane_features = data["triplane"].cuda()

        # STEP 1: obtain reconstructed plane feature and latent code 
        points_all = torch.cat([points_surface, points_empty], dim=1)
        points_surface_num = points_surface.shape[1]

        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = plane_features.clamp(-1.0, 1.0)
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        #####  vae loss: l1 + kl loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        loss_l1 = (reconstructed_plane_feature - plane_features).abs().mean()
        loss_l2 = L2Loss(reconstructed_plane_feature, plane_features)

        loss = loss_l1 + loss_l2 + vae_loss
        loss_dict = {"total_loss": loss,
                      "loss_l1": loss_l1,
                      "loss_l2" : loss_l2,
                      "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss


    def train_vae_sdfyh(self, data):
        points_surface = data["surface_points"].cuda()
        normal_surface = data["surface_normals"].cuda()
        points_empty = data["sdf_points"].cuda()
        sdf_empty_gt = data["sdf_sdfs"].cuda()
        plane_features = data["sdf_triplane"].cuda()

        # STEP 1: obtain reconstructed plane feature and latent code 
        points_all = torch.cat([points_surface, points_empty], dim=1)
        points_surface_num = points_surface.shape[1]

        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = plane_features.clamp(-1.0, 1.0)
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]


        # STEP 2: pass recon back to GenSDF pipeline 
        sdf_empty_pred = self.decoder_model(reconstructed_plane_feature, points_empty)
        surface_sdf = self.decoder_model(reconstructed_plane_feature, points_surface)
        
        
        points_norm_pred = self.gradient_tri(self.decoder_model.forward_with_plane_features, reconstructed_plane_feature, points_all)
        surface_norm_pred = points_norm_pred[:, :points_surface_num]
        # empty_norm_pred = points_norm_pred[:, points_surface_num:]
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        eikonal_loss = ((points_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        surface_sdf_loss = (surface_sdf.abs()).mean()
        normals_loss = ((surface_norm_pred - normal_surface).abs()).norm(2, dim=-1).mean()
        psd_sdf_loss = (sdf_empty_pred.squeeze().abs() - sdf_empty_gt.abs()).abs().mean()
        sdf_loss = eikonal_loss * 0.1 + surface_sdf_loss * 1.0 + normals_loss * 10 + psd_sdf_loss * 1.0


        loss = sdf_loss + vae_loss

        loss_dict =  {"total_loss": loss,
                      "eikonal_loss": eikonal_loss,
                      "surface_sdf_loss": surface_sdf_loss,
                      "normals_loss": normals_loss,
                      "psd_loss": psd_sdf_loss,
                      "vae": vae_loss}
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss


    def on_train_epoch_start(self):
        print("start time: {}, experiment: {}".format(time.strftime('%Y-%m-%d-%H:%M:%S'), self.args.exp_dir))
        return
    
    def on_train_epoch_end(self):
        pass


    def train_vae_dmtet(self, data):
        for key, value in data.items():
            if key in ["mv", "mvp", "campos", "img", "triplane"]:
                data[key] = value.cuda()
        data['resolution'] = self.dmtet_flags.train_res
        data['spp'] = self.dmtet_flags.spp

        plane_features = data["triplane"].clamp(-1., 1.)
        
        # STEP 1: obtain reconstructed plane feature and latent code 
        # plane_features = [B, channel * 3, res, res]
        if self.args.infer_latent:
            with torch.no_grad():
                out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        else:
            out = self.vae_model(plane_features)
        reconstructed_plane_feature, latent = out[0], out[-1]
        
        if reconstructed_plane_feature.device not in self.ctx:
            self.ctx[reconstructed_plane_feature.device] = dr.RasterizeCudaContext(device=reconstructed_plane_feature.device)
            print('Created Cuda context for device', reconstructed_plane_feature.device)
        ctx = self.ctx[reconstructed_plane_feature.device]

        # STEP 2: pass recon back to GenSDF pipeline
        
        plane_loss = F.l1_loss(reconstructed_plane_feature.squeeze(), plane_features.squeeze(), reduction='none')
        plane_loss = reduce(plane_loss, 'b ... -> b (...)', 'mean').mean()
        reconstructed_plane_feature = reconstructed_plane_feature.clamp(-1.0, 1.0)


        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        if self.cur_epoch < 3:
            loss = vae_loss + plane_loss * 10
            loss_dict =  {"img_loss": 0,
                        # "head_img_loss": img_loss[1],
                        "reg_loss": 0,
                        "vae": vae_loss,
                        "plane_loss": plane_loss,
                        "total_loss": loss}
            
            self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

            return loss

        try:
            img_loss, reg_loss, buffers = self.decoder_model(reconstructed_plane_feature, data, ctx)
        except:
            print("meshing failed!{}\n".format(data["shaded_dir"]))
            loss = vae_loss * 0.01 + plane_loss * 10
            loss_dict =  {"img_loss": 0,
                        # "head_img_loss": img_loss[1],
                        "reg_loss": 0,
                        "vae": vae_loss,
                        "plane_loss": plane_loss,
                        "total_loss": loss}
        
            self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

            return loss

        ## dmtet loss

        loss = vae_loss + plane_loss * 0.5 + img_loss[0] * 100 + reg_loss
        # loss = vae_loss + plane_loss + img_loss[0] * 10 + reg_loss + img_loss[1] * 20

        loss_dict =  {"img_loss": img_loss[0],
                      # "head_img_loss": img_loss[1],
                      "reg_loss": reg_loss,
                      "vae": vae_loss,
                      "plane_loss": plane_loss,
                      "total_loss": loss}
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

        return loss


    def train_vae_sdf(self, data):
        _, psdxyz, points, masks, normal, psdsdfs_gt, plane_features = [x.cuda() for x in data]
        plane_features = torch.concat([plane_features[:, 0, :, :, :],
                                       plane_features[:, 1, :, :, :],
                                       plane_features[:, 2, :, :, :]], dim=1)
        # STEP 1: obtain reconstructed plane feature and latent code 
        # plane_features = [B, channel * 3, res, res]
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]
        
        # STEP 2: pass recon back to GenSDF pipeline
        psdsdf_pred = self.decoder_model(reconstructed_plane_feature, psdxyz)
        points_sdf = self.decoder_model(reconstructed_plane_feature, points)
        surface_sdf = points_sdf[masks]
        
        points_norm_pred = self.gradient_tri(self.decoder_model, reconstructed_plane_feature, points)
        surface_norm_pred = points_norm_pred[masks]
        empty_norm_pred = points_norm_pred[~masks]
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        eikonal_loss = ((empty_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        surface_sdf_loss = (surface_sdf.abs()).mean()
        normals_loss = ((surface_norm_pred - normal[masks]).abs()).norm(2, dim=-1).mean() 
        psd_loss = (psdsdf_pred.squeeze().abs() - psdsdfs_gt.abs()).abs().mean()
        sdf_loss = eikonal_loss * 0.1 + surface_sdf_loss + normals_loss + psd_loss * 0.5
        # sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        # sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss * 0.5 + vae_loss

        loss_dict =  {"eikonal_loss": eikonal_loss, "surface_sdf_loss": surface_sdf_loss, "normals_loss": normals_loss, "psd_loss": psd_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

        return loss


    def train_vae(self, data):
        out = self.vae_model(data) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        plane_loss = F.l1_loss(reconstructed_plane_feature.squeeze(), data.squeeze(), reduction='none')
        plane_loss = reduce(plane_loss, 'b ... -> b (...)', 'mean').mean()

        loss = plane_loss + vae_loss

        loss_dict =  {"plane_loss": plane_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

        return loss


    def gradient_tri(self, net, plane_feature, x):
        b = x.shape[0]
        eps = 1e-6
        x_nei = torch.stack([
            x + torch.as_tensor([[eps, 0.0, 0.0]]).to(x),
            x + torch.as_tensor([[-eps, 0.0, 0.0]]).to(x),
            x + torch.as_tensor([[0.0, eps, 0.0]]).to(x),
            x + torch.as_tensor([[0.0, -eps, 0.0]]).to(x),
            x + torch.as_tensor([[0.0, 0.0, eps]]).to(x),
            x + torch.as_tensor([[0.0, 0.0, -eps]]).to(x)
        ], dim=1).view(b, -1, 3)
        sdf_nei = net(plane_feature, x_nei)
        sdf_nei = sdf_nei.view(b, 6, -1, 1)
        gradients = torch.cat([
            0.5 * (sdf_nei[:, 0] - sdf_nei[:, 1]) / eps, 
            0.5 * (sdf_nei[:, 2] - sdf_nei[:, 3]) / eps,
            0.5 * (sdf_nei[:, 4] - sdf_nei[:, 5]) / eps,
        ], dim=-1)
        return gradients


    def train_modulation_nosdfgt(self, data):
        [points_surface, normal_surface, points_empty, sdf_empty_gt] = [x.cuda() for x in data]
        points_all = torch.cat([points_surface, points_empty], dim=1)
        points_surface_num = points_surface.shape[1]

        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = self.decoder_model.pointnet.get_plane_features(points_surface)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]
        
        # STEP 2: pass recon back to GenSDF pipeline 
        surface_sdf = self.decoder_model.forward_with_plane_features(reconstructed_plane_feature, points_surface)
        sdf_empty_pred = self.decoder_model.forward_with_plane_features(reconstructed_plane_feature, points_empty)
        
        points_norm_pred = self.gradient_tri(self.decoder_model.forward_with_plane_features, reconstructed_plane_feature, points_all)
        surface_norm_pred = points_norm_pred[:, :points_surface_num]
        # empty_norm_pred = points_norm_pred[:, points_surface_num:]
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        eikonal_loss = ((points_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        surface_sdf_loss = (surface_sdf.abs()).mean()
        normals_loss = ((surface_norm_pred - normal_surface).abs()).norm(2, dim=-1).mean()
        psd_sdf_loss = (sdf_empty_pred.squeeze().abs() - sdf_empty_gt.abs()).abs().mean()
        # sdf_loss =surface_sdf_loss * 1 + normals_loss * 0.1 + psd_sdf_loss * 1
        sdf_loss = eikonal_loss * 0.1 + surface_sdf_loss * 1 + normals_loss * 1 + psd_sdf_loss * 1
        #### sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        #### sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        # surface_sdf_loss = (surface_sdf.abs()).mean()
        # psd_sdf_loss = (sdf_empty_pred.squeeze() - sdf_empty_gt).abs().mean()
        # sdf_loss = surface_sdf_loss + psd_sdf_loss

        loss = sdf_loss + vae_loss

        loss_dict =  {"total_loss": loss,
                      "eikonal_loss": eikonal_loss,
                      "surface_sdf_loss": surface_sdf_loss,
                      "normals_loss": normals_loss,
                      "psd_loss": psd_sdf_loss,
                      "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)

        return loss


    def train_modulation(self, x):

        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = self.decoder_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.decoder_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        loss_dict =  {"sdf": sdf_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

    def train_diffusion_image_cond(self, x):

        self.train()
        latent_image = x["latent_image"].cuda()
        latent_modulation = x["latent_modulation"].cuda()
        # unconditional training if cond is None 
        cond = latent_image

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent_modulation, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    def train_diffusion_text_cond(self, x):

        self.train()
        latent_text = x["text_feature"].cuda()
        latent_modulation = x["modulation"].cuda()
        # unconditional training if cond is None 
        cond = latent_text

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent_modulation, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    def train_diffusion_partial_cond(self, x):

        self.train()
        latent_text = x["text_feature"].cuda()
        latent_modulation = x["modulation"].cuda()
        # unconditional training if cond is None 
        cond = latent_text

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent_modulation, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    def train_diffusion_pix3d_cond(self, x):

        self.train()
        latent_text = x["latent_image"].cuda()
        latent_modulation = x["latent_modulation"].cuda()
        # unconditional training if cond is None 
        cond = latent_text

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent_modulation, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    def train_diffusion_image_cond_spacial(self, x):
        self.train()
        latent_image = x["latent_image"].cuda()
        latent_modulation = x["latent_modulation"].cuda()[:, :4, :64, :64]
        # unconditional training if cond is None 
        cond = latent_image

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        loss, loss_dict = self.diffusion_model.diffusion_model_from_latent(latent_modulation, c=cond)

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss

    def train_diffusion_unconditioned(self, x):
        self.train()
        latent_modulation = x["modulation"]

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent_modulation, cond=None)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss

    def train_diffusion(self, x):

        self.train()

        pc = x['point_cloud'] # (B, 1024, 3) or False if unconditional 
        latent = x['latent'] # (B, D)

        # unconditional training if cond is None 
        cond = pc if self.specs['diffusion_model_specs']['cond'] else None 

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)

        loss_dict =  {
                        "total": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return diff_loss


    def train_combined_dmtet_diffusion(self, data):
        for key, value in data.items():
            if key in ["mv", "mvp", "campos", "img", "triplane",  "image_latent"]:
                data[key] = value.cuda()
        data['resolution'] = self.dmtet_flags.train_res
        data['spp'] = self.dmtet_flags.spp

        #######################
        ######   stage1
        #######################
        plane_features = data["triplane"].clamp(-1., 1.)
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]


        ## dmtet loss
        if reconstructed_plane_feature.device not in self.ctx:
            self.ctx[reconstructed_plane_feature.device] = dr.RasterizeCudaContext(device=reconstructed_plane_feature.device)
            print('Created Cuda context for device', reconstructed_plane_feature.device)
        ctx = self.ctx[reconstructed_plane_feature.device]

        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch
        try:
          img_loss, reg_loss, buffers = self.decoder_model(reconstructed_plane_feature, data, ctx)
          loss_stage1 = vae_loss + img_loss[0] + reg_loss + img_loss[1] * 10

          loss_dict =  {
                        "img_loss": img_loss[0],
                        "head_img_loss": img_loss[1],
                        "reg_loss": reg_loss,
                        "vae": vae_loss,
                        "plane_loss": 0}
        except:
          print("meshing failed!\n")
          plane_loss = F.l1_loss(reconstructed_plane_feature.squeeze(), plane_features.squeeze(), reduction='none')
          plane_loss = reduce(plane_loss, 'b ... -> b (...)', 'mean').mean()
          loss = vae_loss * 0.01 + plane_loss * 10
          loss_dict =  {
                        "vae": vae_loss,
                        "plane_loss": plane_loss,
                        "total_loss": loss}
          return loss

        ###########################
        ##### stage3
        ###########################
        triplane_latent = latent
        image_latent = data["image_latent"]

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent = self.diffusion_model.diffusion_model_from_latent(triplane_latent, cond=image_latent)

        # STEP 5: use predicted / reconstructed latent to run SDF loss
        generated_plane_feature = self.vae_model.decode(pred_latent)
        
        try:
          img_loss2, reg_loss2, buffers2 = self.decoder_model(generated_plane_feature, data, ctx)
          loss_stage3 = img_loss2[0] + reg_loss2 + img_loss2[1] * 10

          loss_dict.update({"img_loss2": img_loss2[0],
                          "head_img_loss2": img_loss2[1],
                          "reg_loss2": reg_loss2})
        except:
          print("diffusion out meshing failed!\n")
          return loss_stage1 + diff_loss
        
        # combined all loss
        loss = loss_stage1 + diff_loss + loss_stage3

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss


    def train_combined_vaeyh_diffusion(self, data):
        _, psdxyz, points, masks, normal, psdsdfs_gt, plane_features, latent_image = [x.cuda() for x in data]
        plane_features = torch.concat([plane_features[:, 0, :, :, :],
                                       plane_features[:, 1, :, :, :],
                                       plane_features[:, 2, :, :, :]], dim=1)
        # STEP 1: obtain reconstructed plane feature and latent code 
        # plane_features = [B, channel * 3, res, res]
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]
        
        # STEP 2: pass recon back to GenSDF pipeline

        self.decoder_model.eval()
        reconstructed_plane_feature = (reconstructed_plane_feature / 2 + 0.5) * 1 + (-0.5)
        psdsdf_pred = self.decoder_model(reconstructed_plane_feature, psdxyz)
        points_sdf = self.decoder_model(reconstructed_plane_feature, points)

        surface_sdf = points_sdf[masks]
        
        points_norm_pred = self.gradient_tri(self.decoder_model, reconstructed_plane_feature, points)
        surface_norm_pred = points_norm_pred[masks]
        empty_norm_pred = points_norm_pred[~masks]

        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        eikonal_loss = ((empty_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        surface_sdf_loss = (surface_sdf.abs()).mean()
        normals_loss = ((surface_norm_pred - normal[masks]).abs()).norm(2, dim=-1).mean() 
        psd_loss = (psdsdf_pred.squeeze().abs() - psdsdfs_gt.abs()).abs().mean()
        sdf_loss = eikonal_loss * 0.1 + surface_sdf_loss + normals_loss + psd_loss * 0.5

        plane_loss = F.l1_loss(reconstructed_plane_feature.squeeze(), plane_features.squeeze(), reduction='none')
        plane_loss = reduce(plane_loss, 'b ... -> b (...)', 'mean').mean()

        loss_stage1 = sdf_loss * 0.5 + vae_loss + plane_loss * 0.5

        loss_dict =  {"loss_stage1": loss_stage1,
                    "plane_loss": plane_loss,
                    "eikonal_loss": eikonal_loss,
                    "surface_sdf_loss": surface_sdf_loss,
                    "normals_loss": normals_loss,
                    "psd_loss": psd_loss,
                    "vae": vae_loss}

        latent_modulation = latent
        cond = latent_image

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent = self.diffusion_model.diffusion_model_from_latent(latent_modulation, cond=cond)

        # STEP 5: use predicted / reconstructed latent to run SDF loss
        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_plane_feature = (generated_plane_feature / 2 + 0.5) * 1 + (-0.5)
        psdsdf_pred_generated = self.decoder_model(generated_plane_feature, psdxyz)
        points_sdf_generated = self.decoder_model(generated_plane_feature, points)

        surface_sdf_generated = points_sdf_generated[masks]
        
        points_norm_pred_generated = self.gradient_tri(self.decoder_model, generated_plane_feature, points)
        surface_norm_pred_generated = points_norm_pred_generated[masks]
        empty_norm_pred_generated = points_norm_pred_generated[~masks]

        eikonal_loss_generated = ((empty_norm_pred_generated.norm(2, dim=-1) - 1) ** 2).mean()
        surface_sdf_loss_generated = (surface_sdf_generated.abs()).mean()
        normals_loss_generated = ((surface_norm_pred_generated - normal[masks]).abs()).norm(2, dim=-1).mean() 
        psd_loss_generated = (psdsdf_pred_generated.squeeze().abs() - psdsdfs_gt.abs()).abs().mean()
        sdf_loss_generated = eikonal_loss_generated * 0.1 + surface_sdf_loss_generated + normals_loss_generated + psd_loss_generated * 0.5

        # combined all loss
        loss = loss_stage1 + diff_loss + sdf_loss_generated

        loss_stage2_dict =  {
                        "loss_all": loss,
                        "diff_loss": diff_loss,
                        "sdf_loss_generated": sdf_loss_generated,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }
        loss_dict.update(loss_stage2_dict)
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss



    def train_one_stage_diffusion_unconditioned(self, data):
        # class_name = data["class_name"]
        # obj_name = data["obj_name"]
        # image100_path = data["image100_path"]

        points_surface = data["surface_points"].cuda()
        normal_surface = data["surface_normals"].cuda()
        color_points = data["color_points"].cuda()
        color_colors = data["color_colors"].cuda()
        color_points_normal = data['color_points_normal'].cuda()
        points_empty = data["sdf_points"].cuda()
        sdf_empty_gt = data["sdf_sdfs"].cuda()
        plane_features = data["triplane"].cuda()

        # STEP 1: obtain reconstructed plane feature and latent code 
        random_points = torch.from_numpy(
                            np.random.uniform(-1.0, 1.0, size=points_empty.shape).astype(np.float32)
                        ).cuda()
        points_all = torch.cat([points_surface, random_points], dim=1)
        points_surface_num = points_surface.shape[1]
        # # STEP 1: obtain reconstructed plane feature and latent code 
        # plane_features = plane_features.clamp(-1.0, 1.0)
        if self.stats_dir is not None:
            plane_features = normalize(plane_features, self.stats_dir, self.middle, self._range)
        else:
            plane_features = plane_features.clamp(-1.0, 1.0)
        ### training
        
        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, reconstructed_plane_feature, t = self.diffusion_model.diffusion_model_from_latent(plane_features, cond=None)

        if self.stats_dir is not None:
            reconstructed_plane_feature = unnormalize(reconstructed_plane_feature, self.stats_dir, self.middle, self._range)

        # compute loss only for t < 100
        # if torch.sum(t<100) > 0:
        threshold = 100
        if torch.sum(t<threshold) > 0:
            reconstructed_plane_feature = reconstructed_plane_feature[t<threshold]
            points_empty = points_empty[t<threshold]
            points_surface = points_surface[t<threshold]
            points_all = points_all[t<threshold]
            normal_surface = normal_surface[t<threshold]
            color_points = color_points[t<threshold]
            color_points_normal = color_points_normal[t<threshold]
            sdf_empty_gt = sdf_empty_gt[t<threshold]
            color_colors = color_colors[t<threshold]

            
            ###### sdf loss
            sdf_empty_pred = self.decoder_model.forward_sdf(reconstructed_plane_feature, points_empty)
            surface_sdf = self.decoder_model.forward_sdf(reconstructed_plane_feature, points_surface)
            
            points_norm_pred = self.gradient_tri(self.decoder_model.forward_sdf, reconstructed_plane_feature, points_all)
            surface_norm_pred = points_norm_pred[:, :points_surface_num]

            loss_eikonal = ((points_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
            loss_surface_sdf = (surface_sdf.abs()).mean()
            loss_normals = ((surface_norm_pred - normal_surface).abs()).norm(2, dim=-1).mean()
            loss_psd_sdf = (sdf_empty_pred.squeeze().abs() - sdf_empty_gt.squeeze().abs()).abs().mean()
            sdf_loss = loss_eikonal * self.loss_config.loss_eikonal_weight + \
                        loss_surface_sdf * self.loss_config.loss_surface_sdf_weight +  \
                        loss_normals * self.loss_config.loss_normals_weight +  \
                        loss_psd_sdf * self.loss_config.loss_psd_sdf_weight

            ###### color loss
            expand_points, interval = single_points_sampler(color_points, color_points_normal)
            color_points = torch.cat((expand_points, color_points), dim=1)
            expand_colors = get_expand_color(color_colors, interval)
            
            color_colors = torch.cat((expand_colors, color_colors), dim=1)

            pred_rgb_surface = self.decoder_model.forward_rgb(reconstructed_plane_feature, color_points)
            loss_color = (pred_rgb_surface - color_colors).abs().sum(-1).mean()

            loss = sdf_loss + \
                    diff_loss * self.loss_config.loss_diff_weight + \
                    loss_color * self.loss_config.loss_color_weight
            
            loss_dict =  {"loss_total": loss,
                        "loss_eikonal": loss_eikonal,
                        "loss_surface_sdf": loss_surface_sdf,
                        "loss_normals": loss_normals,
                        "loss_psd_sdf": loss_psd_sdf,
                        "loss_color": loss_color,
                        "loss_diff": diff_loss,
                        "loss_diff100": diff_100_loss,
                        "loss_diff1000": diff_1000_loss}
        else:
            loss = diff_loss * self.loss_config.loss_diff_weight
            loss_dict =  {"loss_total": loss,
                        "loss_eikonal": 0,
                        "loss_surface_sdf": 0,
                        "loss_normals": 0,
                        "loss_psd_sdf": 0,
                        "loss_color": 0,
                        "loss_diff": diff_loss,
                        "loss_diff100": diff_100_loss,
                        "loss_diff1000": diff_1000_loss}

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss
    
    def train_one_stage_plane_diffusion_unconditioned(self, data):
        # class_name = data["class_name"]
        # obj_name = data["obj_name"]
        # image100_path = data["image100_path"]

        plane_features = data["triplane"].cuda()


        if self.stats_dir is not None:
            plane_features = normalize(plane_features, self.stats_dir, self.middle, self._range)
        else:
            plane_features = plane_features.clamp(-1.0, 1.0)
        
        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, reconstructed_plane_feature, t = self.diffusion_model.diffusion_model_from_latent(plane_features, cond=None)

        loss_dict =  {"loss_diff": diff_loss,
                    "loss_diff100": diff_100_loss,
                    "loss_diff1000": diff_1000_loss}

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return diff_loss
    
    # the first half is the same as "train_sdf_modulation"
    # the reconstructed latent is used as input to the diffusion model, rather than loading latents from the dataloader as in "train_diffusion"
    def train_combined(self, x):
        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature for SDF and latent code for diffusion
        plane_features = self.decoder_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        #print("plane feat shape: ", feat.shape)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1] # [B, D*3, resolution, resolution], [B, D*3]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.decoder_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF 
        try:
            vae_loss = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        # STEP 4: use latent as input to diffusion model
        cond = pc if self.specs['diffusion_model_specs']['cond'] else None
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)
        
        # STEP 5: use predicted / reconstructed latent to run SDF loss 
        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_sdf_pred = self.decoder_model.forward_with_plane_features(generated_plane_feature, xyz)
        generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

        # we did not experiment with using constants/weights for each loss (VAE loss is weighted using value in specs file)
        # results could potentially improve with a grid search 
        loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss

        loss_dict =  {
                        "total": loss,
                        "sdf": sdf_loss,
                        "vae": vae_loss,
                        "diff": diff_loss,
                        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
                        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
                        # visualizing loss curves can help with debugging if training is unstable
                        #"diff100": diff_100_loss, # note that this can sometimes appear as nan when the training batch does not have sampled timesteps < 100
                        #"diff1000": diff_1000_loss,
                        "gensdf": generated_sdf_loss,
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss
    
    
    def train_svr_sdfcolor(self, data):
        # class_name = data["class_name"]
        # obj_name = data["obj_name"]
        # image100_path = data["image100_path"]
        image = data['image'].cuda()
        points_surface = data["surface_points"].cuda()
        normal_surface = data["surface_normals"].cuda()
        color_points = data["color_points"].cuda()
        color_colors = data["color_colors"].cuda()
        color_points_normal = data['color_points_normal'].cuda()
        points_empty = data["sdf_points"].cuda()
        sdf_empty_gt = data["sdf_sdfs"].cuda()

        # STEP 1: obtain reconstructed plane feature and latent code 
        random_points = torch.from_numpy(
                            np.random.uniform(-1.0, 1.0, size=points_empty.shape).astype(np.float32)
                        ).cuda()
        points_all = torch.cat([points_surface, random_points], dim=1)
        points_surface_num = points_surface.shape[1]
        # # STEP 1: obtain predicted plane feature and latent code 
        ### training
        
        predicted_planes, posterior = self.vae_model(image) # out = [self.decode(z), input, mu, log_var, z]

        #####  vae loss: kl loss
        try:
            loss_vae = self.vae_model.loss_function(posterior)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        ###### sdf loss
        sdf_empty_pred = self.decoder_model.forward_sdf(predicted_planes, points_empty)
        surface_sdf = self.decoder_model.forward_sdf(predicted_planes, points_surface)
        
        points_norm_pred = self.gradient_tri(self.decoder_model.forward_sdf, predicted_planes, points_all)
        surface_norm_pred = points_norm_pred[:, :points_surface_num]


        loss_eikonal = ((points_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        loss_surface_sdf = (surface_sdf.abs()).mean()
        loss_normals = ((surface_norm_pred - normal_surface).abs()).norm(2, dim=-1).mean()
        loss_psd_sdf = (sdf_empty_pred.squeeze().abs() - sdf_empty_gt.abs()).abs().mean()
        sdf_loss = loss_eikonal * self.loss_config.loss_eikonal_weight + \
                    loss_surface_sdf * self.loss_config.loss_surface_sdf_weight +  \
                    loss_normals * self.loss_config.loss_normals_weight +  \
                    loss_psd_sdf * self.loss_config.loss_psd_sdf_weight


        ###### color loss
        expand_points, t = single_points_sampler(color_points, color_points_normal)
        color_points = torch.cat((expand_points, color_points), dim=1)
        expand_colors = get_expand_color(color_colors, t)
        color_colors = torch.cat((expand_colors, color_colors), dim=1)

        pred_rgb_surface = self.decoder_model.forward_rgb(predicted_planes, color_points)
        loss_color = (pred_rgb_surface - color_colors).abs().sum(-1).mean()

        # ###### tv loss
        # loss_tv = tvloss(predicted_planes)


        loss = sdf_loss + \
                loss_vae * self.loss_config.loss_vae_weight + \
                loss_color * self.loss_config.loss_color_weight
        
        loss_dict =  {"loss_total": loss,
                      "loss_eikonal": loss_eikonal,
                      "loss_surface_sdf": loss_surface_sdf,
                      "loss_normals": loss_normals,
                      "loss_psd_sdf": loss_psd_sdf,
                      "loss_color": loss_color,
                      "loss_vae": loss_vae}
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss


    def train_vae_sdfgeo(self, data):
        # class_name = data["class_name"]
        # obj_name = data["obj_name"]
        # image100_path = data["image100_path"]

        points_surface = data["surface_points"].cuda()
        normal_surface = data["surface_normals"].cuda()
        points_empty = data["sdf_points"].cuda()
        sdf_empty_gt = data["sdf_sdfs"].cuda()
        plane_features = data["triplane"].cuda()

        # STEP 1: obtain reconstructed plane feature and latent code 
        random_points = torch.from_numpy(
                            np.random.uniform(-1.0, 1.0, size=points_empty.shape).astype(np.float32)
                        ).cuda()
        points_all = torch.cat([points_surface, random_points], dim=1)
        points_surface_num = points_surface.shape[1]
        # # STEP 1: obtain reconstructed plane feature and latent code 
        # plane_features = plane_features.clamp(-1.0, 1.0)
        if self.stats_dir is not None:
            plane_features = normalize(plane_features, self.stats_dir, self.middle, self._range)
        else:
            plane_features = plane_features.clamp(-1.0, 1.0)
        ### training
        
        out = self.vae_model(plane_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature = out[0]


        #####  vae loss: l1 + kl loss
        try:
            loss_vae = self.vae_model.loss_function(*out)
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        loss_l1 = (reconstructed_plane_feature - plane_features).abs().mean()

        if (self.load_from_pratrain is None) and self.current_epoch < self.warm_up_epoch:
            loss = loss_l1 * self.loss_config.loss_l1_weight + loss_vae * self.loss_config.loss_vae_weight
            loss_dict =  {"loss_total": loss,
                      "loss_eikonal": 0,
                      "loss_surface_sdf": 0,
                      "loss_normals": 0,
                      "loss_psd_sdf": 0,
                      "loss_l1": loss_l1,
                      "loss_vae": loss_vae,
                      "loss_tvloss": 0}
            self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
            return loss

        if self.stats_dir is not None:
            reconstructed_plane_feature = unnormalize(reconstructed_plane_feature, self.stats_dir, self.middle, self._range)
        ###### sdf loss
        sdf_empty_pred = self.decoder_model.forward(reconstructed_plane_feature, points_empty)
        surface_sdf = self.decoder_model.forward(reconstructed_plane_feature, points_surface)
        
        points_norm_pred = self.gradient_tri(self.decoder_model.forward, reconstructed_plane_feature, points_all)
        surface_norm_pred = points_norm_pred[:, :points_surface_num]


        loss_eikonal = ((points_norm_pred.norm(2, dim=-1) - 1) ** 2).mean()
        loss_surface_sdf = (surface_sdf.abs()).mean()
        loss_normals = ((surface_norm_pred - normal_surface).abs()).norm(2, dim=-1).mean()
        loss_psd_sdf = (sdf_empty_pred.squeeze().abs() - sdf_empty_gt.abs()).abs().mean()
        sdf_loss = loss_eikonal * self.loss_config.loss_eikonal_weight + \
                    loss_surface_sdf * self.loss_config.loss_surface_sdf_weight +  \
                    loss_normals * self.loss_config.loss_normals_weight +  \
                    loss_psd_sdf * self.loss_config.loss_psd_sdf_weight

        # ###### tv loss
        # loss_tv = tvloss(reconstructed_plane_feature)


        loss = sdf_loss + loss_vae * self.loss_config.loss_vae_weight
        
        loss_dict =  {"loss_total": loss,
                      "loss_eikonal": loss_eikonal,
                      "loss_surface_sdf": loss_surface_sdf,
                      "loss_normals": loss_normals,
                      "loss_psd_sdf": loss_psd_sdf,
                      "loss_l1": loss_l1,
                      "loss_vae": loss_vae}
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, logger=True)
        return loss


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']