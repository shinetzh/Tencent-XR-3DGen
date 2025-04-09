import torch.nn as nn
import torch.nn.init as init
from easydict import EasyDict as edict
from omegaconf import OmegaConf


# add paths in model/__init__.py for new models
from models import * 
from models.archs.diffusion_arch_latent_cond import DiffusionNetLatentCond
from models.archs.diffusion_arch_latent_uncond import DiffusionNetLatentUncond
from models.archs.diffusion_arch_uncond import UViT
from models.diffusion_latent_cond import DiffusionModelVectorCond
from models.diffusion_latent_uncond import DiffusionModelVectorUncond
from models.diffusion_spacial_cond import DiffusionModelSpacialCond
from models.diffusion_onestage_uncond import DiffusionModelOneStageUncond

from models.autoencoder import *
from models.autoencoder_style import *
from models.autoencoder_img import *
from models.autoencoder_release import *

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)) + "/triplane_nvdiffrec")
from triplane_nvdiffrec.get_dmtet_loss import dmtetNetWork

def get_decoder_model(specs):
    decoder_config = specs["decoder_config"]
    if decoder_config["decoder_type"] == "sdfyh":
        decoder_model = SdfModelYh(decoder_config["config_json"], type="geo")
        if "train_params" in specs.keys() and specs["train_params"] == "vae":
            decoder_model.eval()
            for p in decoder_model.parameters():
                p.requires_grad = False
    elif decoder_config["decoder_type"] == "sdfyh_color_only":
        decoder_model = SdfModelYh(decoder_config["config_json"], type="tex")
        if "train_params" in specs.keys() and specs["train_params"] == "vae":
            decoder_model.eval()
            for p in decoder_model.parameters():
                p.requires_grad = False
    elif decoder_config["decoder_type"] == "sdfyh_and_color":
        decoder_model = SdfModelYhColor(decoder_config["config_json"])
        if "train_params" in specs.keys() and specs["train_params"] == "vae":
            decoder_model.eval()
            for p in decoder_model.parameters():
                p.requires_grad = False

    elif decoder_config["decoder_type"] == "sdfcolor":
        decoder_model = SdfColorModel(decoder_config["config_json"])
        if "train_params" in specs.keys() and specs["train_params"] == "vae":
            decoder_model.eval()
            for p in decoder_model.parameters():
                p.requires_grad = False

    elif decoder_config["decoder_type"] == "dmtet":
        with open(decoder_config["config_json"], 'r') as fr:
            dmtet_flags = edict(json.load(fr))
        decoder_model = dmtetNetWork(dmtet_flags)
        if "train_params" in specs.keys() and specs["train_params"] == "vae":
            decoder_model.eval()
            for p in decoder_model.parameters():
                p.requires_grad = False
    elif decoder_config["decoder_type"] == "sdfgeo":
        decoder_model = SdfGeoModel(decoder_config["config_json"])
        if "train_params" in specs.keys() and specs["train_params"] == "vae":
            decoder_model.eval()
            for p in decoder_model.parameters():
                p.requires_grad = False
        
    else:
        print("did not recogonize decoder name")
        exit(1)

    return decoder_model


def get_diffusion_model(specs):
    diffusion_config = specs["diffusion_config"]
    diffusion_type = diffusion_config["diffusion_type"]
    if diffusion_type == "diffusion_vector":
        diffusion_model_specs = diffusion_config["diffusion_model_specs"]
        diffusion_specs = diffusion_config["diffusion_specs"]
        diffusion_model = DiffusionModelVectorCond(model=DiffusionNetLatentCond(**diffusion_model_specs), **diffusion_specs)
    elif diffusion_type == "diffusion_unconditioned_vector":
        diffusion_model_specs = diffusion_config["diffusion_model_specs"]
        diffusion_specs = diffusion_config["diffusion_specs"]
        diffusion_model = DiffusionModelVectorUncond(model=DiffusionNetLatentUncond(**diffusion_model_specs), **diffusion_specs)
    elif diffusion_type == "diffusion_spacial":
        diffusion_spcial_config = OmegaConf.load(diffusion_config["diffusion_config_path"]).model
        diffusion_model = DiffusionModelSpacialCond(diffusion_spcial_config)
    elif diffusion_type == "diffusion_uncond_uvit":
        diffusion_model_specs = diffusion_config["diffusion_model_specs"]
        diffusion_specs = diffusion_config["diffusion_specs"]
        diffusion_model = DiffusionModelOneStageUncond(model=UViT(**diffusion_model_specs), **diffusion_specs)
    else:
        print("diffusion_type wrong")
        exit(1)

    diffusion_model.apply(weight_init)

    return diffusion_model


def get_vae_model(specs):
    vae_config = specs["vae_config"]
    vae_type = vae_config["vae_type"]
    kl_std = vae_config.get("kl_std", 0.001)
    kl_weight = vae_config.get("kl_weight", 1.0)
    plane_shape = vae_config["plane_shape"]
    z_shape = vae_config.get("z_shape", 256)

    if vae_type == "beta_vae":
        vae_model = BetaVAE(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
    elif vae_type == "beta_vae_2":
        vae_model = BetaVAE2(vae_config)
    elif vae_type == "beta_vae_spacial":
        vae_model = BetaVAESpacial(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
    elif vae_type == "beta_vae_spacial2":
        vae_model = BetaVAESpacial2(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
    elif vae_type == "beta_vae_spacial2_unet":
        vae_model = BetaVAESpacial2_Unet(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
    elif vae_type == "BetaVAERolloutTransformer_v2":
        vae_model = BetaVAERolloutTransformer_v2(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v2_128":
        vae_model = BetaVAERolloutTransformer_v2_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v4_128":
        vae_model = BetaVAERolloutTransformer_v4_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v3":
        vae_model = BetaVAERolloutTransformer_v3(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v5_128":
        vae_model = BetaVAERolloutTransformer_v5_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v6_128":
        vae_model = BetaVAERolloutTransformer_v6_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v7_128":
        vae_model = BetaVAERolloutTransformer_v7_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v8_128":
        vae_model = BetaVAERolloutTransformer_v8_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v9_128":
        vae_model = BetaVAERolloutTransformer_v9_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v10_128":
        vae_model = BetaVAERolloutTransformer_v10_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v11_128":
        vae_model = BetaVAERolloutTransformer_v11_128(vae_config)
    elif vae_type == "BetaVAERolloutTransformer_v12_128":
        vae_model = BetaVAERolloutTransformer_v12_128(vae_config)
    elif vae_type == "Transformer_v13_128_spatial":
        vae_model = Transformer_v13_128_spatial(vae_config)
    elif vae_type == 'TransformerVAE_128':
        vae_model = TransformerVAE_128(vae_config)
    elif vae_type == "BetaVAE_VQ":
        vae_model = BetaVAE_VQ(vae_config)
    elif vae_type == "spacial_vae_rollout":
        vae_model = SpacialVAERollout(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
    elif vae_type == "spacial_vae_rollout_aware3d":
        vae_model = SpacialVAERolloutAware3d(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
    elif vae_type == "spacial_vae_cross_attention":
        vae_config.pop("vae_type")
        vae_model = SpacialVAECrossAttention(**vae_config)
    elif vae_type == "StyleSwinVAE_v2_128":
        vae_model = StyleSwinVAE_v2_128(vae_config)
    elif vae_type == "StyleSwinVAE_v3_128":
        vae_model = StyleSwinVAE_v3_128(vae_config)
    elif vae_type == "StyleSwinVAE_v4_128":
        vae_model = StyleSwinVAE_v4_128(vae_config)
    elif vae_type == "StyleSwinVAE_v5_128":
        vae_model = StyleSwinVAE_v5_128(vae_config)
    elif vae_type == "StyleSwinVAE_v6_128":
        vae_model = StyleSwinVAE_v6_128(vae_config)
    elif vae_type == "StyleSwinVAE_v7_128":
        vae_model = StyleSwinVAE_v7_128(vae_config)
    elif vae_type == "StyleSwinVAE_v8_128":
        vae_model = StyleSwinVAE_v8_128(vae_config)
    elif vae_type == "StyleSwinVAE_v9_128":
        vae_model = StyleSwinVAE_v9_128(vae_config)
    elif vae_type == "StyleSwinVAE_v10_128":
        vae_model = StyleSwinVAE_v10_128(vae_config)
    elif vae_type == "BetaVAETransformer_v1_128":
        vae_model = BetaVAETransformer_v1_128(vae_config)
    elif vae_type == "BetaVAETransformer_v2_128":
        vae_model = BetaVAETransformer_v2_128(vae_config)
    elif vae_type == "BetaVAETransformer_v2_128":
        vae_model = BetaVAETransformer_v2_128(vae_config)
    elif vae_type == "BetaVAETransformer_v3_128":
        vae_model = BetaVAETransformer_v3_128(vae_config)
    elif vae_type == "AutoencoderKL_v1":  # image
        vae_model = AutoencoderKL_v1(vae_config)
    elif vae_type == "AutoencoderKL": # triplane
        vae_model = AutoencoderKL(vae_config)
    elif vae_type == "AutoencoderCNN": # triplane
        vae_model = AutoencoderCNN(vae_config)
    elif vae_type == "AutoencoderKLSAPE":
        vae_model = AutoencoderKLSAPE(vae_config)
    else:
        raise NotImplementedError
    vae_model.apply(weight_init)
    return vae_model


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)