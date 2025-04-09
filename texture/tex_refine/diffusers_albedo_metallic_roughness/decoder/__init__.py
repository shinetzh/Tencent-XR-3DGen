import json
import sys
from easydict import EasyDict as edict
from decoder.occupancy_decoder.triplane_decoder import *
from triplane_nvdiffrec.get_dmtet_loss import dmtetNetWork


def get_decoder_model(specs):
    decoder_config = specs["decoder_config"]
    if decoder_config["decoder_type"] == "occupancy":
        decoder_model = OccupancyModel(decoder_config["config_json"], type="occupancy").eval()
    elif decoder_config["decoder_type"] == "craftsman_vae":
        sys.path.append("/aigc_cfs_2/neoshang/code/craftsman_wx/MV2Mesh")
        from craftsman.models.autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder
        from craftsman.utils.config import ExperimentConfig, load_config

        crm_config_path = decoder_config["config_path"]
        cfg = load_config(crm_config_path)
        decoder_model = MichelangeloAutoencoder(cfg.system.shape_model).eval()
    else:
        sys.path.append("/aigc_cfs_2/neoshang/code/test_Diffusionsdf")
        from models import utils_model_custom
        decoder_model = utils_model_custom.get_decoder_model(specs)
    return decoder_model


def get_vae_model(specs):
    vae_model = utils_model_custom.get_vae_model(specs)
    load_from_pretrain = specs["vae_config"].get("load_from_pretrain", None)
    if load_from_pretrain:
        print("load vae from: {}".format(specs["vae_config"]["load_from_pretrain"]))
        ckpt_state_dict = torch.load(specs["vae_config"]["load_from_pretrain"], map_location="cpu")["state_dict"]
        ckpt_vae_state_dict = {}
        for key, value in ckpt_state_dict.items():
            if "vae_model" in key:
                vae_key = key.split("vae_model.")[-1]
                ckpt_vae_state_dict[vae_key] = value
        vae_model.load_state_dict(ckpt_vae_state_dict, strict=True)

    return vae_model
