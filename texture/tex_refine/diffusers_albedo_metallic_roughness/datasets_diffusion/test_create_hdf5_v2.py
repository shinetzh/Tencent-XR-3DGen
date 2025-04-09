import torch
import json
import h5py
import os
import random
import numpy as np
from tqdm import tqdm

from datasets_triplane_diffusion_v3 import DatasetVAEDiffusionV3

# config = {
#             "exp_save_dir": "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdf_character_kl_v0.2.0",
#             "data_config" : {
#                 "dataset_name" : "vae_diffusion_v3",
#                 "dataset_json" : "/aigc_cfs/neoshang/data/json_for_traintest/character/latent_geotri_Transformer_v20_128_people_20240103_neo_add_latent_taiji2.json",
#                 "train_class_exclude_list": ["DragonBall", "onepiece", "Objaverse_Avatar"],
#                 "latent_from_vae": "kl",
#                 "condition_num": 21,
#                 "std_reciprocal": 2.853907392401399,
#                 "scale_type": "std_scale", 
#                 "load_from_cache_last": True,
#                 "resample": False
#             }
#         }

config = {
            "exp_save_dir": "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdf_character_kl_v0.0.0_test910b_float32_dit",
            "data_config" : {
                "dataset_name" : "vae_diffusion_v3",
                "dataset_json" : "/data1/Data/Debug/merge.json",
                "train_class_exclude_list": ["DragonBall", "onepiece", "Objaverse_Avatar"],
                "latent_from_vae": "kl",
                "condition_num": 21,
                "std_reciprocal": 2.853907392401399,
                "scale_type": "std_scale", 
                "load_from_cache_last": True,
                "resample": False
            }
        }

datasets = DatasetVAEDiffusionV3(config, resample=False, data_type="train", load_from_cache_last=False)
print(datasets.__len__())
indexes = list(range(0, datasets.__len__()))
random.shuffle(indexes)
print(indexes[0:100])
json_save_path = "/aigc_cfs_2/neoshang/data/diffusion_datasets_v0/datasets.json"
hdf5_save_dir_list = ["/data3/neoshang/data/diffusion_v0",
                      "/data4/neoshang/data/diffusion_v0",
                      "/data5/neoshang/data/diffusion_v0",
                      "/data6/neoshang/data/diffusion_v0",
                      "/data7/neoshang/data/diffusion_v0",
                      ]


generate_num = min(1000, len(datasets))
image_length = datasets.condition_num

data_list = []

save_dir_idx = 0
num = 0
for idx in tqdm(indexes):
    item = datasets[idx]
    classname = item["classname"]
    objname = item["objname"]
    try_num = 0
    try:
        obj_hdf5_save_dir = os.path.join(hdf5_save_dir_list[save_dir_idx], classname)
        os.makedirs(obj_hdf5_save_dir, exist_ok=True)
        obj_hdf5_save_path = os.path.join(obj_hdf5_save_dir, objname + ".h5")
        with h5py.File(obj_hdf5_save_path, "w") as h5file:
            h5file.create_dataset("latent_modulation_mean", data=item["latent_modulation_mean"].numpy(), compression='gzip', compression_opts=5, chunks=item["latent_modulation_mean"].numpy().shape)
            h5file.create_dataset("latent_modulation_logvar", data=item["latent_modulation_logvar"].numpy(), compression='gzip', compression_opts=5, chunks=item["latent_modulation_logvar"].numpy().shape)
            h5file.create_dataset("image_latent", data=item["image_latent"].data.cpu().numpy().squeeze(), compression='gzip', compression_opts=5, chunks=image_latent.shape)

        data_list.append({"classname": item["classname"],
                            "objname": item["objname"],
                            "h5_path": obj_hdf5_save_path})
    except:
        try_num += 1
        if try_num < 2:
            save_dir_idx += 1
            obj_hdf5_save_dir = os.path.join(hdf5_save_dir_list[save_dir_idx], classname)
            os.makedirs(obj_hdf5_save_dir, exist_ok=True)
            obj_hdf5_save_path = os.path.join(obj_hdf5_save_dir, objname + ".h5")
            with h5py.File(obj_hdf5_save_path, "w") as h5file:
                h5file.create_dataset("latent_modulation_mean", data=item["latent_modulation_mean"].numpy(), compression='gzip', compression_opts=5, chunks=item["latent_modulation_mean"].numpy().shape)
                h5file.create_dataset("latent_modulation_logvar", data=item["latent_modulation_logvar"].numpy(), compression='gzip', compression_opts=5, chunks=item["latent_modulation_logvar"].numpy().shape)
                h5file.create_dataset("image_latent", data=item["image_latent"].data.cpu().numpy().squeeze(), compression='gzip', compression_opts=5, chunks=image_latent.shape)

            data_list.append({"classname": item["classname"],
                                "objname": item["objname"],
                                "h5_path": obj_hdf5_save_path})
    num += 1
    if num >= generate_num:
        break

with open(json_save_path, 'w') as fw:
    json.dump(data_list, fw, indent=2)


##### read h5 test
h5_test_path = data_list[100]["h5_path"]
with h5py.File(h5_test_path, 'r') as h5r:
    print(h5r["latent_modulation_mean"])
    print(h5r["latent_modulation_mean"].shape)
    print(h5r["latent_modulation_logvar"].shape)
    print(h5r["image_latent"].shape)