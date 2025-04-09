import torch
import json
import h5py
import os
import random
import numpy as np
from tqdm import tqdm

from datasets_triplane_diffusion_v3 import DatasetVAEDiffusionV3

config = {
            "exp_save_dir": "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdf_character_kl_v0.2.0",
            "data_config" : {
                "dataset_name" : "vae_diffusion_v3",
                "dataset_json" : "/aigc_cfs/neoshang/data/json_for_traintest/character/latent_geotri_Transformer_v20_128_people_20240103_neo_add_latent_taiji2.json",
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
json_save_path = "/aigc_cfs_7/neoshang/data/diffusion_datasets_v0/datasets.json"
hdf5_save_dir = "/aigc_cfs_7/neoshang/data/diffusion_datasets_v0/hdf5"
os.makedirs(hdf5_save_dir, exist_ok=True)

mean_h5_name = "mean.hdf5"
logvar_h5_name = "logvar.hdf5"
image_h5_name = "image.hdf5"

mean_file = h5py.File(os.path.join(hdf5_save_dir, mean_h5_name), "w")
logvar_file = h5py.File(os.path.join(hdf5_save_dir, logvar_h5_name), "w")
image_file = h5py.File(os.path.join(hdf5_save_dir, image_h5_name), "w")

generate_num = min(1000, len(datasets))
image_length = datasets.condition_num

mean_h5 = mean_file.create_dataset("mean", (generate_num,4,16,48), dtype='f', compression='gzip', compression_opts=5)
logvar_h5 = logvar_file.create_dataset("logvar", (generate_num,4,16,48), dtype='f', compression='gzip', compression_opts=5)
image_h5 = image_file.create_dataset("image", (generate_num,image_length,512,512,3), dtype='f', compression='gzip', compression_opts=5)
data_list = []


num = 0
for idx in tqdm(indexes):
    item = datasets[idx]
    data_list.append({"classname": item["classname"],
                        "objname": item["objname"],
                        "image_path_list": item["image_path_list"]})
    mean_h5[num] = item["latent_modulation_mean"].numpy()
    logvar_h5[num] = item["latent_modulation_logvar"].numpy()
    for i, image_path in enumerate(item["image_path_list"]):
        image = datasets.preprocess_image(image_path)
        image_h5[num][i] = image

    num += 1
    if num >= generate_num:
        break

with open(json_save_path, 'w') as fw:
    json.dump(data_list, fw, indent=2)

mean_file.close()
logvar_file.close()
image_file.close()

##### read h5 test
with h5py.File(os.path.join(hdf5_save_dir, mean_h5_name), 'r') as mean_file_r:
    print(mean_file_r["mean"][0])
    print(mean_file_r["mean"][0].shape)