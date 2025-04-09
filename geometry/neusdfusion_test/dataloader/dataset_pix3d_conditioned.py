#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict


import os, sys, glob, torch
import numpy as np
import torch
import random
import json
from torch.utils.data import Dataset
from easydict import EasyDict as edict


class DatasetDiffusionPix3D(Dataset):
    def __init__(self, config, data_type="train", resample=False, fix_index=False):
        super(DatasetDiffusionPix3D, self).__init__()
        self.config = config = edict(config)
        self.data_type = data_type
        self.resample = config.get("resample", resample)
        self.pre_load = config.get("pre_load", False)
        self.data_root = config.get("data_root", None)
        self.categories = config.get("categories", ["bookcase", "desk", "bed", "chair", "sofa", "table", "tool", "misc", "wardrobe"])
        print("pre_load: {}".format(self.pre_load))

        # parse objects list
        print("checking dataset...")
        self.dataset_json = config.get("dataset_json", None)

        with open(self.dataset_json, 'r') as fr:
            dataset_dict = json.load(fr)
        
        # self.config = edict(dataset_dict["config"]['Config'])

        dataset_all_list = []
        self.img_features = {}
        self.triplane_latents = {}
        num = 0
        for category in self.categories:
            if category not in dataset_dict['data']['pix3d']:
                    continue
            img_feature_path = os.path.join(config.data_root, "img_latent_2", f"{category}_{self.data_type}.npy")
            self.img_features[category] = torch.from_numpy(np.load(img_feature_path))
            class_dataset_list = []
            for i in range(len(dataset_dict['data']['pix3d'][category][data_type])):
                # dataset_dict['data']['pix3d'][category][data_type][i]
                # {'img': 'img/bookcase/0001.jpg', 'mask': 'mask/bookcase/0001.png', 'model': 'model/bookcase/IKEA_BESTA/manifold_new_full.obj', 'latent': 'latent/bookcase/IKEA_BESTA.npy'}
                img = dataset_dict['data']['pix3d'][category][data_type][i]['img']
                mask = dataset_dict['data']['pix3d'][category][data_type][i]['mask']
                model = dataset_dict['data']['pix3d'][category][data_type][i]['model']
                latent_path = dataset_dict['data']['pix3d'][category][data_type][i]['latent']
                latent_key = "/".join(latent_path.split(".")[0].split("/")[1:])
                if latent_key not in self.triplane_latents:
                    if os.path.exists(os.path.join(config.data_root, latent_path)):
                        self.triplane_latents[latent_key] = torch.from_numpy(np.load(os.path.join(config.data_root, latent_path)))
                    else:
                        # print("no latent {}".format(latent_key))  # bed/IKEA_MALM_1, chair/IKEA_VRETA, wardrobe/IKEA_PAX_4
                        continue
                img_latent_idx = dataset_dict['data']['pix3d'][category][data_type][i]['latent_idx']

                class_dataset_list.append((category, img, mask, model, latent_key, img_latent_idx))
                num += 1
            class_dataset_list.sort()
            print("category {}: {}".format(category, len(class_dataset_list)))
            dataset_all_list += class_dataset_list

        self.data_list = dataset_all_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        category, img, mask, model, latent_key, img_latent_idx = self.data_list[index]

        z = self.triplane_latents[latent_key]
        latent_image = self.img_features[category][img_latent_idx]

        return {
            "category" : category,
            "obj_name" : img+"; "+model,
            "latent_modulation" : z.flatten(),
            'latent_image': latent_image.flatten(),
            "obj_path": os.path.join(self.config.data_root, model), 
            "image_path": os.path.join(self.config.data_root, img)
            }


if __name__ == "__main__":
   
    data_config = {
        "dataset_type" : "diffusion_pix3d_cond",
        "data_root": "pix3d",
        "dataset_json" : "dataset.json",
        "categories":[
        "chair"
        ],
        "resample": False,
    }

    dataset = DatasetDiffusionPix3D(data_config, data_type="train")
    for i in range(5):
        breakpoint()
        print(dataset[i])