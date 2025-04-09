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


class DatasetDiffusionPartialConditioned(Dataset):
    def __init__(self, config, data_type="train", resample=False, fix_index=False):
        super(DatasetDiffusionPartialConditioned, self).__init__()
        self.config = config = edict(config)
        self.data_type = data_type
        self.fix_index = fix_index
        self.resample = config.get("resample", resample)
        self.pre_load = config.get("pre_load", False)

        # parse objects list
        print("checking dataset...")
        self.dataset_json = config.dataset_json
        with open(self.dataset_json, 'r') as fr:
            dataset_dict = json.load(fr)
        
        self.config = edict(dataset_dict["config"]['Config'])

        dataset_all_list = []
        
        num = 0
        for category in config.categories:
            if category not in dataset_dict['data']['shapenet']:
                continue
            class_dataset_list = []
            num = 0
            # load latent and feature
            if data_type == "train":
                shape_latent_all = np.load(os.path.join(config.latent_path, category+"_train.npy"))
            partial_latent_all = np.load(os.path.join(config.cond_latent_path, category+"_"+data_type+".npy"))

            for key, value_dict in dataset_dict['data']['shapenet'][category][data_type].items():
                if self.data_type == "train":
                        modulations = shape_latent_all[value_dict['latent_index']]
                        modulations = torch.from_numpy(modulations).flatten().float()
                else:
                    modulations = torch.randn(3072)
                partial_feature = partial_latent_all[value_dict['botton_feature_index']]
                if value_dict["Obj"] is None:
                    obj_path = None
                else:
                    obj_path = value_dict['Obj']
                
                class_dataset_list.append((category, key, modulations, partial_feature, obj_path))
                num += 1
            class_dataset_list.sort()
            print("category {}: {}".format(category, len(class_dataset_list)))
            dataset_all_list += class_dataset_list


        self.data_list = dataset_all_list
        # self.normalize_all()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        category, key, modulation, partial_feature, obj_path = self.data_list[index]
        return {
            "category" : category,
            "obj_name" : key,
            "modulation" : modulation,
            'text_feature': partial_feature,
            "obj_path": obj_path
            }


if __name__ == "__main__":
    
    data_config = {
        "dataset_type" : "diffusion_partial_cond",
        "dataset_json" : "dataset_withindex.json",
        "latent_path": "/aigc_cfs_4/trevorrkcui/data/DiffusionSDF/geo_all_latent_combine",
        "cond_latent_path": "botton",
        "categories":[
        "02691156", "02828884", "02933112", "02958343", "03001627", "03211117", "03636649", "03691459", "04090263", "04256520", "04379243", "04401088", "04530566"
        ],
        "resample": False
    }

    dataset = DatasetDiffusionPartialConditioned(data_config, data_type="train")
    for i in range(5):
        print(dataset[i])
        breakpoint()