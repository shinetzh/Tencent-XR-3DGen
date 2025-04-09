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


class DatasetDiffusionUnconditioned(Dataset):
    def __init__(self, config, data_type="train", resample=False):
        super(DatasetDiffusionUnconditioned, self).__init__()
        self.config = config = edict(config)
        self.data_type = data_type
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
            for key, value_dict in dataset_dict['data']['shapenet'][category][data_type].items():
                if "latent" not in value_dict.keys():
                    print("no latent {}".format(key))
                    continue
                if self.pre_load:
                    # modulations = np.load(value_dict['latent']).reshape(-1, 1)
                    # modulations = np.split(modulations, 2)
                    # modulations = (torch.tensor(modulations[0], dtype=torch.float32).squeeze(), torch.tensor(modulations[1], dtype=torch.float32).squeeze())
                    modulations = np.load(value_dict['latent'])
                    modulations = torch.from_numpy(modulations).flatten().float()
                    modulations = (modulations, modulations)
                    
                else:
                    modulations = value_dict['latent']
                if value_dict["Obj"] is None:
                    obj_path = None
                else:
                    obj_path = value_dict['Obj']
                
                class_dataset_list.append((category, key, modulations, obj_path))
                num += 1

            class_dataset_list.sort()
            print("category {}: {}".format(category, len(class_dataset_list)))
            dataset_all_list += class_dataset_list


        self.data_list = dataset_all_list
        # self.normalize_all()

    def __len__(self):
        return len(self.data_list)

    def normalize_all(self):
        self.mean = torch.stack([x[2][0] for x in self.data_list], dim=0).mean(dim=0)
        self.std = torch.stack([x[2][0] for x in self.data_list], dim=0).std(dim=0)
        for i in range(len(self.data_list)):
            self.data_list[i] = (self.data_list[i][0], self.data_list[i][1], ((self.data_list[i][2][0] - self.mean) / self.std, self.data_list[i][2][1]), self.data_list[i][3])
        

    def load_modulation(self, modulation_path):
        modulations = np.load(modulation_path)
        z = torch.from_numpy(modulations).flatten().float()

        return z
    
    def __getitem__(self, index):
        category, key, modulation, obj_path = self.data_list[index]

        if self.pre_load:
            z = modulation[0]
            # normalize
            # z = (z - self.mean) / self.std
        else:
            z = self.load_modulation(modulation)

        return {
            "category" : category,
            "obj_name" : key,
            "modulation" : z,
            "obj_path": obj_path
            }


if __name__ == "__main__":
    
    data_config = {
        "dataset_type": "sdf_sdfgeo",
        "dataset_json" : "dataset.json",
        "categories":[
            "02958343",
        ],
        "resample": False,
        "pre_load": False
    }

    dataset = DatasetDiffusionUnconditioned(data_config, data_type="train")
    for i in range(10):
        print(dataset[i])
        breakpoint()