#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict


class DatasetDiffusionUnconditioned(Dataset):
    def __init__(self, config, data_type="train", resample=False):
        super(DatasetDiffusionUnconditioned, self).__init__()
        self.config = config = edict(config)
        self.data_type = data_type
        self.resample = config.get("resample", resample)

        # parse objects list
        print("checking dataset...")
        self.dataset_json = config.dataset_json
        with open(self.dataset_json, 'r') as fr:
            dataset_dict = json.load(fr)
        
        self.config = edict(dataset_dict["config"]['Config'])

        dataset_all_list = []
        
        for class_name, class_dict in dataset_dict["data"].items():
            class_dataset_list = []
            num = 0
            for key, value_dict in class_dict.items():
                if "latent" not in value_dict.keys():
                    continue
                modulations = np.load(value_dict['latent']).reshape(-1, 1)
                modulations = np.split(modulations, 2)
                modulations = (torch.tensor(modulations[0], dtype=torch.float32).squeeze(), torch.tensor(modulations[1], dtype=torch.float32).squeeze())
                if value_dict["ImgDir"] is None:
                    image100_path = None
                else:
                    image100_path = os.path.join(value_dict["ImgDir"], "color", "cam-0031.png")
                
                class_dataset_list.append((class_name, key, modulations, image100_path))
                num += 1
            class_dataset_list.sort()
            print("{}: {}".format(class_name, len(class_dataset_list)))
            dataset_all_list.append(class_dataset_list)


        self.train_list, self.test_list = self.__split_train_test(dataset_all_list)

        print("all available  objs num: {}".format(len(self.train_list) + len(self.test_list)))
        print("train available  objs num: {}".format(len(self.train_list)))
        print("test available  objs num: {}".format(len(self.test_list)))

        if self.data_type == "train":
            self.dataset_list = self.train_list
        elif self.data_type == "test":
            self.dataset_list = self.test_list

        if "max_num_train_obj" in config:
            max_num_train_obj = config.max_num_train_obj
            self.dataset_list = self.dataset_list[:max_num_train_obj]

        self.nobjs = len(self.dataset_list)

        print("final train objs num: {}".format(self.nobjs))


    def __split_train_test(self, dataset_list, test_threshold=0.002, test_min_num=10, max_repeat=30):
        train_list, test_list = [], []
        dataset_list.sort(key=len, reverse=True)
        max_num = len(dataset_list[0]) * (1-test_threshold)
        for i, class_dataset_list in enumerate(dataset_list):
            if len(class_dataset_list) == 0:
                print("dataset objs num is 0")
                continue
            class_name = class_dataset_list[0][0]
            num = len(class_dataset_list)
            if num < test_min_num*3:
                train_list += class_dataset_list * int(max_repeat)
                print("{} dataset objs num is little than test_min_num*3, all for train, after repeat: {}".format(class_name, len(class_dataset_list) * max_repeat))
                continue
            test_num = int(max(num * test_threshold, test_min_num))
            test_list += class_dataset_list[0:test_num]
            train_list += class_dataset_list[test_num:]
        return train_list, test_list

    def __len__(self):
        return len(self.dataset_list)
    
    def __get_one_sample(self, index):
        class_name, key, modulation, image100_path = self.dataset_list[index]

        if self.resample:
            mu, logvar = modulation
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
        else:
            z = modulation[0]
        
        if image100_path is None:
            image100_path = "None"

        return {
            "image100_path" : image100_path,
            "class_name" : class_name,
            "obj_name" : key,
            "modulation" : z
            }

    def __getitem__(self, index):
        return self.__get_one_sample(index)



if __name__ == "__main__":
    data_config = {
    "dataset_type" : "diffusion_uncond",
    "dataset_json" : "shapenet_128_v2.json",
    "resample": False
    }
    dataset = DatasetDiffusionUnconditioned(data_config, data_type="train")
    # print(len(dataset))
    # print(dataset[0])
    # import pdb;pdb.set_trace()
    logvar = []
    for i in range(len(dataset.train_list)):
        logvar.append(dataset.train_list[i][2][1])
    logvar = np.vstack(logvar)
    import pdb;pdb.set_trace()