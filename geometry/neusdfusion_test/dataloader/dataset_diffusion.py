#!/usr/bin/env python3


import os
import json
import torch
import torch.utils.data
from diff_utils.helpers import * 
from glob import glob
import numpy as np

class DatasetDiffusion(torch.utils.data.Dataset):
    def __init__(self, data_config, condition=True, return_filename=False, resample=True, data_type="train"):
        super().__init__()
        self.condition = condition
        self.return_filename = return_filename
        self.resample = resample
        self.dataset_json = data_config["dataset_json"]
        self.triplane_modulation_dir = data_config["triplane_modulation_dir"]
        self.image_latent_dir = data_config["image_latent_dir"]
        self.data_type = data_type
        self.random_image_index_list = [50, 53, 58, 61, 63, 66, 69, 71, 74, 79, 82, 84, 90, 95, 100, 105, 116, 129, 142, 171, 189]

        print("checking dataset...")
        self.dataset_list_train = []
        self.dataset_list_test = []
        with open(self.dataset_json, 'r') as fr:
            dataclass_render_dir_tripath = json.load(fr)
            for dataclass, render_dir_tripath in dataclass_render_dir_tripath.items():
                for render_dir, tripath in render_dir_tripath.items():
                    obj_save_name = "==".join(render_dir.split('/')[-3:])
                    image_latent_save_dir = os.path.join(self.image_latent_dir, dataclass, obj_save_name)
                    if not self.__checkpath__([image_latent_save_dir]):
                        continue
                    triplane_modulation_path = os.path.join(self.triplane_modulation_dir, dataclass, obj_save_name + ".npy")
                    if not self.__checkpath__([triplane_modulation_path]):
                        self.dataset_list_test.append((triplane_modulation_path, image_latent_save_dir, render_dir))
                        continue
                    self.dataset_list_train.append((triplane_modulation_path, image_latent_save_dir, render_dir))
            
        if self.data_type == "train":
            self.dataset_list = self.dataset_list_train
        elif self.data_type == "test":
            self.dataset_list = self.dataset_list_test

        print("all objs num: {}".format(len(self.dataset_list_train) + len(self.dataset_list_test)))
        print("train objs num: {}".format(len(self.dataset_list_train)))
        print("test objs num: {}".format(len(self.dataset_list_test)))
        print("obj nums: {}".format(len(self.dataset_list)))
        
    def __split_train_test(self, dataset_list, test_threshold=0.002, test_min_num=10, max_repeat=30):
        train_list, test_list = [], []
        max_num = int(max([len(x) * (1-test_threshold) for x in dataset_list]))
        for class_dataset_list in dataset_list:
            num = len(class_dataset_list)
            test_num = int(max(num * test_threshold, test_min_num))
            test_list += class_dataset_list[0:test_num]
            train_num = num - test_num
            if self.resample:
                the_repeat_time = 1
                if train_num < (max_num - 100):
                    if train_num * max_repeat > max_num:
                        the_repeat_time = max_num // train_num
                    else:
                        the_repeat_time = max_repeat
                
                print("after repeat: {}".format(the_repeat_time * train_num))
                train_list += class_dataset_list[test_num:] * int(the_repeat_time)
            else:
                train_list += class_dataset_list[test_num:]
        return train_list, test_list


    def __checkpath__(self, path_list):
        for path in path_list:
            if not os.path.exists(path):
                print("{} not exists!".format(path))
                return False
        return True

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        triplane_modulation_path = None
        triplane_modulation_path, image_latent_save_dir, render_dir = self.dataset_list[index]

        if self.data_type == "train":
            latent_modulation = torch.from_numpy(np.load(triplane_modulation_path)).float().squeeze()
            result = {
                "latent_modulation" : latent_modulation
            }
        else:
            result = {}

        if self.data_type == "train":
            image_latent_idx = np.random.choice(self.random_image_index_list)
        else:
            image_latent_idx = 100
        image_latent_path = os.path.join(image_latent_save_dir, "latent_{}.npy".format(str(image_latent_idx).zfill(4)))
        latent_image = torch.from_numpy(np.load(image_latent_path)).float().squeeze()
        result.update({"latent_image" : latent_image})
        

        image100_path = os.path.join(render_dir, "color", "cam-0100.png")
        result.update({"image100_path" : image100_path})
        
        return result