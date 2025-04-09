#!/usr/bin/env python3


import os
import json
import torch
import torch.utils.data
from diff_utils.helpers import * 
from glob import glob
import numpy as np
from easydict import EasyDict as edict

class DatasetDiffusionCat(torch.utils.data.Dataset):
    def __init__(self, data_config, condition=True, resample=True, data_type="train"):
        super().__init__()
        self.config = config = edict(data_config)
        self.condition = condition
        self.resample=resample
        self.dataset_json = config.dataset_json
        self.geo_triplane_modulation_dir = config.geo_triplane_modulation_dir
        self.tex_triplane_modulation_dir = config.tex_triplane_modulation_dir
        self.image_latent_dir = config.image_latent_dir
        self.data_type = data_type
        self.random_image_index_list = [50, 53, 58, 61, 63, 66, 69, 71, 74, 79, 82, 84, 90, 95, 100, 105, 116, 129, 142, 171, 189]

        # parse objects list
        print("checking dataset...")
        self.dataset_json = config.dataset_json
        with open(self.dataset_json, 'r') as fr:
            dataset_dict = json.load(fr)
        
        self.tex_config = edict(dataset_dict["config"]['TexCfg'])

        self.dataset_list_train = []
        self.dataset_list_test = []
        for class_name, class_dict in dataset_dict["data"].items():
            print(class_name)
            for key, value_dict in class_dict.items():
                if value_dict["TexPcd"] is None or value_dict["ImgDir"] is None:
                    continue
                image_latent_save_dir = os.path.join(self.image_latent_dir, class_name, key)
                if not os.path.exists(image_latent_save_dir):
                    continue
                image_dir = value_dict["ImgDir"]
                pcd_points_path = os.path.join(value_dict["TexPcd"], f"pcd_points_{self.tex_config.pcd_num}.npy")
                pcd_colors_path = os.path.join(value_dict["TexPcd"], f"pcd_colors_{self.tex_config.pcd_num}.npy")
                triplane_color_path = value_dict["TexTri"]
                triplane_tex_path = value_dict["GeoTri"]
                geo_triplane_modulation_path = os.path.join(self.geo_triplane_modulation_dir, class_name, key+".npy")
                tex_triplane_modulation_path = os.path.join(self.tex_triplane_modulation_dir, class_name, key+".npy")
                image100_path = os.path.join(value_dict["ImgDir"], "color", "cam-0100.png")

                if not self.__checkpath__([pcd_points_path, pcd_colors_path, triplane_color_path, triplane_tex_path, image100_path]):
                    continue

                if not (os.path.exists(geo_triplane_modulation_path) and os.path.exists(tex_triplane_modulation_path)):
                    self.dataset_list_test.append((class_name, key, geo_triplane_modulation_path, tex_triplane_modulation_path, image_latent_save_dir, image_dir))
                    continue
                self.dataset_list_train.append((class_name, key, geo_triplane_modulation_path, tex_triplane_modulation_path, image_latent_save_dir, image_dir))
            print("train_num: {}, test_num: {}".format(len(self.dataset_list_train), len(self.dataset_list_test)))
        
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
            train_num = num - test_num
            if self.resample:
                the_repeat_time = 1
                if train_num < (max_num - 10):
                    if train_num * max_repeat > max_num:
                        the_repeat_time = max_num // train_num
                    else:
                        the_repeat_time = max_repeat
                
                print("{} after repeat: {}".format(class_name, int(the_repeat_time * train_num)))
                train_list += class_dataset_list[test_num:] * int(the_repeat_time)
                max_num = int(the_repeat_time * train_num)
            else:
                train_list += class_dataset_list[test_num:]
                max_num = train_num
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
        class_name, key, geo_triplane_modulation_path, tex_triplane_modulation_path, image_latent_save_dir, image_dir = self.dataset_list[index]

        result = {"class_name" : class_name,
                    "obj_name" : key}

        if self.data_type == "train":
            geo_latent_modulation = torch.from_numpy(np.load(geo_triplane_modulation_path)).float().squeeze()
            tex_latent_modulation = torch.from_numpy(np.load(tex_triplane_modulation_path)).float().squeeze()
            result.update({
                "latent_modulation" : torch.cat([geo_latent_modulation, tex_latent_modulation], dim=0)
            })

        if self.data_type == "train":
            image_latent_idx = np.random.choice(self.random_image_index_list)
        elif self.data_type == "test":
            image_latent_idx = 100
        image_latent_path = os.path.join(image_latent_save_dir, "latent_{}.npy".format(str(image_latent_idx).zfill(4)))
        
        if not os.path.exists(image_latent_path):
            print("image latent not found: {}".format(image_latent_path))
            return self.__getitem__(index + 1)
        
        try:
            latent_image = torch.from_numpy(np.load(image_latent_path)).float().squeeze()
        except:
            return self.__getitem__(index + 1)
        
        result.update({"latent_image" : latent_image})


        image100_path = os.path.join(image_dir, "color", "cam-0100.png")
        result.update({"image100_path" : image100_path})

    
        return result