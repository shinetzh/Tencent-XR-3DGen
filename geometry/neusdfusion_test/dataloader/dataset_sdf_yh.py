import os, sys, glob, torch
import numpy as np
import torch
import random
import json
from torch.utils.data import Dataset
from easydict import EasyDict as edict


class DatasetSdfYh(Dataset):
    def __init__(self, config, data_type="train", resample=True):
        super(DatasetSdfYh, self).__init__()
        self.config = config = edict(config)
        self.data_type = data_type
        self.resample = resample
        self.points = []
        self.masks = []
        self.normals = []
        self.sdfs = []
        self.colors = []
        # parse objects list
        print("checking dataset...")
        self.dataset_json = config.dataset_json
        with open(self.dataset_json, 'r') as fr:
            dataset_dict = json.load(fr)
        
        self.geo_config = edict(dataset_dict["config"]['GeoCfg'])

        dataset_all_list = []
        for class_name, class_dict in dataset_dict["data"].items():
            class_dataset_list = []
            for key, value_dict in class_dict.items():
                if value_dict["GeoPcd"] is None:
                    continue
                pcd_points_path = os.path.join(value_dict["GeoPcd"], f"pcd_points_{self.geo_config.pcd_num}.npy")
                pcd_normals_path = os.path.join(value_dict["GeoPcd"], f"pcd_normals_{self.geo_config.pcd_num}.npy")
                sdf_points_path = os.path.join(value_dict["GeoPcd"], f"sdf_points_{self.geo_config.sdf_num}.npy")
                sdf_sdfs_path = os.path.join(value_dict["GeoPcd"], f"sdf_sdfs_{self.geo_config.sdf_num}.npy")
                triplane_path = value_dict["GeoTri"]
                if value_dict["ImgDir"] is None:
                    image100_path = None
                else:
                    image100_path = os.path.join(value_dict["ImgDir"], "cam-0100.png")
                if not (os.path.exists(pcd_points_path) and 
                        os.path.exists(pcd_normals_path) and 
                        os.path.exists(sdf_points_path) and 
                        os.path.exists(sdf_sdfs_path) and 
                        os.path.exists(triplane_path)):
                    continue
                class_dataset_list.append((class_name, key, pcd_points_path, pcd_normals_path, 
                                           sdf_points_path, sdf_sdfs_path, triplane_path, image100_path))
                class_dataset_list.sort()
            print("{}: {}".format(class_name, len(class_dataset_list)))
            dataset_all_list.append(class_dataset_list)


        self.train_list, self.test_list = self.__split_train_test(dataset_all_list)

        print("all objs num: {}".format(len(self.train_list) + len(self.test_list)))
        print("train objs num: {}".format(len(self.train_list)))
        print("test objs num: {}".format(len(self.test_list)))

        if self.data_type == "train":
            self.dataset_list = self.train_list
        elif self.data_type == "test":
            self.dataset_list = self.test_list

        self.nobjs = len(self.dataset_list)

        print("objs num: {}".format(self.nobjs))

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


    def __len__(self):
        return len(self.dataset_list)
    
    def __get_one_sample(self, index, points_num=30000):
        class_name, objname, pcd_points_path, pcd_normals_path, \
          sdf_points_path, sdf_sdfs_path, sdf_triplane_path, image100_path = self.dataset_list[index]
        try:
            pcd_points = torch.from_numpy(np.load(pcd_points_path)).float()
            pcd_normals = torch.from_numpy(np.load(pcd_normals_path)).float()
            sdf_points = torch.from_numpy(np.load(sdf_points_path)).float()
            sdf_sdfs = torch.from_numpy(np.load(sdf_sdfs_path)).float()
            sdf_triplane = torch.load(sdf_triplane_path, map_location=torch.device('cpu')).squeeze()
        except:
            return self.__get_one_sample(random.randint(0, self.__len__()-1))

        if pcd_points.shape[0] < points_num:
            return self.__get_one_sample(random.randint(0, self.__len__()-1))
        
        perm_surface = np.random.permutation(pcd_points.shape[0])[:points_num]
        perm_sdf = np.random.permutation(sdf_points.shape[0])[:points_num]

        pcd_points = pcd_points[perm_surface]
        pcd_normals = pcd_normals[perm_surface]
        sdf_points = sdf_points[perm_sdf]
        sdf_sdfs = sdf_sdfs[perm_sdf]

        if image100_path is None:
            image100_path = "None"

        return {
            "image100_path" : image100_path,
            "class_name" : class_name,
            "obj_name" : objname,
            "surface_points" : pcd_points,
            "surface_normals" : pcd_normals,
            "sdf_points" : sdf_points,
            "sdf_sdfs": sdf_sdfs,
            "sdf_triplane" : sdf_triplane
            }


    def __getitem__(self, index):
        return self.__get_one_sample(index)
