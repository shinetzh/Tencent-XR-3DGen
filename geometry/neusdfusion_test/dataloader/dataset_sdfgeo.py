import os, sys, glob, torch
import numpy as np
import torch
import random
import json
from torch.utils.data import Dataset
from easydict import EasyDict as edict


def single_points_sampler(points, normals):
    '''
    expand points along normals, for each surface 3D point,  sample 1 point along its normal with random interval.
    
    input:
    points: [B, N,3]
    normals: [B, N,3]

    return:
    new_points: [B, N, 3]
    interval: [B, N]

    '''
    t = torch.normal(mean=0, std = 0.1 * torch.ones(points.shape[0], points.shape[1], 1)).to(points.device)
    new_points = torch.clamp(points + normals * t, -1.0, 1.0)
    return new_points, t


def get_expand_color(colors, t):
    '''
    get the color for points in the space based on surface color and surface distance

    colors: [B, N,3] surface color RGB
    t: [B, N,1] surface distance

    new_colors: [N, 3] the colors for space points
    '''
    weights = torch.exp(- t**2)
    new_colors = colors * weights
    return new_colors


class DatasetSdfGeo(Dataset):
    def __init__(self, config, data_type="train", resample=False):
        super(DatasetSdfGeo, self).__init__()
        self.config = config = edict(config)
        self.data_type = data_type
        self.resample = config.get("resample", resample)
        self.datasets = config.get("datasets", ["shapenet"])
        self.gen_latent = config.get("gen_latent", False)

        # parse objects list
        print("checking dataset...")
        self.dataset_json = config.dataset_json
        with open(self.dataset_json, 'r') as fr:
            dataset_dict = json.load(fr)
        
        self.config = edict(dataset_dict["config"]['Config'])

        dataset_all_list = []
        
        num = 0
        if "shapenet" in self.datasets:
            for category in config.categories:
                if category not in dataset_dict['data']['shapenet']:
                    continue
                class_dataset_list = []
                for key, value_dict in dataset_dict['data']['shapenet'][category][data_type].items():
                    if value_dict["GeoPcd"] is None:
                        continue
                    surface_pcd_points_path = os.path.join(value_dict["GeoPcd"], f"pcd_points_{self.config.pcd_num}.npy")
                    surface_pcd_normals_path = os.path.join(value_dict["GeoPcd"], f"pcd_normals_{self.config.pcd_num}.npy")
                    sdf_points_path = os.path.join(value_dict["GeoPcd"], f"sdf_points_{self.config.sdf_num}.npy")
                    sdf_sdfs_path = os.path.join(value_dict["GeoPcd"], f"sdf_sdfs_{self.config.sdf_num}.npy")
                    if "Tri" not in value_dict:
                        continue
                    triplane_path = value_dict["Tri"]
                    if value_dict["Obj"] is None:
                        obj_path = None
                    else:
                        obj_path = value_dict['Obj']

                    class_dataset_list.append((category, key, surface_pcd_points_path, surface_pcd_normals_path, 
                                            sdf_points_path, sdf_sdfs_path, triplane_path, obj_path))
                    num += 1
                # class_dataset_list = class_dataset_list[:60]
                class_dataset_list.sort()
                print("category {}: {}".format(category, len(class_dataset_list)))
                dataset_all_list += class_dataset_list

        if "pix3d" in self.datasets:
            for category in config.categories:
                if category not in dataset_dict['data']['pix3d']:
                    continue
                class_dataset_list = []
                for key, value_dict in dataset_dict['data']['pix3d'][category][data_type].items():
                    if value_dict["GeoPcd"] is None:
                        continue
                    surface_pcd_points_path = os.path.join(value_dict["GeoPcd"], f"pcd_points_{self.config.pcd_num}.npy")
                    surface_pcd_normals_path = os.path.join(value_dict["GeoPcd"], f"pcd_normals_{self.config.pcd_num}.npy")
                    sdf_points_path = os.path.join(value_dict["GeoPcd"], f"sdf_points_{self.config.sdf_num}.npy")
                    sdf_sdfs_path = os.path.join(value_dict["GeoPcd"], f"sdf_sdfs_{self.config.sdf_num}.npy")
                    if "Tri" not in value_dict:
                        continue
                    triplane_path = value_dict["Tri"]
                    if value_dict["Obj"] is None:
                        obj_path = None
                    else:
                        obj_path = value_dict['Obj']

                    class_dataset_list.append((category, key, surface_pcd_points_path, surface_pcd_normals_path, 
                                            sdf_points_path, sdf_sdfs_path, triplane_path, obj_path))
                    num += 1
                class_dataset_list.sort()
                print("category {}: {}".format(category, len(class_dataset_list)))
                dataset_all_list += class_dataset_list

        self.data_list = dataset_all_list

    def __len__(self):
        return len(self.data_list)
    
    def __get_one_sample(self, index, points_num=30000):
        # print(self.data_list[index])
        class_name, key, surface_pcd_points_path, surface_pcd_normals_path, \
            sdf_points_path, sdf_sdfs_path, triplane_path, obj_path= self.data_list[index]
        try:
            surface_pcd_points = torch.from_numpy(np.load(surface_pcd_points_path)).float()
            surface_pcd_normals = torch.from_numpy(np.load(surface_pcd_normals_path)).float()
            sdf_points = torch.from_numpy(np.load(sdf_points_path)).float()
            sdf_sdfs = torch.from_numpy(np.load(sdf_sdfs_path)).float()
            triplane = torch.load(triplane_path, map_location=torch.device('cpu')).squeeze()

        except:
            return self.__get_one_sample(random.randint(0, self.__len__()-1))

        if surface_pcd_points.shape[0] < points_num:
            return self.__get_one_sample(random.randint(0, self.__len__()-1))
        
        perm_surface = np.random.permutation(surface_pcd_points.shape[0])[:points_num]
        perm_sdf = np.random.permutation(sdf_points.shape[0])[:points_num]

        surface_pcd_points = surface_pcd_points[perm_surface]
        surface_pcd_normals = surface_pcd_normals[perm_surface]
        sdf_points = sdf_points[perm_sdf]
        sdf_sdfs = sdf_sdfs[perm_sdf]

        if obj_path is None:
            obj_path = "None"

        return {
            "obj_path" : obj_path,
            "class_name" : class_name,
            "obj_name" : key,
            "surface_points" : surface_pcd_points,
            "surface_normals" : surface_pcd_normals,
            "sdf_points" : sdf_points,
            "sdf_sdfs": sdf_sdfs,
            "triplane" : triplane,
            }


    def __getitem__(self, index):
        return self.__get_one_sample(index)


if __name__ == "__main__":
    
    data_config = {
        "dataset_type": "sdf_sdfgeo",
        "dataset_json" : "all_cfs.json",
        "categories":[
            "03001627",
        ],
        "resample": False
    }

    dataset = DatasetSdfGeo(data_config, data_type="val")
    for key, value in dataset[0].items():
        try:
            print(key, value.shape)
        except:
            print(key, value)
    print(dataset[0].keys())