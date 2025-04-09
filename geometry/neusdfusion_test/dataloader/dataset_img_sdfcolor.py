import os, sys, glob, torch
import numpy as np
import torch
import random
import json
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import torchvision.transforms as transforms
from PIL import Image


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


class DatasetImgSdfColor(Dataset):
    def __init__(self, config, data_type="train", resample=False):
        super(DatasetImgSdfColor, self).__init__()
        self.config = config = edict(config)
        self.data_type = data_type
        self.resample = config.get("resample", resample)
        self.pre_load = config.get("pre_load", True)

        self.gen_latent = config.get("gen_latent", False) 
        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4717, 0.3895, 0.3079], std=[0.1916, 0.2059, 0.2106])
        ])

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
                if value_dict["GeoPcd"] is None:
                    continue
                surface_pcd_points_path = os.path.join(value_dict["GeoPcd"], f"pcd_points_{self.config.pcd_num}.npy")
                surface_pcd_normals_path = os.path.join(value_dict["GeoPcd"], f"pcd_normals_{self.config.pcd_num}.npy")
                sdf_points_path = os.path.join(value_dict["GeoPcd"], f"sdf_points_{self.config.sdf_num}.npy")
                sdf_sdfs_path = os.path.join(value_dict["GeoPcd"], f"sdf_sdfs_{self.config.sdf_num}.npy")
                if "Tri" not in value_dict:
                    continue
                color_pcd_points_path = os.path.join(value_dict["TexPcd"], f"pcd_points_{self.config.color_num}.npy")
                pcd_colors_path = os.path.join(value_dict["TexPcd"], f"pcd_colors_{self.config.color_num}.npy")
                color_pcd_points_normal_path = os.path.join(value_dict["TexPcd"], f"pcd_tex_normals_{self.config.color_num}.npy")
                image100_path = os.path.join(value_dict["ImgDir"], "color", "cam-0031.png")
                if self.pre_load:
                    image = self._load_image(image100_path)
                else:
                    image = image100_path
                if not self.gen_latent:
                    if num < 200 and not self.__checkpath__([surface_pcd_points_path, surface_pcd_normals_path, sdf_points_path, 
                                            sdf_sdfs_path, color_pcd_points_path, pcd_colors_path, color_pcd_points_normal_path]):
                        continue

                class_dataset_list.append((class_name, key, surface_pcd_points_path, surface_pcd_normals_path, 
                                           sdf_points_path, sdf_sdfs_path, color_pcd_points_path,
                                           pcd_colors_path, color_pcd_points_normal_path, image, image100_path))
                num += 1
            class_dataset_list.sort()
            print("{}: {}".format(class_name, len(class_dataset_list)))
            dataset_all_list.append(class_dataset_list)

        self.train_list, self.test_list = self.__split_train_test(dataset_all_list)


        # random.shuffle(self.train_list)

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

    def _load_image(self, image100_path):
        try:
            image = Image.open(image100_path).convert('RGB')
            image = self.img_transform(image)
        except:
            # raise ValueError("Load error: {}".format(image100_path))
            print("Load error: {}".format(image100_path))
            image = torch.randn(3, 256, 256)
        return image

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
        for i, path in enumerate(path_list):
            if path is None:
                return False
            if not os.path.exists(path):
                print("{} not exists!".format(path))
                return False
        return True

    def __len__(self):
        return len(self.dataset_list)
    
    def __get_one_sample(self, index, points_num=30000):
        class_name, key, surface_pcd_points_path, surface_pcd_normals_path, \
            sdf_points_path, sdf_sdfs_path, color_pcd_points_path, \
                pcd_colors_path, color_pcd_points_normal_path, image, image100_path = self.dataset_list[index]
        try:
            surface_pcd_points = torch.from_numpy(np.load(surface_pcd_points_path)).float()
            surface_pcd_normals = torch.from_numpy(np.load(surface_pcd_normals_path)).float()
            sdf_points = torch.from_numpy(np.load(sdf_points_path)).float()
            sdf_sdfs = torch.from_numpy(np.load(sdf_sdfs_path)).float()
            color_pcd_points = torch.from_numpy(np.load(color_pcd_points_path)).float()
            pcd_colors = torch.from_numpy(np.load(pcd_colors_path)).float()
            color_pcd_points_normal = torch.from_numpy(np.load(color_pcd_points_normal_path)).float()

        except:
            return self.__get_one_sample(random.randint(0, self.__len__()-1))

        if surface_pcd_points.shape[0] < points_num:
            return self.__get_one_sample(random.randint(0, self.__len__()-1))
        
        perm_surface = np.random.permutation(surface_pcd_points.shape[0])[:points_num]
        perm_sdf = np.random.permutation(sdf_points.shape[0])[:points_num]
        perm_color = np.random.permutation(color_pcd_points.shape[0])[:points_num]

        surface_pcd_points = surface_pcd_points[perm_surface]
        surface_pcd_normals = surface_pcd_normals[perm_surface]
        sdf_points = sdf_points[perm_sdf]
        sdf_sdfs = sdf_sdfs[perm_sdf]
        color_pcd_points = color_pcd_points[perm_color]
        pcd_colors = pcd_colors[perm_color]
        color_pcd_points_normal = color_pcd_points_normal[perm_color]
        if not self.pre_load:
            image = self._load_image(image)

        return {
            "image" : image,
            "image100_path": image100_path,
            "class_name" : class_name,
            "obj_name" : key,
            "surface_points" : surface_pcd_points,
            "surface_normals" : surface_pcd_normals,
            "sdf_points" : sdf_points,
            "sdf_sdfs": sdf_sdfs,
            "color_points": color_pcd_points,
            "color_colors": pcd_colors,
            "color_points_normal" : color_pcd_points_normal
            }


    def __get_one_sample_for_latent(self, index, points_num=30000):
        class_name, key, surface_pcd_points_path, surface_pcd_normals_path, \
            sdf_points_path, sdf_sdfs_path, triplane_path, color_pcd_points_path, \
                pcd_colors_path, color_pcd_points_normal_path, image, image100_path = self.dataset_list[index]
        
        try:
            triplane = torch.load(triplane_path, map_location=torch.device('cpu')).squeeze()
        except:
            return self.__get_one_sample_for_latent(random.randint(0, self.__len__()-1))
        
        if image100_path is None:
            image100_path = "None"

        return {
            "image100_path" : image100_path,
            "class_name" : class_name,
            "obj_name" : key,
            "triplane" : triplane,
            }

    def __getitem__(self, index):

        return self.__get_one_sample(index)




if __name__ == "__main__":
    from torch.utils.data import DataLoader
    def compute_mean_std(dataset):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        n_samples = 0

        from tqdm.auto import tqdm
        for batch in tqdm(dataloader):
            images = batch['image']
            n_samples += images.size(0)
            mean += images.sum(dim=(0, 2, 3))
            std += (images ** 2).sum(dim=(0, 2, 3))

        mean /= (n_samples * 128 * 128)
        std /= (n_samples * 128 * 128)
        std = torch.sqrt(std - mean ** 2)

        return mean, std

    data_config = {
    "dataset_type": "img_sdfcolor",
    "dataset_json" : "shapenet_128_v2_taiji.json",
    "resample": True
    }
    dataset = DatasetImgSdfColor(data_config, data_type="train")
    mean, std = compute_mean_std(dataset)
    print("Mean:", mean)
    print("Std:", std)

    # Now you can use these mean and std values to normalize your images using transforms.Normalize
    normalize = transforms.Normalize(mean, std)