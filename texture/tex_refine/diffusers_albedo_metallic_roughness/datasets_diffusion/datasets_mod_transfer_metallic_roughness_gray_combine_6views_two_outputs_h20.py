import numpy as np
import torch
from torch.utils.data import Dataset
import json
import itertools
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
from torchvision.utils import make_grid 
from tqdm import tqdm
import cv2
import random
import json
import os, sys
import shutil

def get_train_path(origin_path, train_path, force_copy=False):
    if os.path.exists(train_path) and (not force_copy):
        return train_path
    else:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        shutil.copy(origin_path, train_path)
        return origin_path

def get_brightness_scale(cond_image, target_image):
    cond_image_a = np.array(cond_image)
    target_image_a = np.array(target_image)

    cond_image_a_gray = cv2.cvtColor(cond_image_a, cv2.COLOR_RGBA2GRAY)
    target_image_a_gray = cv2.cvtColor(target_image_a, cv2.COLOR_RGBA2GRAY)

    cond_brightness = np.mean(cond_image_a_gray, where=cond_image_a[..., -1] > 0)
    target_brightness = np.mean(target_image_a_gray, where=target_image_a[..., -1] > 0)

    brightness_scale = cond_brightness / (target_brightness + 0.000001)
    return min(brightness_scale, 1.0)

def lighting_fast(img, light, mask_img=None):
    """
        img: rgb order, shape:[h, w, 3], range:[0, 255]
        light: [-100, 100]
        mask_img: shape:[h, w], range:[0, 255]
    """
    assert -100 <= light <= 100
    max_v = 4
    bright = (light/100.0)/max_v
    mid = 1.0+max_v*bright
    # print('bright: ', bright, 'mid: ', mid)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    thresh = gray * gray * (mask_img.astype(np.float32) / 255.0)
    t = np.mean(thresh, where=(thresh > 0.1))

    mask = np.where(thresh > t, 255, 0).astype(np.float32)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = cv2.dilate(mask, kernel, iterations=3)
    mask[mask_img==0] = 0
    # cv2.imwrite("mask4.png", mask)
    brightrate = np.where(mask == 255.0, bright, (1.0/t*thresh)*bright)
    mask = np.where(mask == 255.0, mid, (mid-1.0)/t*thresh+1.0)
    img_float = img/255.0
    img_float = np.power(img_float, 1.0/mask[:, :, np.newaxis])*(1.0/(1.0-brightrate[:, :, np.newaxis]))
    img_float = np.clip(img_float, 0, 1.0)*255.0
    return img_float.astype(np.uint8)

def to_rgb_image(maybe_rgba: Image.Image, bg_color=127, edge_aug_threshold=0, bright_scale=None):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        # img = np.random.randint(random_grey_low, random_grey_high, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = np.ones([rgba.size[1], rgba.size[0], 3], dtype=np.uint8) * bg_color
        img = Image.fromarray(img, 'RGB')

        #### bright adapt
        if bright_scale is not None:
            rgba_array = np.array(rgba)
            rgb = cv2.convertScaleAbs(rgba_array[..., :3], alpha=bright_scale, beta=0)
            rgb = Image.fromarray(rgb)
            img.paste(rgb, mask=rgba.getchannel('A'))
        else:
            img.paste(rgba, mask=rgba.getchannel('A'))

        #### edge augmentation
        if edge_aug_threshold > 0 and (random.random() < edge_aug_threshold):
            mask_img = np.array(rgba.getchannel('A'))
            mask_img[mask_img > 0] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            iterration_num = random.randint(1, 2)
            mask_img_small = cv2.erode(mask_img, kernel, iterations=iterration_num)
            mask_img_edge = mask_img - mask_img_small
            mask_img_edge = np.concatenate([mask_img_edge[..., None]]*3, axis=-1) / 255.0
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img_array = np.array(img) * (1 - mask_img_edge) + rand_color * mask_img_edge
            img = Image.fromarray(img_array.astype(np.uint8))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)
    

class DatasetModTransfer_metallic_roughness_two_outputs_h20(Dataset):
    def __init__(self,
        configs,
        data_type = "train",
        bg_color="white",
        load_from_cache_last=True,
        num_samples: Optional[int] = None,
        num_validation_samples=None,
        shuffle = False
        ) -> None:

        exp_dir = configs.get("exp_dir", None)
        assert exp_dir is not None
        self.exp_dir = exp_dir
        print(f"exp_dir: {exp_dir}")
        data_config = configs["data_config"]
        self.image_size = data_config["image_size"]
        self.dataset_json = data_config["dataset_json"]
        self.cond_idx_list = data_config["cond_idx_list"]
        self.images_num_per_group = data_config["images_num_per_group"]
        self.group_idx_list = data_config["group_idx_list"]
        self.group_idx_range = data_config["group_idx_range"]
        self.load_from_cache_last = data_config.get("load_from_cache_last", load_from_cache_last)
        self.bg_color = bg_color
        self.validation = (data_config.get("data_type", data_type) == "test")
        self.num_samples = num_samples
        
        with open(self.dataset_json, 'r') as fr:
            json_dict = json.load(fr)
        
        data_dict = json_dict["data"]["objaverse"]

        train_json_save_path = os.path.join(exp_dir, "train.json")
        test_json_save_path = os.path.join(exp_dir, "test.json")

        if not self.load_from_cache_last:
            print("rechecking data... ")
            all_data_list = self.read_data(data_dict)
            print(len(all_data_list))
            # print(all_data_list[0])
            # print("here!")
            # breakpoint()
            # data_train_list, data_test_list = self.__split_train_test(all_data_list)
            dataset_list_train = all_data_list
            # print(data_train_list)

            # dataset_list_train = list(itertools.chain(*data_train_list))
            # dataset_list_test = list(itertools.chain(*data_test_list))
            # dataset_list_train.sort()
            # dataset_list_test.sort()

            # print(dataset_list_train[0])
            # breakpoint()
        
            print("writing load cache")
            with open(train_json_save_path, "w") as fw:
                if shuffle:
                    random.shuffle(dataset_list_train)
                json.dump(dataset_list_train, fw, indent=2)
            # with open(test_json_save_path, "w") as fw:
                # json.dump(dataset_list_test, fw, indent=2)
        else:
            print("load from cache last")
            with open(train_json_save_path, 'r') as fr:
                dataset_list_train = json.load(fr)
            # with open(test_json_save_path, 'r') as fr:
                # dataset_list_test = json.load(fr)


        if not self.validation:
            self.all_objects = dataset_list_train
        else:
            print("the data can not be used for training !!!")
            # self.all_objects = dataset_list_test
            # if num_validation_samples is not None:
                # self.all_objects = self.all_objects[:num_validation_samples]

        print("loading", len(self.all_objects), " objects in the dataset")

        # Preprocessing the datasets.
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self):
        return len(self.all_objects)

    def read_data(self, data_dict):
        all_data_list = []
        all_num = 0
        # for classname, classdict in tqdm(data_dict.items()):
        #     class_data_list = []
        for objname, objdict in tqdm(data_dict.items()):
            if "ImgDir" not in objdict or "metallic_roughness" not in objdict:
                continue
            image_dir = objdict["ImgDir"]
            metallic_roughness_dir = objdict["metallic_roughness"]
            image_train_dir = objdict["ImgDir_copy_path"]
            metallic_roughness_train_dir = objdict["metallic_roughness_copy_path"]
            # print(image_dir)
            # print(metallic_roughness_dir)
            # breakpoint()
            # for groupi in self.group_idx_list:
            for groupi in range(self.group_idx_range):
                # class_data_list.append([image_dir, metallic_roughness_dir, groupi])

                all_data_list.append([image_dir, metallic_roughness_dir, image_train_dir, metallic_roughness_train_dir, groupi])
        return all_data_list

    # def read_data(self, data_dict):
    #     all_data_list = []
    #     all_num = 0
    #     for classname, classdict in tqdm(data_dict.items()):
    #         class_data_list = []
    #         for objname, objdict in tqdm(classdict.items()):
    #             if "ImgDir" not in objdict or "metallic_roughness" not in objdict:
    #                 continue
    #             image_dir = objdict["ImgDir"]
    #             metallic_roughness_dir = objdict["metallic_roughness"]
    #             print(image_dir)
    #             print(metallic_roughness_dir)
    #             # for groupi in self.group_idx_list:
    #             for groupi in range(self.group_idx_range):
    #                     class_data_list.append([image_dir, metallic_roughness_dir, groupi])

    #         all_data_list.append(class_data_list)
    #     return all_data_list

    def __split_train_test(self, dataset_list, test_threshold=0.001, test_min_num=10):
        train_list, test_list = [], []
        # print("11111")
        # print(len(dataset_list))
        # breakpoint()
        for i, class_dataset_list in enumerate(dataset_list):
            # print("22222")
            # print(class_dataset_list)
            # breakpoint()
            if len(class_dataset_list) == 0:
                print("dataset objs num is 0")
                continue
            class_name = class_dataset_list[0][0]
            num = len(class_dataset_list)
            if num < test_min_num*3:
                print(f"{class_name} dataset objs num is little than test_min_num*3, all {num} for train")
                continue
            test_num = int(max(num * test_threshold, test_min_num))
            test_list.append(class_dataset_list[0:test_num])
            train_list.append(class_dataset_list[test_num:])
            print(f"class {class_name} split {num-test_num} for train and {test_num} for test")
        return train_list, test_list 


    def __getitem__(self, index):
        # print(self.all_objects[index])
        # breakpoint()
        image_dir, metallic_roughness_dir, image_train_dir, metallic_roughness_train_dir, groupi = self.all_objects[index]
        
        image_list = []
        metallic_list = []
        roughness_list = []
        # material_list = []

        random_integer = random.randint(0, 1)

        # if random_integer == 1:
        #     cond_idx_list_use = [a + b for a, b in zip(self.cond_idx_list, [4, 4, 4, 4])]
        # else:
        #     cond_idx_list_use = self.cond_idx_list

        cond_idx_list_use = self.cond_idx_list

        # print("cond_idx_list_use: ", cond_idx_list_use)

        # for condi in self.cond_idx_list:
        for condi in cond_idx_list_use:
            image_cond_sub_idx = self.images_num_per_group * groupi + condi
            metallic_roughness_cond_sub_idx = condi

            rgb_dir = os.path.join(image_dir, "color")
            rgb_train_dir = os.path.join(image_train_dir, "color")
            image_cond_path = os.path.join(rgb_dir, f"cam-{str(image_cond_sub_idx).zfill(4)}.png")
            image_train_cond_path = os.path.join(rgb_train_dir, f"cam-{str(image_cond_sub_idx).zfill(4)}.png")
            # image_cond_path = get_train_path(image_cond_path, image_train_cond_path)
            
            metallic_dir = os.path.join(metallic_roughness_dir, "metallic/color")
            metallic_train_dir = os.path.join(metallic_roughness_train_dir, "metallic/color")
            metallic_cond_path = os.path.join(metallic_dir, f"cam-{str(metallic_roughness_cond_sub_idx).zfill(4)}.png")
            metallic_train_cond_path = os.path.join(metallic_train_dir, f"cam-{str(metallic_roughness_cond_sub_idx).zfill(4)}.png")
            # metallic_cond_path = get_train_path(metallic_cond_path, metallic_train_cond_path)

            roughness_dir = os.path.join(metallic_roughness_dir, "roughness/color")
            roughness_train_dir = os.path.join(metallic_roughness_train_dir, "roughness/color")
            roughness_cond_path = os.path.join(roughness_dir, f"cam-{str(metallic_roughness_cond_sub_idx).zfill(4)}.png")
            roughness_train_cond_path = os.path.join(roughness_train_dir, f"cam-{str(metallic_roughness_cond_sub_idx).zfill(4)}.png")
            # roughness_cond_path = get_train_path(roughness_cond_path, roughness_train_cond_path)

            image = Image.open(image_cond_path)
            image = to_rgb_image(image, 127)
            image = self.train_transforms(image)

            metallic = Image.open(metallic_cond_path)
            metallic = to_rgb_image(metallic, 127)
            metallic = self.train_transforms(metallic)

            roughness = Image.open(roughness_cond_path)
            roughness = to_rgb_image(roughness, 127)
            roughness = self.train_transforms(roughness)

            # print(metallic.shape)
            # print(roughness.shape)

            # hold = metallic[0,:,:].new_zeros(metallic[0,:,:].size())
            # print(hold.shape)
            # print(hold)

            # material = torch.cat((metallic[0,:,:].unsqueeze(0), roughness[0,:,:].unsqueeze(0), hold.unsqueeze(0)), dim=0)
            # print(material.shape)
            # breakpoint()

            image_list.append(image[None])
            metallic_list.append(metallic[None])
            roughness_list.append(roughness[None])
            # material_list.append(material[None])

        images = torch.cat(image_list)
        metallics = torch.cat(metallic_list)
        roughnesses = torch.cat(roughness_list)
        # materials = torch.cat(material_list)

        return {
            "images": images,
            "metallics": metallics,
            "roughnesses": roughnesses,
            # "materials": materials,
        }

if __name__ == "__main__":

    configs = {
                "exp_dir": "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/modality_transfer/rgb2norm_v1",
                "data_config":{
                    "dataset_name" : "DatasetModTransfer_metallic_roughness",
                    # "dataset_json" : "/aigc_cfs_2/neoshang/data/data_list/20240701/part1_3_400k.json",
                    # "dataset_json" : "/aigc_cfs_11/Asset/active_list/3d_diffusion/pbr/all_with_normal.json",
                    "dataset_json" : "/aigc_cfs_4/xibin/code/scripts/train.json",
                    "image_size": 512,
                    "group_idx_list": [0, 1],
                    "group_idx_range": 57,
                    "cond_idx_list": [0,1,2,3,4,5],
                    "images_num_per_group": 6,
                    "load_from_cache_last": False
                }
            }

    train_dataset = DatasetModTransfer_metallic_roughness_two_outputs_h20(configs, data_type="train", load_from_cache_last=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 1, num_workers=0, pin_memory=True, shuffle=True)

    for data in train_dataloader:
        images = data["images"]
        roughnesses = data["roughnesses"]
        metallics = data["metallics"]
        # materials = data["materials"]

        print(images.shape)
        print(f"images.min(): {images.min()}")
        print(f"images.max(): {images.max()}")
        
        print(roughnesses.shape)
        print(f"roughnesses.min(): {roughnesses.min()}")
        print(f"roughnesses.max(): {roughnesses.max()}")
        
        print(metallics.shape)
        print(f"metallics.min(): {metallics.min()}")
        print(f"metallics.max(): {metallics.max()}")
        
        from torchvision import utils as vutils
        from torchvision.utils import make_grid, save_image

        images = make_grid(images[0], nrow=2, padding=0) * 0.5 + 0.5
        save_image(images, "images.jpg")
        
        metallics = make_grid(metallics[0], nrow=2, padding=0) * 0.5 + 0.5
        save_image(metallics, "metallics.jpg")

        roughnesses = make_grid(roughnesses[0], nrow=2, padding=0) * 0.5 + 0.5
        save_image(roughnesses, "roughnesses.jpg")

        # materials = make_grid(materials[0], nrow=2, padding=0) * 0.5 + 0.5
        # save_image(materials, "materials.jpg")
        breakpoint()
