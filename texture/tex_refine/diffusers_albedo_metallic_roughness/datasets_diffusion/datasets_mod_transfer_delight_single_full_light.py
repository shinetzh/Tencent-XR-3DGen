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
from torchvision.transforms import ToPILImage

def get_brightness_scale(cond_image, target_image):
    cond_image_a = np.array(cond_image)
    target_image_a = np.array(target_image)

    cond_image_a_gray = cv2.cvtColor(cond_image_a, cv2.COLOR_RGBA2GRAY)
    target_image_a_gray = cv2.cvtColor(target_image_a, cv2.COLOR_RGBA2GRAY)

    cond_brightness = np.mean(cond_image_a_gray, where=cond_image_a[..., -1] > 0)
    target_brightness = np.mean(target_image_a_gray, where=target_image_a[..., -1] > 0)

    brightness_scale = cond_brightness / (target_brightness + 0.000001)
    # return min(brightness_scale, 1.0)
    return brightness_scale

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
        return maybe_rgba, None
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        mask = np.array(rgba.getchannel('A'))
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
        return img, mask
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

def to_material_image_mask(material_path, mask, resolution):
    material = cv2.imread(material_path, cv2.IMREAD_UNCHANGED)
    material = cv2.resize(material, (int(resolution), int(resolution)), interpolation=cv2.INTER_CUBIC)
    material = material[:,:,:3]

    mask = np.array(mask)
    # print(mask.shape)

    mask = cv2.resize(mask, (int(resolution), int(resolution)), interpolation=cv2.INTER_CUBIC)

    material = material.astype('float32')
    # material = material.transpose(2, 0, 1)

    material[material > 127] = 255.0
    material[material<= 127] = 0

    # print(material.shape)

    material[:,:,0][mask==0] = 127
    material[:,:,1][mask==0] = 127
    material[:,:,2][mask==0] = 127

    # breakpoint()

    material = Image.fromarray(material.astype(np.uint8))

    # 将图像转换为 PyTorch 张量
    # normal_tensor = torch.from_numpy(normal) / 255.0
    # normal_tensor = torch.from_numpy(normal) / 255.0
    # normal_tensor = torch.from_numpy(normal) / 255.0

    # print("max_normal: ",  torch.max(normal_tensor))
    # print("min_normal: ", torch.min(normal_tensor))
    # breakpoint()

    return material
    
    

class DatasetModTransfer_single_full_light(Dataset):
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
        self.cond_input_idx_list = data_config["cond_input_idx_list"]
        self.cond_gt_idx_list = data_config["cond_gt_idx_list"]
        self.images_num_per_group = data_config["images_num_per_group"]
        self.group_idx_list = data_config["group_idx_list"]
        self.group_idx_range = data_config["group_idx_range"]
        self.load_from_cache_last = data_config.get("load_from_cache_last", load_from_cache_last)
        self.bg_color = bg_color
        self.validation = (data_config.get("data_type", data_type) == "test")
        self.num_samples = num_samples
        
        with open(self.dataset_json, 'r') as fr:
            json_dict = json.load(fr)
        
        # data_dict = json_dict["data"]["objaverse"]
        data_dict = json_dict["data"]

        train_json_save_path = os.path.join(exp_dir, "train.json")
        test_json_save_path = os.path.join(exp_dir, "test.json")

        if not self.load_from_cache_last:
            print("rechecking data... ")
            all_data_list = self.read_data(data_dict)
            print(len(all_data_list))
            dataset_list_train = all_data_list
        
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
        # print(data_dict.keys())
        # breakpoint()
        for classname, classdict in tqdm(data_dict.items()):
            # class_data_list = []
            for objname, objdict in tqdm(data_dict[classname].items()):
                if "img_light_path" not in objdict or "img_gt_path" not in objdict:
                    continue
                image_dir = objdict["img_light_path"]
                gt_dir = objdict["img_gt_path"]
                for groupi in range(self.group_idx_range):
                    # class_data_list.append([image_dir, metallic_roughness_dir, groupi])

                    all_data_list.append([image_dir, gt_dir, groupi])
        # print(len(all_data_list))
        return all_data_list

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
        image_dir, gt_dir, groupi = self.all_objects[index]
        
        # image_list = []
        # metallic_list = []
        # roughness_list = []
        # material_list = []

        random_integer = random.randint(0, 5)

        # if random_integer == 1:
        #     cond_idx_list_use = [a + b for a, b in zip(self.cond_idx_list, [4, 4, 4, 4])]
        # else:
        #     cond_idx_list_use = self.cond_idx_list

        cond_input_idx_list_use = self.cond_input_idx_list
        cond_gt_idx_list_use = self.cond_gt_idx_list

        condi_input = cond_input_idx_list_use[random_integer]
        condi_gt = cond_gt_idx_list_use[random_integer]

        # print("cond_idx_list_use: ", cond_idx_list_use)

        # for condi in self.cond_idx_list:
        # for condi in cond_idx_list_use:
        if True:
            image_cond_sub_idx = self.images_num_per_group * groupi + condi_input
            gt_cond_sub_idx = condi_gt
            image_cond_path = os.path.join(image_dir, "color", f"cam-{str(image_cond_sub_idx).zfill(4)}.png")
            gt_cond_path = os.path.join(gt_dir, "color", f"cam-{str(gt_cond_sub_idx).zfill(4)}.png")

            if os.path.exists(image_cond_path) and os.path.exists(gt_cond_path):

                self.used_image_cond_path = image_cond_path
                self.used_gt_cond_path = gt_cond_path

                image = Image.open(image_cond_path)
                gt = Image.open(gt_cond_path)

                prob = random.uniform(0, 1)

                # print(prob)

                if prob < 0.3:
                    image = image.resize((512, 512))
                    image = image.resize((1024, 1024))
                    gt = gt.resize((512, 512))
                    gt = gt.resize((1024, 1024))

                brightness = get_brightness_scale(image, gt)

                image = image.resize((self.image_size, self.image_size))
                image, mask = to_rgb_image(image, 127)
                image = self.train_transforms(image)

                
                gt = gt.resize((self.image_size, self.image_size))
                gt, _ = to_rgb_image(gt, 127, bright_scale=brightness)
                # metallic = to_material_image_mask(metallic_cond_path, mask, resolution=self.image_size )
                gt = self.train_transforms(gt)
            
            else:
                image = Image.open(self.used_image_cond_path)
                gt = Image.open(gt_cond_path)

                brightness = get_brightness_scale(image, gt)

                image = image.resize((self.image_size, self.image_size))
                image, mask = to_rgb_image(image, 127)
                image = self.train_transforms(image)

                # gt = Image.open(self.used_gt_cond_path)
                gt = gt.resize((self.image_size, self.image_size))
                gt, _ = to_rgb_image(gt, 127, bright_scale=brightness)
                gt = self.train_transforms(gt)

        return {
            "image": image,
            "gt": gt,
            # "roughnesses": roughnesses,
            # "materials": materials,
        }

if __name__ == "__main__":

    configs = {
                "exp_dir": "/aigc_cfs_4/xibin/code/diffusers_triplane_models/delight_models_albedo",
                "data_config":{
                    "dataset_name" : "DatasetModTransfer_single_full_light",
                    # "dataset_json" : "/aigc_cfs_2/neoshang/data/data_list/20240701/part1_3_400k.json",
                    # "dataset_json" : "/aigc_cfs_11/Asset/active_list/3d_diffusion/pbr/all_with_normal.json",
                    # "dataset_json" : "/aigc_cfs_4/xibin/code/scripts/delight_150k_train.json",
                    "dataset_json" : "/aigc_cfs_4/xibin/code/scripts/delight_1024_data/delight_all_1024.json",
                    "image_size": 1024,
                    "group_idx_list": [0, 1],
                    "group_idx_range": 32,
                    "cond_input_idx_list": [0,1,2,3,4,5],
                    "cond_gt_idx_list": [0,1,2,3,4,5],
                    "images_num_per_group": 6,
                    "load_from_cache_last": False
                }
            }

    train_dataset = DatasetModTransfer_single_full_light(configs, data_type="train", load_from_cache_last=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 1, num_workers=0, pin_memory=True, shuffle=True)

    i = 0

    to_pil = ToPILImage()

    for data in train_dataloader:
        images = data["image"].squeeze(0)
        gt = data["gt"].squeeze(0)
        # metallics = data["metallics"]
        # materials = data["materials"]

        print(images.shape)
        print(f"images.min(): {images.min()}")
        print(f"images.max(): {images.max()}")
        
        # print(roughnesses.shape)
        # print(f"roughnesses.min(): {roughnesses.min()}")
        # print(f"roughnesses.max(): {roughnesses.max()}")
        
        print(gt.shape)
        print(f"gt.min(): {gt.min()}")
        print(f"gt.max(): {gt.max()}")
        
        from torchvision import utils as vutils
        from torchvision.utils import make_grid, save_image

        images = to_pil(images * 0.5 + 0.5)
        images.save("images_" + str(i) + ".jpg")
        
        gt = to_pil(gt* 0.5 + 0.5)
        gt.save("gt_" + str(i) + ".jpg")

        # roughnesses = make_grid(roughnesses[0], nrow=2, padding=0) * 0.5 + 0.5
        # save_image(roughnesses, "roughnesses_" + str(i) + ".jpg")

        # materials = make_grid(materials[0], nrow=2, padding=0) * 0.5 + 0.5
        # save_image(materials, "materials_" + str(i) + ".jpg")

        i += 1
        breakpoint()
        # if i > 30:
            # breakpoint()
