import os
import glob
import json
import cv2

import torch
import numpy as np
import random

from easydict import EasyDict as edict
from torch.utils.data import Dataset
import dataloader.util_dmtet as util


###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path, depth_path=None, suffixs=['png', 'jpg']):
    if path.split('.')[-1] in suffixs:
        img_name = path
    else:
        files = glob.glob(path + '.*')
        assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
        img_name = files[0]
    # img = util.load_image_raw(img_name)
    # print('debug img ', img.shape, img.dtype)
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4] uint8
    img[..., :3] = img[..., :3][..., ::-1]
    
    if img.dtype != np.float32:  # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    if img.shape[-1] == 3 and depth_path is not None:
        # append alpha
        depth = util.load_image_raw(depth_path)
        mask = depth > 0
        img = torch.cat([img, torch.tensor(mask, dtype=torch.float32).unsqueeze(-1)], dim=-1)
    return img

def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0,  0, 1)))
    return np.matmul(T,origin) #T * origin


class DatasetDmtet(Dataset):
    def __init__(self, specs, data_type='train', resample=True, stage3=False):
        super(DatasetDmtet, self).__init__()
        data_config = specs["data_config"]
        decoder_config = specs["decoder_config"]
        with open(decoder_config["config_json"], 'r') as fr:
            self.FLAGS = edict(json.load(fr))

        self.n_images_each_obj = 300
        self.data_type = data_type
        self.resample = resample
        self.batch_views = self.FLAGS.n_views
        self.stage3 = stage3
        self.dataset_json = data_config["dataset_json"]

        self.shaded_dataset_checked = []
        self.resolution, self.aspect, self.fovy, self.proj = None, None, None, None
        shaded_dataset_dirs_list = self.__check_all_dataset(self.dataset_json)
        self.train_list, self.test_list = self.__split_train_test(shaded_dataset_dirs_list)

        if self.stage3:
            self.image_latent_dir = data_config["image_latent_dir"]
            self.random_image_index_list = [50, 53, 58, 61, 63, 66, 69, 71, 74, 79, 82, 84, 90, 95, 100, 105, 116, 129, 142, 171, 189]

        if self.data_type == "train":
            self.data_list = self.train_list
        elif self.data_type == "test":
            self.data_list = self.test_list

        if "max_num_train_obj" in data_config:
            max_num_train_obj = data_config["max_num_train_obj"]
            self.data_list = self.data_list[:max_num_train_obj]

        print("data obj nums: {}".format(len(self.data_list)))
        print("{} views per obj".format(self.batch_views))

    # def __split_train_test(self, shaded_dataset_dirs_list, test_threshold=0.01, test_min_num=10):
    #     train_list, test_list = [], []
    #     for shaded_dataset_dirs in shaded_dataset_dirs_list:
    #         num = len(shaded_dataset_dirs)
    #         test_num = int(max(num * test_threshold, test_min_num))
    #         test_list += shaded_dataset_dirs[0:test_num]
    #         train_list += shaded_dataset_dirs[test_num:]
    #     return train_list, test_list

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


    def __check_all_dataset(self, dataset_json):
        shaded_dirs_list = []
        with open(dataset_json, 'r') as fr:
            dataclass_render_dir_tripath = json.load(fr)
        for dataclass, render_dir_tripath_dict in dataclass_render_dir_tripath.items():
            shaded_dirs = self.__check_dataset(dataclass, render_dir_tripath_dict)
            if len(shaded_dirs) == 0:
                continue
            shaded_dirs_list.append(shaded_dirs)
        return shaded_dirs_list

    def __get_triplane_name(self, shaded_dir):
        if "vroid" in shaded_dir:
            obj_name = shaded_dir.split("/")[-2]
        else:
            obj_name = shaded_dir.split("/")[-1]
        return obj_name

    def __check_dataset(self, dataclass, render_dir_tripath_dict):
        print("checking dataset: {}".format(dataclass))
        shaded_dirs_checked = []
        if not self.__check_cam_config(dataclass, render_dir_tripath_dict):
            print("checked data:{}\nshaded configs are not same:{}\n".format(self.shaded_dataset_checked, dataclass))
            exit(0)

        shaded_dir_list = list(render_dir_tripath_dict.keys())
        shaded_dir_list.sort()
        for shaded_dir in shaded_dir_list:
            triplanepath = render_dir_tripath_dict[shaded_dir]
            if not os.path.exists(triplanepath):
                continue
            shaded_dirs_checked.append((dataclass, shaded_dir, triplanepath)) ## datalist 
        print("available num: {}".format(len(shaded_dirs_checked)))
        return shaded_dirs_checked

    def __check_cam_config(self, dataclass, render_dir_tripath_dict):
        shaded_dir_list = list(render_dir_tripath_dict.keys())
        shaded_dir_list.sort()
        shaded_cfg_path = None
        for shaded_dir in shaded_dir_list:
            shaded_cfg_path = os.path.join(shaded_dir, 'cam_parameters.json')
            if not os.path.exists(shaded_cfg_path):
                continue
            image_shaded_path = os.path.join(shaded_dir, "color", "cam-0000.png")
            resolution, aspect, fovy, proj = self.__parse_cam_intrinsic(shaded_cfg_path, image_shaded_path)
            if self.resolution is None:
                self.resolution, self.aspect, self.fovy, self.proj = resolution, aspect, fovy, proj
                self.shaded_dataset_checked.append(dataclass)
                return True
            elif (self.resolution == resolution) and \
                        (self.fovy == fovy) and \
                        (self.aspect == aspect) and \
                        self.proj.equal(proj):
                self.shaded_dataset_checked.append(dataclass)
                return True
            else:
                print("self.resolution: {} - resolution: {}\nself.fovy: {} - fovy: {}\nself.aspect: {} - aspect: {}\nself.proj: {} - proj: {}\n".format(
                    self.resolution, resolution,
                    self.fovy, fovy,
                    self.aspect, aspect,
                    self.proj, proj,))
                return False

    def __parse_cam_intrinsic(self, obj_cfg_path, image_shaded_path):
        """parse camera Intrinsic
        """
        image = cv2.imread(image_shaded_path)
        h, w, c = image.shape
        resolution = [h, w]
        aspect = w / h
        with open(obj_cfg_path, 'r') as fr:
            obj_cfg = json.load(fr)
        fovy = util.focal_length_to_fovy(obj_cfg['cam-0001']['k'][0][0], h)
        proj = util.perspective(fovy, aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        return resolution, aspect, fovy, proj

    def __parse_obj(self, obj_dir):
        
        img, mv, mvp, campos = self._parse_frame(obj_dir)
        if img is None:
            return None, None, None, None, None
        return img, mv, mvp, campos

    def _parse_frame(self, obj_dir, suffix_color='.png'):
        # Config projection matrix (static, so could be precomputed)
        # fov of vroid is 60 deg
        cam_parameters_path = os.path.join(obj_dir, 'cam_parameters.json')
        with open(cam_parameters_path, 'r') as fr:
            cam_config_dict = json.load(fr)
        image_number = len(cam_config_dict.keys())
        idxes_img = np.random.permutation(image_number)[:self.batch_views]
        img_list, mv_list, mvp_list, campos_list = [], [], [], []
        for idx_img in idxes_img:
            key_name = "cam-" + str(idx_img).zfill(4)
            k = np.array(cam_config_dict[key_name]['k'])
            pose = np.array(cam_config_dict[key_name]['pose'])
            color_path = os.path.join(obj_dir, 'color', key_name + suffix_color)
            if not os.path.exists(color_path):
                return None, None, None, None
            fovy = util.focal_length_to_fovy(k[0, 0], self.resolution[0])
            proj = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
            
            # Load image data and modelview matrix
            ## TODO(csz) future, if we have RGBA img, no need load depth img
            img    = _load_img(color_path)
            mv     = torch.linalg.inv(torch.tensor(opencv_to_blender(pose), dtype=torch.float32))
            mv     = mv @ util.rotate_x(-np.pi / 2) #TODO(csz) only about mesh output
            campos = torch.linalg.inv(mv)[:3, 3]
            mvp    = proj @ mv

            img = img[None, ...]
            mv = mv[None, ...]
            mvp = mvp[None, ...]
            campos = campos[None, ...]
            img_list.append(img)
            mv_list.append(mv)
            mvp_list.append(mvp)
            campos_list.append(campos)
        return torch.cat(img_list, dim=0), torch.cat(mv_list, dim=0), \
                torch.cat(mvp_list, dim=0), torch.cat(campos_list, dim=0)


    def rotate_scene(self, itr, itr_all=50, cam_radius=2.):
        """only for eval, render rotate circle imgs
        """
        # Smooth rotation for display.
        ### debug face
        ang    = (itr / itr_all) * np.pi * 2
        mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = self.proj @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        # make mvp [1, 1, 4, 4]  background to [1, nv=1, h, w, 3]
        idx_obj = torch.tensor(0, dtype=torch.long).unsqueeze(dim=0)
        return {
            'mv': mv[None, None, ...].cuda(),
            'mvp': mvp[None, None, ...].cuda(),
            'campos': campos[None, None, ...].cuda(),
            'resolution': self.FLAGS.train_res,
            'spp': self.FLAGS.spp,
            'background': torch.ones((1, 1, self.FLAGS.train_res[0], self.FLAGS.train_res[1], 3), dtype=torch.float32, device='cuda'),
            'img': None
        }    


    def __len__(self):
        # return self.n_objs_all 
        return len(self.data_list)

    def __getitem__(self, idx_obj):
        ## getitem means get obj with nv imgs
        # [1, nv, h, w, 3]
        dataclass, shaded_dir, triplanepath = self.data_list[idx_obj]
        obj_save_name = "==".join(shaded_dir.split('/')[-3:])
        img, mv, mvp, campos = self.__parse_obj(shaded_dir)

        if img is None:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        idx_obj = torch.tensor(idx_obj, dtype=torch.long).unsqueeze(dim=0)
        
        triplane = torch.load(triplanepath, map_location=torch.device('cpu'))

        
        result = {
                "dataclass" : dataclass,
                "obj_save_name" : obj_save_name, 
                "shaded_dir" : shaded_dir,
                'mv' : mv,
                'mvp' : mvp,
                'campos' : campos,
                "triplane" : triplane,
                'img' : img
            }

        if self.stage3:
            image_latent_idx = np.random.choice(self.random_image_index_list)
            image_latent_path = os.path.join(self.image_latent_dir, dataclass, obj_save_name, "latent_{}.npy".format(str(image_latent_idx).zfill(4)))
            latent_image = torch.from_numpy(np.load(image_latent_path)).float().squeeze()
            result.update({"latent_image" : latent_image})
        return result