import math
import os
import json
from dataclasses import dataclass, field

import random
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer
import sys
import trimesh
import miniball

from vae import register
from vae.utils.base import Updateable
from vae.utils.config import parse_structured
from vae.utils.typing import *

from omegaconf import OmegaConf
from camera_utils import pose_generation

import h5py


try:
    import fvdb
    from fvdb.nn import LeakyReLU, Sigmoid, Linear, VDBTensor
except:
    pass

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R

def rotate(pcd, pcd_transpose_matrix):
    pcd_norm = np.ones((pcd.shape[0], 4), dtype=np.float32)
    pcd_norm[:, :3] = pcd
    pcd_norm = pcd_norm @ pcd_transpose_matrix.T
    pcd_rotate = pcd_norm[:, :3]
    return pcd_rotate

# rotate poincloud and normal
def rotate_pcd_normal(pcd, rotate_azimuth, rotate_elevation):
    geo_pcd_points = pcd[:, :3]
    geo_pcd_normal = pcd[:, 3:]

    x_rotate90 = np.array([[1, 0, 0, 0],
                           [0, 0, -1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],])

    x_rotate_inv90 = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1],])
    
    cam_pose0 = np.array([[ 1.,  0.,  0.,  0.],
                            [ 0.,  0.,  1., -3.],
                            [ 0., -1.,  0.,  0.],
                            [ 0.,  0.,  0.,  1.]], dtype=np.float32)
    
    # breakpoint()
    cam_pose = np.stack(pose_generation(rotate_azimuth, rotate_elevation, current_fov_list=[1])[1]).astype(np.float32)[0,:]

    pcd_transpose_matrix = cam_pose0 @ np.linalg.inv(cam_pose)
    pcd_transpose_matrix = x_rotate_inv90 @ pcd_transpose_matrix @ x_rotate90

    geo_pcd_points_rotate = rotate(geo_pcd_points, pcd_transpose_matrix)
    geo_pcd_normal_rotate = rotate(geo_pcd_normal, pcd_transpose_matrix)
    pcd_normal = np.concatenate([geo_pcd_points_rotate, geo_pcd_normal_rotate], axis=-1)
    return pcd_normal.astype(np.float32)


# rotate poincloud and normal
def rotate_rand_pcd(pcd, rotate_azimuth, rotate_elevation):

    x_rotate90 = np.array([[1, 0, 0, 0],
                           [0, 0, -1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],])

    x_rotate_inv90 = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1],])
    
    cam_pose0 = np.array([[ 1.,  0.,  0.,  0.],
                            [ 0.,  0.,  1., -3.],
                            [ 0., -1.,  0.,  0.],
                            [ 0.,  0.,  0.,  1.]], dtype=np.float32)
    
    cam_pose = np.stack(pose_generation(rotate_azimuth, rotate_elevation, current_fov_list=[1])[1]).astype(np.float32)[0,:]

    # breakpoint()
    pcd_transpose_matrix = cam_pose0 @ np.linalg.inv(cam_pose)
    pcd_transpose_matrix = x_rotate_inv90 @ pcd_transpose_matrix @ x_rotate90

    pcd = rotate(pcd, pcd_transpose_matrix)

    return pcd.astype(np.float32)

def calculate_scale_matrix(mesh_verts: np.array, standard_height: float = 1.92):
    original_mesh = trimesh.base.Trimesh(vertices=mesh_verts)
    hull_vertices = original_mesh.convex_hull.vertices
    bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
        hull_vertices)

    obj_center = bounding_sphere_C
    length = 2*math.sqrt(bounding_sphere_r2)
    scale = standard_height / length
    translation = -1 * obj_center
    transformation = np.array(
        [[scale, 0, 0, scale*translation[0]],
         [0, scale, 0, scale*translation[1]],
         [0, 0, scale, scale*translation[2]],
         [0, 0, 0, 1]]
    )
    return transformation


@dataclass
class ObjaverseDataModuleConfig:
    data_type: str = "occupancy"         # occupancy or sdf
    n_samples: int = 4096                # number of points in input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0             # noise level of the input point cloud
    
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "occupancy"  # occupancy, sdf, tsdf, tsdf_w_surface
    n_supervision: int = 4096           # number of points in supervision
    sampling_strategy: Optional[str] = "random"
    
    load_image: bool = False             # whether to load images 
    image_data_path: str = ""            # path to the image data
    image_type: str = "rgb"              # rgb, normal
    background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )
    idx: Optional[List[int]] = None      # index of the image to load
    n_views: int = 1                     # number of views
    rotate_points: bool = False          # whether to rotate the input point cloud and the supervision
    rotate_points_prob: float = 0.5          # probability of rotating 

    img_idx:  Optional[List[int]] = None      # index of the image to load (objaverse-mix)
    
    load_caption: bool = False           # whether to load captions
    caption_type: str = "text"           # text, clip_embeds
    tokenizer_pretrained_model_name_or_path: str = ""

    batch_size: int = 32
    num_workers: int = 0
    data_json: str = '/aigc_cfs_11/Asset/active_list/3d_diffusion/avatar/avatar_52k_with_normal_with_point_cloud.json'
    use_near: bool = False
    
    task: str = 'diffusion'
    
    view_group: int = 5
    group_size: int = 24
    img_size: int = 512
    
    view_cfg_prob: float = 0.1
    
    high_res_prob: float = 0.8
    chunk_size: int = 10000
    pointcloud_size: int = 500000
            
    load_voxel: bool = False
    voxel_resolution: int = 64

# take high-resolution point cloud
class ObjaverseDatasetHigh(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.split = split
        
        if self.cfg.load_caption:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.cfg.tokenizer_pretrained_model_name_or_path)

        self.background_color = torch.as_tensor(self.cfg.background_color)
        self.distance = 1.0
        
        with open(self.cfg.data_json, 'r') as fr:
            self.point_json = json.load(fr)
        
        self.key_list = []
        count = 0
        for class_name in self.point_json['data'].keys():
            class_dict = self.point_json['data'][class_name]
            for name, value_dict in class_dict.items():
                # check 
                h5_file_path = os.path.join(value_dict['GeoPcd'], 'sample.h5')
                if os.path.isfile(h5_file_path): # this might be slow if data is huge
                    count += 1
                    
                    self.key_list.append([class_name, name])
                    
        # this is a tmp solution for train/val split
        if len(self.key_list) > 100:
            train_amount = int(len(self.key_list)*0.998)
            if self.split == 'train':
                self.key_list = self.key_list[:train_amount]
            elif self.split == 'val':
                self.key_list = self.key_list[train_amount:]
        else:
            self.key_list = self.key_list

        print('data amount:', len(self.key_list))
            
        self.rotate_flag = False
        
    def __len__(self):
        return len(self.key_list)

    def _load_shape(self, index: int, surface=None, normal=None) -> Dict[str, Any]:
        if self.cfg.data_type == "occupancy":
            # for input point cloud
            class_name = self.key_list[index][0]
            obj_name = self.key_list[index][1]

            surface = np.concatenate([surface, normal], axis=1)
            if self.rotate_flag:
                surface = rotate_pcd_normal(surface, self.rotate_azimuth, self.rotate_elevation)
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")
                
        # random sampling
        if self.cfg.sampling_strategy == "random":
            rng = np.random.default_rng()
            ind = rng.choice(surface.shape[0], self.cfg.n_samples, replace=False)
            surface = surface[ind]
        elif self.cfg.sampling_strategy == "fps":
            import fpsample
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(surface[:, :3], self.cfg.n_samples, h=5)
            
            surface = surface[kdline_fps_samples_idx]
            # print(surface.shape)
        elif self.cfg.sampling_strategy is None:
            pass
        
        # rescale data
        surface[:, :3] = surface[:, :3] * self.cfg.scale # target scale
        # add noise to input point cloud
        surface[:, :3] += (np.random.rand(surface.shape[0], 3) * 2 - 1) * self.cfg.noise_sigma
        
        ret = {
            "uid": class_name + '_' + obj_name,
            "surface": surface.astype(np.float32),
        }
        # compute voxel
        if self.cfg.load_voxel:
            aa = -1
            bb = 1
            input_xyz = torch.from_numpy(surface[:, :3])
            input_xyz = (input_xyz - aa) / (bb - aa)
            index_points = (input_xyz * (self.cfg.voxel_resolution - 1)).long()
            index = index_points[:, 2] + self.cfg.voxel_resolution * (index_points[:, 1] + self.cfg.voxel_resolution * index_points[:, 0])
            voxels = torch.zeros(self.cfg.voxel_resolution**3).long()
            voxels[index.tolist()] = 1
            voxels = voxels.reshape(self.cfg.voxel_resolution, self.cfg.voxel_resolution, self.cfg.voxel_resolution)
            ret["voxel"] = voxels

        return ret

    def _load_shape_supervision(self, index: int, occupancies=None, rand_points=None) -> Dict[str, Any]:
        # for supervision
        ret = {}
        if self.cfg.data_type == "occupancy":
            class_name = self.key_list[index][0]
            obj_name = self.key_list[index][1]
            
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        if self.rotate_flag:
            rand_points = rotate_rand_pcd(rand_points, self.rotate_azimuth, self.rotate_elevation)
        
        # random sampling
        rng = np.random.default_rng()
        ind = rng.choice(rand_points.shape[0], self.cfg.n_supervision, replace=False)
        rand_points = rand_points[ind]
        rand_points = rand_points * self.cfg.scale
        ret["rand_points"] = rand_points.astype(np.float32)

        if self.cfg.data_type == "occupancy":
            assert self.cfg.supervision_type == "occupancy", "Only occupancy supervision is supported for occupancy data"
            occupancies = occupancies[ind]
            ret["occupancies"] = occupancies.astype(np.float32)

        else:
            raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret
    
    def get_data(self, index):
        # load shape
        class_name = self.key_list[index][0]
        obj_name = self.key_list[index][1]
        
        chunk_size=self.cfg.chunk_size
        max_chunk_idx = self.cfg.pointcloud_size // chunk_size
        
        # randomly rotate pont cloud
        if self.cfg.rotate_points:
            if random.random() < self.cfg.rotate_points_prob:
                self.rotate_azimuth, self.rotate_elevation =[random.randint(0,360)], [random.randint(-90, 90)]
                self.rotate_flag = True
            else:
                self.rotate_azimuth, self.rotate_elevation =[0, 0]
                self.rotate_flag = False
        
        h5_file = os.path.join(self.point_json['data'][class_name][obj_name]['GeoPcd'], 'sample.h5')
        
        # load point cloud data from h5 file
        with h5py.File(h5_file, 'r') as f:
            
            # read a chunk of point cloud to speed up
            index = random.randint(0, max_chunk_idx-1)
            chunk_start = index * chunk_size
            chunk_end = (index+1) * chunk_size
            surface = np.array(f["surface_points"][chunk_start:chunk_end,:])
            normal = np.array(f["surface_normals"][chunk_start:chunk_end,:])
            rand_points = np.array(f["space_points"][chunk_start:chunk_end,:])
            occupancies = np.array(f["space_occupancy"][chunk_start:chunk_end,:])[:,0]
            
            # alternatively read entire point cloud, depend on your h5 file format
            # surface = np.array(f["surface_points"])
            # normal = np.array(f["surface_normals"])
            # rand_points = np.array(f["space_points"])
            # occupancies = np.array(f["space_occupancy"])[:,0]

        ret = self._load_shape(index,surface,normal)

        # load supervision for shape
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index,rand_points=rand_points, occupancies=occupancies))
        
        ret['key'] = class_name + '-' + obj_name
        return ret
        
    def __getitem__(self, index):
        # return self.get_data(index)
        try:
            return self.get_data(index)
        except Exception as e:
            # print('error!')
            return self.__getitem__(np.random.randint(len(self)))


    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch
    
    

@register("objaverse-datamodule-high")
class ObjaverseDataModuleHigh(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDatasetHigh(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDatasetHigh(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseDatasetHigh(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
    
if __name__ == "__main__":
    cfg = parse_structured(ObjaverseDataModuleConfig)
    train_dataset = ObjaverseDatasetHigh(cfg, "train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 1, num_workers=4, pin_memory=False, shuffle=True)
    
    for data in train_dataloader:
        print('hh')
        pass
        
