import os, sys, glob, torch
import numpy as np
import torch
import random
import json
from torch.utils.data import Dataset
from easydict import EasyDict as edict

def get_rays(H, W, K, c2w, inverse_y):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()

    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


class SdfNogtLoader(Dataset):

    def __init__(self, config, return_filename=False, return_triplane=False):
        super(SdfNogtLoader, self).__init__()
        
        self.config = config = edict(config)
        self.points = []
        self.masks = []
        self.normals = []
        self.psdsdfs = []
        self.return_filename = return_filename
        self.return_triplane = return_triplane

        # parse objects list
        print("checking data list... \n")
        if isinstance(config.objects, str):
          if config.objects.endswith('.txt'):
              with open(config.objects, 'r') as ftxt:
                  self.objnames = [line.strip('\n') for line in ftxt.readlines()]
          elif os.path.exists(config.objects):
              self.objnames = []
              for root, subdir, files in os.walk(config.objects):
                  for filename in files:
                      if filename == f"dpcp_{self.config.factor}.npy":
                          self.objnames.append(root)
          else:
              print("not obj path list or obj dir")
              exit(1)
        elif isinstance(config.objects, list):
            self.objnames = []
            for objs_dir in config.objects:
              for filename in os.listdir(objs_dir):
                pc_path = os.path.join(objs_dir, filename, filename + "_output_transparent", f"dpcp_{self.config.factor}.npy")
                ep_path = os.path.join(objs_dir, filename, filename + "_output_transparent", f"psdpts_{self.config.psd_factor}.npy")
                if os.path.exists(pc_path) and os.path.exists(ep_path):
                    self.objnames.append(os.path.dirname(pc_path))
        else:
            print("not obj path list or obj dir")
            exit(1)

        if return_triplane:
            with open(config.objects_triplane_dict_path, 'r') as fr:
                self.obj_triplane = json.load(fr)
            self.triplane_dir = config.triplane_dir

        self.nobjs = len(self.objnames)
        self.obj_iters = [0] * self.nobjs
        print(f"{self.nobjs} objs in total")
        print("return triplane: {}".format(return_triplane))

    def get_one_sample(self, idx, sample_num=30000):
        # load objects
        objname = self.objnames[idx]

        # surface points
        sp = torch.from_numpy(np.load(os.path.join(objname, f"dpcp_{self.config.factor}.npy"))).float()
        if sp.shape[0] < sample_num:
            return self.get_one_sample(random.randint(0, self.__len__()-1))
        # surface normals
        sn = torch.from_numpy(np.load(os.path.join(objname, f"dpcn_{self.config.factor}.npy"))).float()

        # pseduo sdfs
        ep = torch.from_numpy(np.load(os.path.join(objname, f"psdpts_{self.config.psd_factor}.npy"))).float()
        psdsdfs = torch.from_numpy(np.load(os.path.join(objname, f"psdsdf_{self.config.psd_factor}.npy"))).float()

        perm_surface = np.random.permutation(sp.shape[0])[:sample_num]
        perm_empty = np.random.permutation(ep.shape[0])[:sample_num]

        points_surface = sp[perm_surface]
        normal_surface = sn[perm_surface]
        points_empty = ep[perm_empty]
        sdf_empty = psdsdfs[perm_empty]

        out = [points_surface, normal_surface, points_empty, sdf_empty]

        if self.return_triplane:
            triplane = self.get_triplane(objname)
            if triplane is None:
                return self.get_one_sample(random.randint(0, self.__len__()-1))
            out.append(self.get_triplane(objname))
        if self.return_filename:
            out.append(objname)

        return out

    def __len__(self):
        return len(self.objnames)

    def get_triplane(self, objpath):
        if objpath in self.obj_triplane:
            triplane_name = self.obj_triplane[objpath]
            triplane_path = os.path.join(self.triplane_dir, triplane_name + ".tar")
            triplane = torch.load(triplane_path).squeeze()
            return triplane
        else:
            return None

    def __getitem__(self, objid):
        return self.get_one_sample(objid)
