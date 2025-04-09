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


class SdfNogtTriplaneLoader(Dataset):

    def __init__(self, config, return_filename=False):
        super(SdfNogtTriplaneLoader, self).__init__()
        
        self.config = config = edict(config)
        self.points = []
        self.masks = []
        self.normals = []
        self.psdsdfs = []
        self.return_filename = return_filename

        # select grid points with resolution {config.psd_factor}
        X = torch.linspace(-1., 1., config.psd_factor)
        Y = torch.linspace(-1., 1., config.psd_factor)
        Z = torch.linspace(-1., 1., config.psd_factor)
        xx, yy, zz = torch.meshgrid(X, Y, Z)
        self.psdxyz = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
        self.psdxyz_num = len(self.psdxyz)

        # parse objects list
        if isinstance(config.objects, str):
            if config.objects.endswith('.txt'):
                with open(config.objects, 'r') as ftxt:
                    self.objnames = [line.strip('\n') for line in ftxt.readlines()]
            else:
                 self.objnames = [config.objects]
        elif isinstance(config.objects, list):
             self.objnames = config.objects

        self.nobjs = len(self.objnames)
        self.bsize = config.bsize
        self.obj_iters = [0] * self.nobjs
        print(f"{self.nobjs} objs in total")

    def get_one_sample(self, idx, sample_num=16384, surface_samplenum=10238):
        # load objects
        objname = self.objnames[idx]

        # surface points
        sp = torch.from_numpy(np.load(os.path.join(objname, f"dpcp_{self.config.factor}.npy"))).float()
        # surface normals
        sn = torch.from_numpy(np.load(os.path.join(objname, f"dpcn_{self.config.factor}.npy"))).float()
        # empty points
        ep = torch.from_numpy(np.random.uniform(-1., 1., size=(sp.shape[0], 3))).float()
        # pseduo sdfs
        psdsdfs = torch.from_numpy(np.load(os.path.join(objname, f"psdsdf_{self.config.psd_factor}.npy"))).float()

        points = torch.cat([sp, ep], 0)
        normals = torch.cat([sn, torch.zeros_like(ep)], 0)
        masks = torch.ones(points.shape[0]).bool()
        masks [len(sp):] = False
        perm = np.random.permutation(points.shape[0])[:sample_num]

        points = points[perm]
        masks = masks[perm]
        normal = normals[perm]

        points_surface = sp[np.random.permutation(sp.shape[0])[:surface_samplenum]]
        
        try:
            triplane_path = os.path.join(os.path.join(objname, "triplane.pt"))
            triplane = torch.load(triplane_path)
        except:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        out = [points_surface, self.psdxyz, points, masks, normal, psdsdfs, triplane]

        if self.return_filename:
            out.append(objname)

        return out

    def __len__(self):
        return len(self.objnames)
    
    def get_len(self, objid):
        return int(self.points[objid].shape[0] // self.bsize)


    def perm(self):
        perm = np.random.permutation(self.points.shape[0])
        self.points = self.points.detach()[perm].requires_grad_()
        self.masks = self.masks[perm]
        self.normals = self.normals[perm]

    def __getitem__(self, objid):
        return self.get_one_sample(objid)

    def save_objnames(self, savedir):
        with open(savedir, "w") as f:
            for fname in self.objnames:
                f.writelines(fname+'\n')