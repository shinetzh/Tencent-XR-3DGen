import torch
from torch import nn
import numpy as np

from .model.smplx import SMPLX
from .utils import rotation_converter, util
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from .model.siren import  ParamNet
class SMPL_with_scale (nn.Module):

    def __init__(self, cfg, init_beta=None):
        super(SMPL_with_scale, self).__init__()
        self.cfg = cfg

        # -- define canonical space
        pose = torch.zeros([55, 3], dtype=torch.float32)  # 55
        angle = 90 * np.pi / 180.
        pose[0, 0] = angle
        canonical_pose_matrix = rotation_converter.batch_euler2matrix(pose)

        self.smplx = SMPLX(self.cfg.model)
        self.smplx.set_canonical_space(canonical_pose_matrix)


    def to(self, device):
        super().to(device)
        self.smplx.to(device)
        return self



    def _2_canonical_verts(self, data, posed_verts):
        '''
        data :  body_pose
                global_pose
                betas
                transl

        '''

        body_pose = rotation_converter.batch_euler2matrix(data["body_pose"].reshape(21, 3))  # 21,3,3
        global_pose = rotation_converter.batch_euler2matrix(data["global_pose"])  # 21,3,3

        smplx = self.smplx

        # base_verts = smplx.verts[None, ...]

        # get transform from given beta and pose
        verts_transform = smplx.backward_skinning(
            full_pose=None,
            shape_params=data["betas"],
            body_pose=body_pose[None],
            global_pose=global_pose[None],
            transl=data["transl"]
        )

        canoniocal_verts = util.batch_transform(verts_transform, posed_verts)

        return canoniocal_verts[0]







    def forward(self, data):

        body_pose = rotation_converter.batch_euler2matrix(data["body_pose"].reshape(21, 3))  # 21,3,3

        global_pose = rotation_converter.batch_euler2matrix(data["global_pose"])  # 21,3,3

        smplx = self.smplx

        posed_vertices, landmarks, joints = smplx.forward(
            shape_params =data["betas"],
            body_pose=body_pose[None],
            global_pose=global_pose[None],
            transl=data["transl"],
            head_scale=data["head_scale"],
            body_scale=data["body_scale"]
        )

        return smplx.faces, posed_vertices[0]





    def forward_skinning(self, data):


        body_pose = rotation_converter.batch_euler2matrix(data["body_pose"].reshape(21, 3))  # 21,3,3

        global_pose = rotation_converter.batch_euler2matrix(data["global_pose"])  # 21,3,3

        smplx = self.smplx

        verts_transform = smplx.forward_skinning(
            shape_params =data["betas"],
            body_pose=body_pose[None],
            global_pose=global_pose[None],
            transl=data["transl"],
            head_scale=data["head_scale"],
            body_scale=data["body_scale"]
        )





        return smplx.faces, smplx.v_template, verts_transform

import open3d as o3d


