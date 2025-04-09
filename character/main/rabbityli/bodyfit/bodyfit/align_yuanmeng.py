import importlib.util
import os, sys
import argparse
import shutil
from glob import glob
import pathlib

import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.registration import SMPLRegistration
import open3d as o3d

from lib.utils.util import batch_transform


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--matches", type=str, default=None)
    parser.add_argument("--curve_matches", type=str,
                        default="/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/naked/curve_match.json")

    args = parser.parse_args()
    return args


class LayeredAvatar():
    def __init__(self, param_data,  smplxs,  trns=None ):

        self.transform = trns

        if type(param_data) == str or type(param_data) == pathlib.PosixPath:
            param_data = torch.load(param_data)
        self.faces, self.template, self.T = smplxs.forward_skinning(param_data)

        if self.transform is not None:
            self.transform = torch.from_numpy(self.transform).float().to(self.T.device)

        part_idx_dict = smplxs.smplx.get_part_index()
        face_idx = part_idx_dict['face']
        ndp_offset = param_data["offset"].view(1, -1, 3, 1).detach()
        ndp_offset[:, face_idx] = 0
        self.T[..., :3, 3:] = self.T[..., :3, 3:] + ndp_offset

        self.posed_verts = batch_transform(self.T, self.template)

if __name__ == '__main__':


    # args = parse_args()
    from lib.configs.config_yuanmeng import get_cfg_defaults




    cfg = get_cfg_defaults()
    # cfg.mesh_path = args.mesh_path
    # cfg.matches = args.matches

    cfg.mesh_path = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/naked/body.obj"
    cfg.matches = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/correspondence/"
    cfg.curve_matches ="/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/yuanmeng/naked/curve_match.json"



    reg = SMPLRegistration(config = cfg)

    data, warpped_mesh = reg.align_yuanmeng(viz=True)


    # exit(0)

    torch.save(data, reg.smplx_offset_dump_path )
    warped_mesh_path =  os.path.join( pathlib.Path( reg.smplx_offset_dump_path).parent, "warpped_smpl_test.obj")
    print( "warped_mesh", warped_mesh_path)
    o3d.io.write_triangle_mesh( warped_mesh_path, warpped_mesh)





    T = LayeredAvatar(data, reg.model)
    data = { "posed_verts": T.posed_verts, "faces": T.faces, "T": T.T }
    torch.save(data, os.path.join( pathlib.Path( reg.smplx_offset_dump_path).parent, "smplx_and_offset_smplified.npz") )