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

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--matches", type=str, default=None)
    parser.add_argument("--curve_matches", type=str, default=None)
    parser.add_argument("--reuse", type=str, default=None)
    parser.add_argument("--config", choices=["daz", "vroid", "base"], default=None)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    # if args.config == "daz" :
    #     from lib.configs.config_daz import get_cfg_defaults
    # else:
    # from lib.configs.config_daz import get_cfg_defaults
    from lib.configs.config_cartoon import get_cfg_defaults
    # from lib.configs.config_vroid import get_cfg_defaults



    cfg = get_cfg_defaults()
    cfg.mesh_path = args.mesh_path
    cfg.matches = args.matches
    cfg.curve_matches = args.curve_matches
    cfg.reuse = args.reuse

    reg = SMPLRegistration(config = cfg)


    data, warpped_mesh = reg.align_cartoon()


    # torch.save(data, reg.smplx_offset_dump_path)
    # o3d.io.write_triangle_mesh( os.path.join(
    #      str( pathlib.Path( reg.smplx_offset_dump_path).parent ), "warpped_smpl.obj") , warpped_mesh)
