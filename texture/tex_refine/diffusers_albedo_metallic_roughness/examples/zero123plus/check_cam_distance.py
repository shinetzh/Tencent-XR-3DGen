import os
import json
import numpy as np

cam_json_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/render_512_Valour/cam_parameters.json"

with open(cam_json_path, 'r') as fr:
    all_cam_dict = json.load(fr)

print(len(all_cam_dict.keys()))
for i, (cam, cam_dict) in enumerate(all_cam_dict.items()):
    if i >= 8:
        break
    cam_pose = np.array(cam_dict["pose"])
    print(np.linalg.norm(cam_pose[:3, 3], ord=2))