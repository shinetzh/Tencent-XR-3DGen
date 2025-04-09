import json
import os

dir64 = "/apdcephfs_cq3/share_1615605/rabbityli/triplanes/head_occ/epoch_last_64"
dir256 = "/apdcephfs_cq3/share_1615605/rabbityli/triplanes/head_occ/epoch_last_256"
save_path = "/apdcephfs_cq3/share_1615605/neoshang/code/diffusers_triplane/data/liyang/head_occ_1102.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

data_dict = {}
data_dict["data"] = {}
data_dict["data"]["vroid"] = {}
for filename in os.listdir(dir64):
    data_dict["data"]["vroid"][filename.split('.')[0]] = {}
    path64 = os.path.join(dir64, filename)
    path256 = os.path.join(dir256, filename)
    data_dict["data"]["vroid"][filename.split('.')[0]]["path64"] = path64
    data_dict["data"]["vroid"][filename.split('.')[0]]["path256"] = path256

with open(save_path, "w") as fw:
    json.dump(data_dict, fw, indent=2)


