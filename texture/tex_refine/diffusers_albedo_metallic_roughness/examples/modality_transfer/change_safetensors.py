
# import pickle
# import numpy as np
# import torch

# # 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
# with open('/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v2/checkpoint-40000/optimizer.bin','rb') as fr:
#     data = torch.load(fr, map_location='cpu')#可使用cpu或gpu
#     print(data)

# # data = np.load("/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v2/checkpoint-40000/random_states_0.pkl", allow_pickle=True)
# # print(data)

from regex import D
from safetensors import safe_open
import torch
import json
import numpy as np
from safetensors.torch import save_file

model_path = "/aigc_cfs_2/neoshang/models/stable-diffusion-2/unet/diffusion_pytorch_model.safetensors"
# model_path = "/aigc_cfs_2/neoshang/models/stable-diffusion-2/unet/diffusion_pytorch_model.fp16.safetensors"
# model_path = "/aigc_cfs_2/neoshang/models/zero123plus-v1.2_v2/unet_cond/diffusion_pytorch_model.safetensors"
save_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/modality_transfer/rgb2norm_v1/unet/diffusion_pytorch_model.safetensors"
weight_dtype = torch.float32

tensors = {}
with safe_open(model_path, framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

tensors["conv_in.weight"] = torch.cat([tensors["conv_in.weight"]]*2, dim=1)
save_file(tensors, save_path)



# import torch
# import torch.utils.checkpoint

# import sys
# sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")
# sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane/src")

# from diffusers import UNet2DConditionModel

# # unet_config = UNet2DConditionModel.load_config("/aigc_cfs_2/neoshang/models/zero123plus-v1.2/unet")
# # unet = UNet2DConditionModel(**unet_config).cuda()

# UNet2DConditionModel.from_pretrained("/aigc_cfs_2/neoshang/models/zero123plus-v1.2", subfolder="unet")

# # noisy_latents = torch.randn(4, 4, 16, 48).cuda()
# # timesteps = torch.tensor([99, 100, 200, 300], dtype=torch.long).cuda()
# # encoder_hidden_states = torch.randn(4, 197, 768).cuda()
# # output = unet(noisy_latents, timesteps, encoder_hidden_states)
# # print(output['sample'].shape)
# # import pdb;pdb.set_trace()