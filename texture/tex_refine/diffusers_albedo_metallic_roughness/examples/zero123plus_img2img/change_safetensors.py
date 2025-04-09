from safetensors import safe_open
import torch
import json
import numpy as np
from safetensors.torch import save_file

# wonder3d_model_path = "/aigc_cfs_2/neoshang/models/wonder3d_pretrain_ckpt/wonder3d-v1.0/unet/diffusion_pytorch_model.bin"
# wonder3d_state_dict = torch.load(wonder3d_model_path, map_location="cpu")

stable_diffusion_model_path = "/aigc_cfs_2/neoshang/models/stable-diffusion-2/unet/diffusion_pytorch_model.safetensors"
model_save_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus_img2img/v2.0/unet/diffusion_pytorch_model.safetensors"

tensors = {}
with safe_open(stable_diffusion_model_path, framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

tensors["conv_in.weight"] = torch.cat([tensors["conv_in.weight"], torch.zeros_like(tensors["conv_in.weight"])], dim=1)
print(tensors["conv_in.weight"].shape)

config_path = "/aigc_cfs_2/neoshang/models/zero123plus-v1.2/model_index.json"
with open(config_path, 'r') as fr:
    model_index_config = json.load(fr)

ramping_coefficients = np.array(model_index_config["ramping_coefficients"])[..., None]
ramping_coefficients_tensor = torch.tensor(ramping_coefficients, dtype=torch.float32, device="cpu")
tensors["ramping_coefficients_weight"] = ramping_coefficients_tensor

class_embedding_dict = {
    "class_embedding.linear_1.weight": [1280, 6],
    "class_embedding.linear_1.bias": [1280],
    "class_embedding.linear_2.weight": [1280, 1280],
    "class_embedding.linear_2.bias": [1280],
}

for key, weight_shape in class_embedding_dict.items():
    if "bias" in key:
        weight = torch.zeros(weight_shape, dtype=torch.float32)
    else:
        weight = torch.randn(weight_shape, dtype=torch.float32)
    tensors[key] = weight

save_file(tensors, model_save_path)



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