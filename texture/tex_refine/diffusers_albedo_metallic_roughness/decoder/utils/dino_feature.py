import os
import time
import torch
import torch.utils.checkpoint

from transformers import ViTImageProcessor, ViTModel
from PIL import Image

from rpyc import Service
from rpyc.utils.server import ThreadedServer


image_processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path='facebook/dino-vitb16', local_files_only=True)
model = ViTModel.from_pretrained(pretrained_model_name_or_path='facebook/dino-vitb16', local_files_only=True)

image_norm = torch.nn.LayerNorm(768, elementwise_affine=False) # de facto instance norm
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(device)

def dino_feature(image_path):
    start_time=time.strftime('%Y-%m-%d-%H:%M:%S')
    print(f"{start_time}: request image_path: {image_path}")
    image = Image.open(image_path)
    width = image.width
    height = image.height
    img2 = Image.new('RGB', size=(width, height), color=(255, 255, 255))
    img2.paste(image, (0, 0), mask=image)
    image = img2
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    image_features = last_hidden_states

    print(image_features.shape)

    fea_map = image_features.cpu().detach().numpy()
    end_time=time.strftime('%Y-%m-%d-%H:%M:%S')
    print(f"{end_time}: success")
    return fea_map

class CharacterService(Service):
    def __init__(self) -> None:
        super().__init__()

    def exposed_dino_feature(self, image_path):
        feature = dino_feature(image_path)
        return feature

if __name__ == "__main__":
    service = ThreadedServer(CharacterService, port=8080)
    print("dino feature service start")
    service.start()
    # image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdfcolor_objaverse_kl_v0.0.0/triplane_2023-12-22-15:39:58/objaverse/1e0b0d7b929b4b22be233aa896267b7a_manifold_full_output_512_MightyWSB/0000/cam-0031.png"
    # feature = dino_feature(image_path)
    ## breakpoint()
