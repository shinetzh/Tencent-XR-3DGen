import sys
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane/src")
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# model_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/stable-diffusion-2"

model_dir = "/aigc_cfs_2/neoshang/models/zero123plus-v1.2_test"
# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_dir, scheduler=scheduler, torch_dtype=torch.float16)

pipe = pipe.to("cuda")

breakpoint()

prompt = "a woman sit on the beach, face front, beauty, full body, high quality, 4k"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")