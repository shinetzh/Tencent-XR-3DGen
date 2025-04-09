import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image

# load pipeline
controlnet = SD3ControlNetModel.from_pretrained("/aigc_cfs_2/neoshang/models/SD3-Controlnet-Tile", local_files_only=True)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "/aigc_cfs_2/neoshang/models/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet,
    local_files_only=True
)
pipe.to("cuda", torch.float16)

# config

image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/test_tiger1.jpg"


control_image = load_image(image_path)
prompt = '4views, multiview. front, back in first row, left, right in second row. tiger'
n_prompt = 'NSFW, nude, naked, porn, ugly'
# n_prompt = "sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"
image = pipe(
    prompt, 
    negative_prompt=n_prompt, 
    control_image=control_image, 
    controlnet_conditioning_scale=0.5,
).images[0]
image.save('image.jpg')