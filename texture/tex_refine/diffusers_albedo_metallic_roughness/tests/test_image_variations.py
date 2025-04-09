from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms
import numpy as np

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "/aigc_cfs_2/neoshang/models/sd-image-variations-diffusers",
    revision="v2.0",
    local_files_only=True
  )

sd_pipe = sd_pipe.to(device)


tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])

im = Image.open("/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation/mario.png")
im = im.convert("RGB")
inp = tform(im).to(device).unsqueeze(0)

out = sd_pipe(inp, guidance_scale=3)
out["images"][0].save("result.jpg")