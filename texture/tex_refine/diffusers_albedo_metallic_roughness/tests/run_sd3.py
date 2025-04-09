import torch
from diffusers import StableDiffusion3Pipeline
import json
import os

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "/aigc_cfs_2/neoshang/models/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    local_files_only=True
)
pipe = pipe.to("cuda")

with open("/aigc_cfs_3/weixuansun/data_list/caption_data_20240625.json", 'r') as fr:
    json_dict = json.load(fr)

save_dir = "/aigc_cfs_3/neoshang/data/sd3_generate"
os.makedirs(save_dir, exist_ok=True)

num = 0
for classname, classdict in json_dict.items():
    for objname, objdict in classdict.items():
        caption_list = []
        if "caption" in objdict:
            caption_list = caption_list + objdict["caption"]
        if "cap3d" in objdict:
            caption_list.append(objdict["cap3d"])
        if "3dtopia" in objdict:
            caption_list.append(objdict["3dtopia"])
        if "shot_caption" in objdict:
            caption_list = caption_list + objdict["short_caption"]
        for caption in caption_list:
            image_save_path = os.path.join(save_dir, f"{str(num).zfill(8)}.png")
            text_save_path = os.path.join(save_dir, f"{str(num).zfill(8)}.json")

            if (caption == "") or (caption is None):
                continue
            try:
                image = pipe(
                    caption + " blender, 8k, masterpiece, figma, HD, UHD, clean background",
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                ).images[0]
            except:
                continue

            image.save(image_save_path)
            with open(text_save_path, 'w') as fw:
                save_dict = {"class": classname,
                          "objname": objname,
                          "caption": caption}
                json.dump(save_dict, fw, indent=2)
            num += 1