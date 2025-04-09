try:
    from .utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image, to_white_bg
except:
    try:
        from sam_preprocess.utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image, to_white_bg
    except:
        from utils import image_preprocess, pred_bbox, sam_init, sam_out_nosave, resize_image, to_white_bg
import os
import torch
import numpy as np
from PIL import Image
import cv2

# load SAM checkpoint
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
# sam_predictor = sam_init("/aigc_cfs_gdp/neoshang/models/sam/sam_vit_h_4b8939.pth", gpu)
# sam_predictor = sam_init("/aigc_cfs_2/neoshang/models/sam/sam_vit_h_4b8939.pth", gpu)
sam_predictor = sam_init("/aigc_cfs/xibinsong/models/sam_model/sam_vit_h_4b8939.pth")
sam_predictor.model.eval()
print("load sam ckpt done.")


def bbox_clip(bbox):
    bbox_clip = []
    for x in bbox:
        if x < 10:
            x = 10
        elif x > 500:
            x = 500
        bbox_clip.append(x)
    return [x for x in bbox_clip]
    

def process_image(image, bg_color, wh_ratio=0.8, save_samout=False, use_sam=False, center=True):
    if isinstance(image, str):
        try:
            input_raw = Image.open(image)
        except:
            return -1
    else:
        input_raw = image
    input_raw = resize_image(input_raw, max(input_raw.size))
    image_rgb = input_raw.convert("RGB")
    # image_rgb.save("test.png")
    # breakpoint()
    with torch.no_grad():
        bbox = pred_bbox(input_raw)
        bbox = bbox_clip(bbox)

    if not use_sam:
        image_out = to_white_bg(np.array(image_rgb), return_rgba=True) * 255
        image_out = np.clip(image_out, 0, 255)
        image_seg = Image.fromarray(image_out.astype(np.uint8), mode="RGBA")
    else:
        #### samout
        image_seg = sam_out_nosave(
            sam_predictor, image_rgb, bbox
        )

    # #### rmbg
    # image_out = to_white_bg(np.array(image_rgb), return_rgba=True) * 255
    # image_out = np.clip(image_out, 0, 255)
    # image_sam = Image.fromarray(image_out.astype(np.uint8), mode="RGBA")

    image_processed_rgba = image_preprocess(image_seg, save_samout=save_samout, lower_contrast=False, rescale=True, bg_color=bg_color, wh_ratio=wh_ratio, center=center)
    return image_processed_rgba


def repadding_rgba_image(image, rescale=True, ratio=0.8, bg_color=255, center=True):
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except:
            return -1
    in_w = image.width
    in_h = image.height
    x, y, w, h = cv2.boundingRect(np.array(image)[:, :, -1])

    white_bg_image = Image.new('RGB', size=(in_w, in_h), color=(bg_color, bg_color, bg_color))
    white_bg_image.paste(image, (0, 0), mask=image)
    if not center:
        rgba_image = np.concatenate([np.array(white_bg_image), np.array(image)[..., -1:]], axis=-1)
        rgba_image = rgba_image.astype(np.uint8)
        return rgba_image
    max_size = max(w, h)
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.ones((side_len, side_len, 3), dtype=np.uint8) * bg_color
    mask = np.zeros((side_len, side_len, 1), dtype=np.uint8)
    center = side_len // 2

    padded_image[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = np.array(white_bg_image)[y : y + h, x : x + w]

    mask[
        center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w
    ] = np.array(image)[..., -1:][y : y + h, x : x + w]

    rgba_image = np.concatenate([padded_image, mask], axis=-1)

    return rgba_image


def lighting_fast(img, light, mask_img=None):
    assert -100 <= light <= 100
    max_v = 4
    bright = (light/100.0)/max_v
    mid = 1.0+max_v*bright
    print('bright: ', bright, 'mid: ', mid)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    thresh = gray * gray * (mask_img.astype(np.float32) / 255.0)
    t = np.mean(thresh, where=(thresh > 0.2))

    mask = np.where(thresh > t, 255, 0).astype(np.float32)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = cv2.dilate(mask, kernel, iterations=3)
    mask[mask_img==0] = 0
    # cv2.imwrite("mask4.png", mask)
    brightrate = np.where(mask == 255.0, bright, (1.0/t*thresh)*bright)
    mask = np.where(mask == 255.0, mid, (mid-1.0)/t*thresh+1.0)
    img_float = img/255.0
    img_float = np.power(img_float, 1.0/mask[:, :, np.newaxis])*(1.0/(1.0-brightrate[:, :, np.newaxis]))
    img_float = np.clip(img_float, 0, 1.0)*255.0
    return img_float.astype(np.uint8)

def white_balance(img_rgb, img_mask):
    r, g, b = cv2.split(img_rgb)  #图像bgr通道分离
    img_mask_float = img_mask / 255.0

    avg_b = np.average(b, weights=(img_mask_float/255.0))
    avg_g = np.average(g, weights=(img_mask_float/255.0))
    avg_r = np.average(r, weights=(img_mask_float/255.0))
    
    k = (avg_b+avg_g+avg_r)/3  #计算k值
    
    kr = k/avg_r #计算rgb的增益(增益通常在0-2的浮点数之间)
    kb = k/avg_b
    kg = k/avg_g
    
    #根据增益逐个调整RGB三通道的像素值，超出部分取255（数据类型还是要转换回uint8）
    new_b = np.where((kb * b) > 255, 255, kb * b).astype(np.uint8)
    new_g = np.where((kg * g) > 255, 255, kg * g).astype(np.uint8)
    new_r = np.where((kr * r) > 255, 255, kr * r).astype(np.uint8)
    
    # 合并三个通道
    img_new = cv2.merge([new_r, new_g, new_b])
    img_new[img_mask==0] = 255

    return img_new


def preprocess_image(image, bg_color=255, wh_ratio=0.8, save_samout=False, use_sam=False, center=True):
    if isinstance(image, str):
        image = Image.open(image)
    
    ## check if have alpha channel and the alpha channel is valid
    if image.mode == "RGBA" and (np.unique(np.array(image)[..., -1]).shape[0] > 5):
        rgba = repadding_rgba_image(image, rescale=True, ratio=wh_ratio, bg_color=bg_color, center=center)
    rgba = process_image(image, bg_color, wh_ratio, save_samout=save_samout, use_sam=use_sam, center=center)

    # Image.fromarray(rgba).save("rgba.png")

    ### use lighting_fast
    # rgb_lighting = lighting_fast(rgba[..., :-1], light=-25, mask_img=rgba[..., -1])
    # rgb_balance = white_balance(rgb_lighting, rgba[..., -1])
    rgb_lighting = rgba[..., :-1]
    image_rgb = Image.fromarray(rgb_lighting)
    image_mask = Image.fromarray(rgba[:, :, -1])
    ##### concat for debug
    # image_concat = Image.fromarray(np.concatenate([rgba[..., :3], rgb_lighting], axis=1))
    # return image_rgb, image_concat

    return image_rgb, image_mask

def process_image_path(image_path, bg_color=255, wh_ratio=0.8, save_samout=False, use_sam=False, center=True):
    return preprocess_image(image_path, bg_color, wh_ratio, save_samout=save_samout, use_sam=use_sam, center=center)


if __name__ == "__main__":
    # from tqdm import tqdm
    # image_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/lora_muchangwuyu_v0.0.0/test_lora/2024-01-24-10:17:10"
    # save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_images_tmp1"
    # os.makedirs(save_dir, exist_ok=True)
    # for imagename in tqdm(os.listdir(image_dir)):
    #     if not imagename == "test009_002.png":
    #         continue
    #     print(imagename)
    #     image_path = os.path.join(image_dir, imagename)
    #     save_path = os.path.join(save_dir, imagename)
    #     image_processed = process_image(image_path)
    #     image_processed.save(save_path)

    image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_quality/001_copy_4.png"
    image = preprocess_image(image_path)
    image.save("test.png")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_path", required=True)
#     parser.add_argument("--save_path", required=True)
#     parser.add_argument("--ckpt_path", default="/aigc_cfs_2/neoshang/models/sam/sam_vit_h_4b8939.pth")
#     args = parser.parse_args()

#     # load SAM checkpoint
#     gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
#     sam_predictor = sam_init(args.ckpt_path, gpu)
#     print("load sam ckpt done.")

#     input_raw = Image.open(args.image_path)
#     # input_raw.thumbnail([512, 512], Image.Resampling.LANCZOS)
#     input_raw = resize_image(input_raw, 512)
#     image_sam = sam_out_nosave(
#         sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
#     )

#     image_preprocess(image_sam, args.save_path, lower_contrast=False, rescale=True)