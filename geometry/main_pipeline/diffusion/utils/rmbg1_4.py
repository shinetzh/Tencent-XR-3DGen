import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation, pipeline
from torchvision.transforms.functional import normalize

# model_dir = "pretrain_ckpts/RMBG-1.4" #### need download RMBG-1.4 first
pretrained_weights_path = "Path of the Tencent-XR-3DGen"
model_dir = f"{pretrained_weights_path}/RMGB-1.4"

if not os.path.exists(model_dir):
    exit("Error: Please download RMBG-1.4 model first")

print("load rmbg1.4 from model_dir")
model = AutoModelForImageSegmentation.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def read_image(image_path):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(image_bgr.shape) == 2:
        image_rgb = np.stack([image_bgr]*3, axis=-1)
    elif len(image_bgr.shape) == 3:
        if image_bgr.shape[-1] == 1:
            image_rgb = np.concatenate([image_bgr]*3, axis=-1)
        elif image_bgr.shape[-1] == 3:
            image_rgb = image_bgr[:, :, ::-1]
        elif image_bgr.shape[-1] == 4:
            image_rgb = np.concatenate([image_bgr[:, :, :3][:, :, ::-1], image_bgr[:, :, -1:]], axis=-1)
        else:
            assert False, "image channel should be in [1, 3, 4]!"
        
    else:
        assert False, "image shape length should be 2 or 3!"
    image = Image.fromarray(image_rgb)
    return image

def rmbg14_preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def rmbg14_postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

def rmbg(image):
    if isinstance(image, str) and os.path.exists(image):
        image = read_image(image)
    elif isinstance(image, str) and not os.path.exists(image):
        print(f"{image} not exists")
        return

    orig_im_array = np.array(image)
    orig_im_size = orig_im_array.shape[0:2]
    model_input_size = [1024,1024]
    image = rmbg14_preprocess_image(orig_im_array, model_input_size).to(device)

    # inference 
    result=model(image)
    # post process
    mask = rmbg14_postprocess_image(result[0][0], orig_im_size)
    rgba_array = np.concatenate([orig_im_array, np.array(mask)[..., None]], axis=-1)
    return rgba_array


def bbox_clip(bbox, h, w):
    xmin, ymin, xmax, ymax = bbox
    xmin = 5 if (xmin < 5) else xmin
    ymin = 5 if (ymin < 5) else ymin
    xmax = w - 5 if (xmax > w-5) else xmax
    ymax = h - 5 if (ymax > h-5) else ymax

    return [xmin, ymin, xmax, ymax]


def process_rgb(image, bg_color, wh_ratio=0.8, rmbg_type="1.4"): 
    """
    rmbg_type: "1.4" | "1.0" | "sam"
    """
    print(f"rmbg_type: {rmbg_type}")
    if isinstance(image, str):
        try:
            image = read_image(image)
        except:
            return -1
    if image.mode == "RGBA":
        width = image.width
        height = image.height
        imgbg = Image.new('RGB', size=(width, height), color=(255, 255, 255))
        imgbg.paste(image, (0, 0), mask=image)
        image_rgb = imgbg
    else:
        image_rgb = image

    image_out = rmbg(image_rgb)
    image_out = np.clip(image_out, 0, 255)
    image_seg = Image.fromarray(image_out.astype(np.uint8), mode="RGBA")

    return repadding_rgba_image(image_seg, ratio=wh_ratio, bg_color=bg_color)

def repadding_rgba_image(image, ratio=0.8, bg_color=255):
    if isinstance(image, str):
        try:
            image = read_image(image)
        except:
            return -1
    in_w = image.width
    in_h = image.height

    mask_array = np.array(image)[:, :, -1]
    mask_array[mask_array < 127] = 0
    mask_array[mask_array >= 127] = 255
    x, y, w, h = cv2.boundingRect(mask_array)
    white_bg_image = Image.new('RGB', size=(in_w, in_h), color=(bg_color, bg_color, bg_color))
    white_bg_image.paste(image, (0, 0), mask=image)

    max_size = max(w, h)
    side_len = int(max_size / ratio)
    padded_image = np.ones((side_len, side_len, 3), dtype=np.uint8) * bg_color
    mask = np.zeros((side_len, side_len, 1), dtype=np.uint8)

    center =  side_len // 2

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


def preprocess_image(image, bg_color=255, wh_ratio=0.8, rmbg_type="1.4", rmbg_force=True):
    if isinstance(image, str):
        image = read_image(image)
    
    if image.mode == "RGBA" and rmbg_force:
        print("force remove bg")
        width = image.width
        height = image.height
        imgbg = Image.new('RGB', size=(width, height), color=(255, 255, 255))
        imgbg.paste(image, (0, 0), mask=image)
        image = imgbg
        rgba = process_rgb(image, bg_color, wh_ratio, rmbg_type=rmbg_type)
    elif image.mode == "RGBA" and (np.unique(np.array(image)[..., -1]).shape[0] > 1):
        print("recognize as rgba image")
        rgba = repadding_rgba_image(image, ratio=wh_ratio, bg_color=bg_color)
    else:
        print("recognize as rgb img")
        rgba = process_rgb(image, bg_color, wh_ratio, rmbg_type=rmbg_type)


    ### use lighting_fast
    # rgb_lighting = lighting_fast(rgba[..., :-1], light=-25, mask_img=rgba[..., -1])
    # rgb_balance = white_balance(rgb_lighting, rgba[..., -1])
    rgb_lighting = rgba[..., :-1]
    try:
        image_rgb = Image.fromarray(rgb_lighting)
        image_mask = Image.fromarray(rgba[:, :, -1])
        w = image_mask.width
        h = image_mask.height

        max_forground_ratio = 0.68
        if np.sum(rgba[:, :, -1]) / (h * w * 255) > max_forground_ratio:
            forground_ratio = np.sum(rgba[:, :, -1]) / (h * w * 255)
            wh_ratio_new = 0.5 + (wh_ratio - 0.5) * (1.0 - (forground_ratio - max_forground_ratio)/0.2)
            print(f"forground ratio:{forground_ratio} is too big, rescale from {wh_ratio} to {wh_ratio_new}")
            return preprocess_image(image, bg_color=bg_color, wh_ratio=wh_ratio_new, rmbg_type=rmbg_type, rmbg_force=rmbg_force)

    except:
        print(f"segmentation failed...")
        image_rgb = image.convert("RGB")
        image_mask = Image.fromarray((np.ones((image.size[1], image.size[0])) * 255).astype(np.uint8))
    
    return image_rgb, image_mask

def process_image_path(image_path, bg_color=255, wh_ratio=0.8, rmbg_type="1.4", rmbg_force=True):
    return preprocess_image(image_path, bg_color, wh_ratio, rmbg_type=rmbg_type, rmbg_force=rmbg_force)

def get_mask_center(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return int(x+w/2.0), int(y+h/2.0), w, h

def rescale_mask_img(image_out, origin_scale, new_scale, center_align=False):
    h, w, c = image_out.shape

    pad_h = int((h * origin_scale / new_scale - h) / 2)
    pad_w = int((w * origin_scale / new_scale - w) / 2)

    rgb = image_out[..., :3]
    mask = image_out[..., 3]
    rgb = np.pad(rgb, ((pad_h,pad_h),(pad_w,pad_w), (0,0)), mode='constant', constant_values=255.0)
    if not center_align:
        print("with out center_align")
        return rgb
    mask = np.pad(mask, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant', constant_values=0.0)
    rgb_mask = np.concatenate([rgb, mask[..., None]], axis=-1)
    rgb_mask_resized = cv2.resize(rgb_mask, (w, h))

    x0, y0, _, _ = get_mask_center(image_out[..., -1])
    x1, y1, w_obj, h_obj = get_mask_center(rgb_mask_resized[..., -1])

    final_image = np.ones((h, w, 3)) * 255.0
    final_image_mask = np.zeros((h, w, 1))
    final_image_rgba = np.concatenate([final_image, final_image_mask], axis=-1)

    final_image_rgba[
        y0 - h_obj // 2 : y0 - h_obj // 2 + h_obj, x0 - w_obj // 2 : x0 - w_obj // 2 + w_obj
    ] = np.array(rgb_mask_resized)[y1 - h_obj // 2 : y1 - h_obj // 2 + h_obj, 
                                   x1 - w_obj // 2 : x1 - w_obj // 2 + w_obj]
    return final_image_rgba

def change_img_background_simple(images_array, rescale=False, origin_scale=0.9, new_scale=0.8, bg_color=255):
    channel_second = False
    if images_array.shape[1] == 3:
        channel_second = True
        images_array = np.transpose(images_array, (0, 2, 3, 1))
    images_array_out = []
    for image in images_array:
        rgba_array = rmbg(image)
        mask_float = rgba_array[:, :, -1:] / 255
        image_out = rgba_array[:, :, :3] * mask_float + 255 * (1-mask_float)
        if rescale:
            image_out = rescale_mask_img(image_out.clip(0, 255).astype(np.uint8), origin_scale, new_scale)
        image_out = image_out[..., :3]
        images_array_out.append(image_out)
    if channel_second:
        return np.stack(images_array_out, axis=0).transpose(0, 3, 1, 2)
    else:
        return np.stack(images_array_out, axis=0)

def get_rgba_list_ratio(rgba_list, wh_ratio=0.9):
    wh_list = []
    h_ratio = 10
    for rgba in rgba_list:
        in_h, in_w, _  = rgba.shape
        mask_array = rgba[:, :, -1]
        mask_array[mask_array < 127] = 0
        mask_array[mask_array >= 127] = 255
        x, y, w, h = cv2.boundingRect(mask_array)
        max_size = max(w, h)
        if max_size > h:
            h_ratio_new = h / (max_size / wh_ratio)
            if h_ratio_new < h_ratio:
                h_ratio = h_ratio_new
        wh_list.append([w, h])
    if h_ratio == 10:
        return [wh_ratio] * len(rgba_list)
    wh_ratio_list = []
    for wh in wh_list:
        w, h = wh
        wh_ratio_list.append(max(w, h) / (h / h_ratio))
    return wh_ratio_list


def process_image_path_list(image_list, bg_color=255, wh_ratio=0.8, rmbg_type="1.4"):
    rgba_list = []
    for image in image_list:
        if isinstance(image, str):
            image = read_image(image)
        
        ## check if have alpha channel and the alpha channel is valid
        if image.mode == "RGBA" and (np.unique(np.array(image)[..., -1]).shape[0] > 1):
            print("recognize as rgba image")
            rgba = repadding_rgba_image(image, ratio=wh_ratio, bg_color=bg_color)
        else:
            print("recognize as rgb img")
            rgba = process_rgb(image, bg_color, wh_ratio, rmbg_type=rmbg_type)
        rgba_list.append(rgba)
    wh_ratio_list = get_rgba_list_ratio(rgba_list, wh_ratio=wh_ratio)
    print(wh_ratio_list)
    result_list = []
    for image, wh_ratio in zip(image_list, wh_ratio_list):
        result_list.append(process_image_path(image, bg_color=bg_color, wh_ratio=wh_ratio, rmbg_type=rmbg_type))
    return result_list
