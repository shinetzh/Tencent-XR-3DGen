from PIL import Image
import numpy as np
import random
import cv2

def to_rgb_image(maybe_rgba: Image.Image, bg_color=127, edge_aug_threshold=0):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        # img = np.random.randint(random_grey_low, random_grey_high, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = np.ones([rgba.size[1], rgba.size[0], 3], dtype=np.uint8) * bg_color
        if edge_aug_threshold > 0 and (random.random() < edge_aug_threshold):
            mask_img = np.array(rgba.getchannel('A'))
            # threshold = np.random.randint(100, 156)
            mask_img[mask_img > 0] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            iterration_num = random.randint(1, 2)
            mask_img_small = cv2.erode(mask_img, kernel, iterations=iterration_num)
            mask_img_edge = mask_img - mask_img_small
            mask_img_edge = np.concatenate([mask_img_edge[..., None]]*3, axis=-1) / 255.0
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = Image.fromarray(img.astype(np.uint8), 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        img_array = np.array(img) * (1 - mask_img_edge) + rand_color * mask_img_edge
        img = Image.fromarray(img_array.astype(np.uint8))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)
    
if __name__ == "__main__":
    image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/render_512_Valour/color/cam-0007.png"
    image = Image.open(image_path)
    for i in range(10):
        img = to_rgb_image(image, bg_color=255, edge_aug_threshold=1.0)
        img.save(f"{str(i)}.png")
