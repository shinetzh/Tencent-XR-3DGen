from PIL import Image
import cv2
from glob import glob
import numpy as np
from augmentation import transforms

colorjitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)

def aug_bg(image):
    back_num = np.random.randint(0, 256)
    background = Image.new("RGB", image.size, (back_num, back_num, back_num))
    background.paste(image, mask=image.split()[3])
    alpha = np.array(image.split()[3]).astype(float) / 255
    return background, alpha

def white_bg(image):
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    alpha = np.array(image.split()[3]).astype(float) / 255
    return background, alpha


def aug_noise(image):
    prob = np.random.rand()
    if prob < 0.25:
      noise = np.random.randn(*image.shape) * 255
      # noise = np.random.rand(*image.shape) * 255
      image = image.astype(float) + prob * noise
      image = np.clip(image, 0, 255)
      image = image.astype(np.uint8)
    return image

def aug_blur(image):
  prob = np.random.rand()
  if prob < 0.25:
    kernel_size = np.random.randint(1, 4) * 2 + 1
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
  return image


def mix_bg(image, bg_path_list, alpha, threshold = 0.7):
    prob = np.random.random()
    if prob < threshold:
      bg_num = len(bg_path_list)
      bg_idx = np.random.randint(0, bg_num)
      bg_path = bg_path_list[bg_idx]
      bg_img = Image.open(bg_path)
      bg_np = np.array(bg_img)
      bg_shape = bg_np.shape
      if not len(bg_shape) == 3:
          return mix_bg(image, bg_path_list, alpha, threshold = 0.7)
      h_bg, w_bg, c_bg = bg_np.shape
      h_img, w_img, c_img = image.shape
      if h_img < h_bg:
          h_start = np.random.randint(0, h_bg - h_img)
          h_end = h_start + h_img
      else:
          h_start = 0
          h_end = h_bg
      if w_img < w_bg:
          w_start = np.random.randint(0, w_bg - w_img)
          w_end = w_start + w_img
      else:
          w_start = 0
          w_end = w_bg
      alpha = np.concatenate([alpha[..., None]] * 3, axis=-1)
      bg_np_crop = cv2.resize(bg_np[h_start:h_end, w_start:w_end, :], [w_img, h_img], interpolation=cv2.INTER_LINEAR)
      bg_np_crop = bg_np_crop.astype(float) * prob * (1 - alpha) + image.astype(float) * alpha
      return bg_np_crop.astype(np.uint8)
    return image


def augment_color(image):
    image = colorjitter(image)
    return image

def image_augmentation(image_pil, bg_path_list):
    """
    usage:
    image_pil = Image.open(image_path)
    image_pil = augmentation(image_pil)
    """
    # image_pil, alpha = aug_bg(image_pil)
    image_pil, alpha = white_bg(image_pil)
    image_np_aug = np.array(image_pil)
    # image_np_aug = mix_bg(image_np_aug, bg_path_list, alpha, threshold = 0.2)
    # image_np_aug = augment_color(image_np_aug)
    # image_np_aug = aug_blur(image_np_aug)
    # image_np_aug = aug_noise(image_np_aug)
    return Image.fromarray(image_np_aug)
