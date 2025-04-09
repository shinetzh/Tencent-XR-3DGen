import albumentations as A
import cv2
import os
from tqdm import tqdm

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.ImageCompression(10, 50, p=0.5), 
    # A.GlassBlur(sigma=0.3, max_delta=1, iterations=1, always_apply=False, mode='fast', p=0.5),
    A.Downscale(scale_min=0.35, scale_max=0.5, interpolation=None, always_apply=False, p=0.5),
])

transform1 = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
])

transform2 = A.Compose([
    A.ImageCompression(10, 100, p=0.5),  ## just test, when training need change
])

transform3 = A.Compose([
    A.GlassBlur(sigma=0.3, max_delta=1, iterations=1, always_apply=False, mode='fast', p=0.5),
])

transform4 = A.Compose([
    A.Downscale(scale_min=0.25, scale_max=0.8, interpolation=None, always_apply=False, p=0.5),
])

if __name__ == "__main__":
    image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_muchengwuyu_images/Designcenter_1_0a1a2faf89cec1db190d05a6e5537a067ff8d81c_manifold_full_output_512_MightyWSBcam-0100.png"
    # Read an image with OpenCV and convert it to the RGB colorspace
    save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_muchengwuyu_images_aug/compose"
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in tqdm(range(50)):
        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        save_path = os.path.join(save_dir, "test_"+str(i).zfill(3)+".jpg")
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, transformed_image)