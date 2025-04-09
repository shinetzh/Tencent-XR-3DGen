import cv2
import numpy as np


image1 = cv2.imread('/aigc_cfs_2/neoshang/code/diffusers_triplane/data/render_512_Valour/color/cam-0032.png', flags=cv2.IMREAD_UNCHANGED)
image2 = cv2.imread('/aigc_cfs_2/neoshang/code/diffusers_triplane/data/render_512_Valour/color/cam-0017.png', flags=cv2.IMREAD_UNCHANGED)

cv2.imwrite("image1.png", image1)
cv2.imwrite("image2.png", image2)

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_image1.png", gray_image1)
cv2.imwrite("gray_image2.png", gray_image2)

# hist_image1 = cv2.calcHist([gray_image1], [0], mask=image1[..., -1], histSize=[256], ranges=[0, 256])
# hist_image2 = cv2.calcHist([gray_image2], [0], mask=image2[..., -1], histSize=[256], ranges=[0, 256])

# hist_image1 = cv2.normalize(hist_image1, hist_image1).flatten()
# hist_image2 = cv2.normalize(hist_image2, hist_image2).flatten()

avg_brightness1 = np.mean(gray_image1, where=image1[..., -1]>0)
avg_brightness2 = np.mean(gray_image2, where=image2[..., -1]>0)
print(f"avg_brightness1:{avg_brightness1}, avg_brightness2:{avg_brightness2}")

brightness_scale = avg_brightness1 / avg_brightness2

adjusted_image2 = cv2.convertScaleAbs(image2, alpha=brightness_scale, beta=0)
mask = image2[..., -1] / 255.0
mask_rgb = np.concatenate([mask[..., None]]*3, -1)
adjusted_image2_rgb = adjusted_image2[..., :3] * mask_rgb + 255 * (1 - mask_rgb)
cv2.imwrite("adjusted_image2.png", adjusted_image2_rgb)