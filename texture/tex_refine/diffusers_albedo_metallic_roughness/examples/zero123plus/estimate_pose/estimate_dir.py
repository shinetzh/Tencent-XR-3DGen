import os
from utils_estimate_azimuth import get_image_elevation

obj_dir = "/aigc_cfs_gdp/sz/result/general_generate_z123_v21_step37500_v2"
for filename in os.listdir(obj_dir):
    sub_dir = os.path.join(obj_dir, filename)
    print(sub_dir)
    try:
        get_image_elevation(sub_dir)
    except KeyboardInterrupt:
        break
    except:
        continue