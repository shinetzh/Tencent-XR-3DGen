import torch
import numpy as np
from PIL import Image

def load_multiview_image(image_path, save_dir=None):
    image = Image.open(image_path)
    image = np.array(image)

    image_list = [image[:256, :256, :], image[:256, 256:512, :],
                  image[256:512, :256, :], image[256:512, 256:512, :],
                  image[512:768, :256, :], image[512:768, 256:512, :],
                  image[768:1024, :256, :], image[768:1024, 256:512, :]]
    image_concat1 = np.concatenate([image_list[0], image_list[4]], axis=1)
    image_concat2 = np.concatenate([image_list[1], image_list[5]], axis=1)
    image_concat3 = np.concatenate([image_list[2], image_list[6]], axis=1)
    image_concat4 = np.concatenate([image_list[3], image_list[7]], axis=1)
    Image.fromarray(np.concatenate([image_concat1, image_concat2], axis=0)).save("test1.jpg")
    Image.fromarray(np.concatenate([image_concat1, image_concat3], axis=0)).save("test2.jpg")
    Image.fromarray(np.concatenate([image_concat1, image_concat4], axis=0)).save("test3.jpg")

def load_multiview_image_t2m(image_path, save_dir=None):
    image = Image.open(image_path)
    image = np.array(image)

    # image_list = [image[:256, :256, :], image[:256, 256:512, :],
    #               image[:256, 512:768, :], image[:256, 768:1024, :],
    #               image[:256, 1024:1280, :], image[:256, 1280:1536, :],
    #               image[:256, 1536:1792, :], image[:256, 1792:2048, :],]
    
    image_list = [image[512:, :256, :], image[512:, 256:512, :],
                  image[512:, 512:768, :], image[512:, 768:1024, :],
                  image[512:, 1024:1280, :], image[512:, 1280:1536, :],
                  image[512:, 1536:1792, :], image[512:, 1792:2048, :],]

    image_concat1 = np.concatenate([image_list[0], image_list[2]], axis=1)
    image_concat2 = np.concatenate([image_list[1], image_list[3]], axis=1)
    image_concat3 = np.concatenate([image_list[4], image_list[5]], axis=1)
    image_concat4 = np.concatenate([image_list[6], image_list[7]], axis=1)
    Image.fromarray(np.concatenate([image_concat1, image_concat2], axis=0)).save("test_tiger1.jpg")
    Image.fromarray(np.concatenate([image_concat1, image_concat3], axis=0)).save("test_tiger2.jpg")
    Image.fromarray(np.concatenate([image_concat1, image_concat4], axis=0)).save("test_tiger3.jpg")


if __name__ == "__main__":
    # image_path = "/aigc_cfs_gdp/sz/result/pipe_test/12793ba6-328a-472f-a8ba-341e6cad56d5/zero123plus_result.jpg"
    # save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/tests/test_result"
    # load_multiview_image(image_path, save_dir)


    image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/weixuan_outputs/20240623_36w_epoch7_linear_cfg_20_1_step100_animal/A large, intimidating tiger with a striped coat..png"
    save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/tests/test_result"
    load_multiview_image_t2m(image_path, save_dir)