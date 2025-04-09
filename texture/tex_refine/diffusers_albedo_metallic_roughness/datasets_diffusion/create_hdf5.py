import torch
import json
import h5py
import os
import random
import uuid
from queue import Queue
import numpy as np
from tqdm import tqdm
import threading
from threading import Thread
from datasets_triplane_diffusion_v3 import DatasetVAEDiffusionV3


ok_num = 0
all_num = 0
save_dir_idx = 0

class Worker(threading.Thread):
    def __init__(self, queue_params, lock):
        super().__init__()
        self.queue_params = queue_params
        self.lock = lock
    def run(self):
        global all_num
        global ok_num

        while True:
            param_list = self.queue_params.get()
            if param_list is None:
                self.queue_params.task_done()
                break
            chunk_item_list = param_list
            make_h5(chunk_item_list, self.lock)

            with self.lock:
                ok_num += chunk_num
                print("\rcomplete percentage: %.2f %%;" % (ok_num*100/all_num), 'complete sample amount: {}/{}'.format(ok_num, all_num), end="", flush=True)

            self.queue_params.task_done()

def save_h5(chunk_mean, chunk_logvar, chunk_image_latent, save_dir_idx):
    try:
        h5_uid = str(uuid.uuid4()).replace("-", "")
        chunk_h5_save_path = os.path.join(hdf5_save_dir_list[save_dir_idx], h5_uid+".h5")
        os.makedirs(hdf5_save_dir_list[save_dir_idx], exist_ok=True)
        with h5py.File(chunk_h5_save_path, "w") as h5file:
            h5file.create_dataset("latent_modulation_mean", data=chunk_mean, compression='gzip', compression_opts=5, chunks=chunk_mean.shape)
            h5file.create_dataset("latent_modulation_logvar", data=chunk_logvar, compression='gzip', compression_opts=5, chunks=chunk_logvar.shape)
            h5file.create_dataset("image_latent", data=chunk_image_latent, compression='gzip', compression_opts=5, chunks=chunk_image_latent.shape)
        return chunk_h5_save_path
    except:
        return None
    
def make_h5(chunk_item_list, lock):
    global save_dir_idx
    chunk_classname = []
    chunk_objname = []
    chunk_image_path_list = []
    chunk_mean_list = []
    chunk_logvar_list = []
    chunk_image_latent_list = []

    for item in chunk_item_list:
        chunk_classname.append(item["classname"])
        chunk_objname.append(item["objname"])
        chunk_image_path_list.append(item["image_path"])
        chunk_mean_list.append(item["latent_modulation_mean"].numpy())
        chunk_logvar_list.append(item["latent_modulation_logvar"].numpy())
        chunk_image_latent_list.append(item["image_latent"])

    chunk_mean = np.stack(chunk_mean_list, axis=0)
    chunk_logvar = np.stack(chunk_logvar_list, axis=0)
    chunk_image_latent = np.stack(chunk_image_latent_list, axis=0)

    for i in range(save_dir_idx, len(hdf5_save_dir_list)):
        h5_save_path = save_h5(chunk_mean, chunk_logvar, chunk_image_latent, save_dir_idx)
        if not h5_save_path is None:
            break
        else:
            print(f"{hdf5_save_dir_list[save_dir_idx]} disk is full")
            save_dir_idx += 1
    
    if h5_save_path is None:
        exit("all save disk are full, please add other disk for saving hdf5")

    with lock:
        data_list.append({"classname": chunk_classname,
                            "objname": chunk_objname,
                            "image_path": chunk_image_path_list,
                            "h5_path": h5_save_path})


if __name__ == "__main__":
    config = {
                "exp_save_dir": "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdf_character_kl_v0.0.0_test910b_float32_dit",
                "data_config" : {
                    "dataset_name" : "vae_diffusion_v3",
                    "dataset_json" : "/data1/Data/Debug/merge.json",
                    "train_class_exclude_list": ["DragonBall", "onepiece", "Objaverse_Avatar"],
                    "latent_from_vae": "kl",
                    "condition_num": 21,
                    "std_reciprocal": 2.853907392401399,
                    "scale_type": "std_scale", 
                    "load_from_cache_last": True,
                    "resample": False
                }
            }

    datasets = DatasetVAEDiffusionV3(config, resample=False, data_type="train", load_from_cache_last=False)
    print(datasets.__len__())
    indexes = list(range(0, datasets.__len__()))
    random.shuffle(indexes)
    print(indexes[0:100])


    json_save_path = "/aigc_cfs_2/neoshang/data/diffusion_datasets_v0/datasets.json"
    hdf5_save_dir_list = ["/data3/neoshang/data/diffusion_v0",
                        "/data4/neoshang/data/diffusion_v0",
                        "/data5/neoshang/data/diffusion_v0",
                        "/data6/neoshang/data/diffusion_v0",
                        "/data7/neoshang/data/diffusion_v0",
                        ]

    generate_num = min(1000000000, len(datasets))

    params_queue = Queue()
    worker_num = 128
    lock = threading.Lock()

    worker_list = []
    for _ in range(worker_num):
        worker = Worker(params_queue, lock)
        worker.start()
        worker_list.append(worker)


    data_list = []
    num = 0
    chunk_num = 64
    chunk_idx = 0
    chunk_item_list = []

    for idx in tqdm(indexes):
        item = datasets[idx]
        if chunk_idx < chunk_num:
            chunk_idx += 1
            chunk_item_list.append(item)
        else:
            params_queue.put(chunk_item_list.copy())
            all_num += chunk_num

            chunk_idx = 1
            chunk_item_list = []
            chunk_item_list.append(item)
        
        num += 1
        if num >= generate_num:
            break
        


    # 将 None 添加到队列中，通知线程退出
    for _ in range(worker_num):
        params_queue.put(None)
    params_queue.join()

    # 等待所有线程完成
    for worker in worker_list:
        worker.join()

    with open(json_save_path, 'w') as fw:
        json.dump(data_list, fw, indent=2)
    print("\n")

    ##### read h5 test
    h5_path = data_list[-1]["h5_path"]
    with h5py.File(h5_path, 'r') as h5file:
        print(h5file["latent_modulation_mean"].shape)
        print(h5file["latent_modulation_logvar"].shape)