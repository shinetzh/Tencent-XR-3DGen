import os
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict

class TriplaneDataLoader(Dataset):

    def __init__(self, config, return_filename=False, sort=False):
        super(TriplaneDataLoader, self).__init__()
        
        self.config = config = edict(config)
        self.return_filename = return_filename
        self.vroid_output_dir = self.config.vroid_output_dir
        triplane_dir_list = config.objects
        # load all triplanes

        self.triplane_path_list = []
        for triplane_dir in triplane_dir_list:
          for triplane_name in os.listdir(triplane_dir):
            if not (triplane_name.endswith(".pt") or triplane_name.endswith(".tar")) or "mlp" in triplane_name:
                continue
            triplane_path = os.path.join(triplane_dir, triplane_name)
            self.triplane_path_list.append(triplane_path)
        # self.minnum_save = 10000
        # self.maxnum_save = -10000

        self.minnum = -0.5
        self.maxnum = 0.5
        self.base = self.maxnum - self.minnum

        if sort:
            self.triplane_path_list.sort()


        print(f"{self.__len__()} objs in total")


    def get_one_sample(self, idx):
        # load objects
        triplane = torch.load(self.triplane_path_list[idx]).squeeze()
        # if torch.min(triplane) < self.minnum_save:
        #     self.minnum_save = torch.min(triplane)
        #     with open("min_max.txt", 'w') as fw:
        #         fw.write("minnum: {}\nmaxnum: {}".format(self.minnum_save, self.maxnum_save))
        # if torch.max(triplane) > self.maxnum_save:
        #     self.maxnum_save = torch.max(triplane)
        #     with open("min_max.txt", 'w') as fw:
        #         fw.write("minnum: {}\nmaxnum: {}".format(self.minnum_save, self.maxnum_save))
        if len(triplane.shape) > 3:
            triplane = torch.concat([triplane[0, :, :, :],
                              triplane[1, :, :, :],
                              triplane[2, :, :, :]], dim=0)

        triplane = ((triplane - self.minnum) / self.base - 0.5) * 2
        if self.return_filename:
            triplane_path = self.triplane_path_list[idx]
            vroid_class = triplane_path.split("/")[-3].split("vroid")[-1]
            vroid_objname = triplane_path.split("/")[-1].split('.')[0]
            image_path = os.path.join(self.vroid_output_dir, vroid_class, vroid_objname, vroid_objname+"_output", "color/cam-0100.jpg")

            return triplane, image_path
        else:
            return triplane

    def __len__(self):
        return len(self.triplane_path_list)

    def __getitem__(self, objid):
        return self.get_one_sample(objid)