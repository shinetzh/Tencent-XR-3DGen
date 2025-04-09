
'''
Normalize the triplane dataset, scaling each channel individually to [-1, 1]

Different channels have vastly different ranges of values, unlike RGB images.
'''

import argparse
import glob
import json
from pkgutil import get_data
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
# import blobfile as bf
import torch

import sys
sys.path.append("/aigc_cfs_2/neoshang/code/diffusers_triplane")
from datasets_diffusion import get_dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        normalize=False,  # Whether to rescale individual channels to [-1, 1] based on their respective ranges
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        stats_dir=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.normalize = normalize
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.stats_dir = stats_dir

        if self.normalize:
            print('Will normalize triplanes in the training loop.')
            if self.stats_dir is None:
                raise Exception('Need to provide a directory of stats to use for normalization.')
            # Load in min and max numpy arrays (shape==[96,] - one value per channel) for normalization
            # self.min_values = np.load('util/min_values.npy').astype(np.float32).reshape(-1, 1, 1)  # should be (96, 1, 1)
            # self.max_values = np.load('util/max_values.npy').astype(np.float32).reshape(-1, 1, 1)
            self.min_values = np.load(f'{self.stats_dir}/lower_bound.npy').astype(np.float32).reshape(-1, 1, 1)
            self.max_values = np.load(f'{self.stats_dir}/upper_bound.npy').astype(np.float32).reshape(-1, 1, 1)
            self.range = self.max_values - self.min_values
            self.middle = (self.min_values + self.max_values) / 2
        else:
            print('Not using normalization in ds.')

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        # Load np array
        # arr = np.load(path)
        # arr = torch.load(path).cpu().numpy()
        arr = torch.load(path)['triplane'][0].cpu().numpy()
        # Get rid of these extra operations I guess? (already inheriting variance from triplane generator)

        # if self.random_crop:
        #     arr = random_crop_arr(pil_image, self.resolution)
        # else:
        #     arr = center_crop_arr(pil_image, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     arr = arr[:, ::-1]

        # Normalize individual channels
        arr = arr.astype(np.float32)  # / 127.5 - 1  <-- need to normalize the triplanes in their own way.
        arr = arr.reshape([-1, arr.shape[-2], arr.shape[-1]])
        if self.normalize:
            arr = (arr - self.middle) / (self.range / 2)
        # np.save('random_normalized_triplane', arr)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # print(arr.shape)
        # return np.transpose(arr, [2, 0, 1]), out_dict
        return arr, out_dict

    def unnormalize(self, sample):
        sample = sample * (self.range / 2) + self.middle
        return sample


def unnormalize(sample, stats_dir):
    # raise Exception('This version of unnormalize is deprecated.')
    # RESCALE IMAGE -- Needs to be aligned with input normalization!
    min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
    max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
    _range = max_values - min_values
    middle = (min_values + max_values) / 2
    # print(sample.shape)  # ex: [4, 96, 128, 128]
    if torch.is_tensor(sample):
        middle = torch.tensor(middle, dtype=torch.float32, device=sample.device)
        _range = torch.tensor(_range, dtype=torch.float32, device=sample.device)

    sample = sample * (_range / 2) + middle
    return sample

def normalize(sample, stats_dir):
    # raise Exception('This version of unnormalize is deprecated.')
    # RESCALE IMAGE -- Needs to be aligned with input normalization!
    min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
    max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
    _range = max_values - min_values
    middle = (min_values + max_values) / 2

    # print(sample.shape)  # ex: [4, 96, 128, 128]
    if torch.is_tensor(sample):
        middle = torch.tensor(middle, dtype=torch.float32, device=sample.device)
        _range = torch.tensor(_range, dtype=torch.float32, device=sample.device)

    sample = (sample - middle) / (_range / 2)
    return sample

def main():
    parser = argparse.ArgumentParser(description='Normalize a dataset of triplanes')
    parser.add_argument('--config_dir', type=str, default='/apdcephfs_cq3/share_1615605/neoshang/code/diffusers_triplane/configs/triplane_occupancy_v1',
                    help='where to save stats.', required=False)
    parser.add_argument('--key', type=str, default='triplane', help='keyname', required=False)
    parser.add_argument('--save_name', type=str, default='stats', help='where to save stats.', required=False)
    
    args = parser.parse_args()

    stats_save_dir = os.path.join(args.config_dir, args.save_name)

    os.makedirs(stats_save_dir, exist_ok=True)
    
    config_path = os.path.join(args.config_dir, "train_configs.json")
    with open(config_path, 'r') as fr:
        configs = json.load(fr)
    ds = get_dataset(configs, data_type="train", resample=False)

    num_examples = len(ds)
    print(f'Finding statistics across {num_examples} examples.')

    min_values = None
    max_values = None
    mean_values = None
    std_values = None

    num_stds = 10
    print_shape = False

    for idx, batch in tqdm(enumerate(ds)):
        triplane = batch[args.key].squeeze()
        if len(triplane.shape) == 4:
            triplane = torch.cat([triplane[0, ...], triplane[1, ...], triplane[2, ...]], dim=0) # [96, 256, 256]
        triplane = triplane.cpu().numpy()
        if not print_shape:
            print(triplane.shape)
            print_shape = True
        if min_values is None:
            min_values = np.full(int(triplane.shape[0]), 1000)
            max_values = np.full(int(triplane.shape[0]), -1000)
            mean_values = np.full((len(ds), int(triplane.shape[0])), 0.0)
            std_values = np.full((len(ds), int(triplane.shape[0])), 0.0)
        
        _min = np.amin(triplane, axis=(1, 2))  # Collapse two dimensions
        _max = np.amax(triplane, axis=(1, 2))  # Collapse two dimensions
    
        # Update min and max
        min_values = np.where(_min < min_values, _min, min_values)
        max_values = np.where(_max > max_values, _max, max_values)

        # Update mean and SD
        mean_values[idx] = np.mean(triplane, axis=(1, 2))
        std_values[idx] = np.std(triplane, axis=(1, 2))
    
    means = np.mean(mean_values, axis=(0))
    stds = np.mean(std_values, axis=(0))

    lower_bound = means - (num_stds * stds)
    upper_bound = means + (num_stds * stds)

    np.save(f'{stats_save_dir}/lower_bound', lower_bound)
    np.save(f'{stats_save_dir}/upper_bound', upper_bound)
    np.save(f'{stats_save_dir}/means', means)

    # Save csv
    combined_arr = np.concatenate((min_values.reshape(-1, 1), max_values.reshape(-1, 1), 
                                lower_bound.reshape(-1, 1), upper_bound.reshape(-1, 1), means.reshape(-1, 1)), axis=1)
    np.savetxt(f'{stats_save_dir}/stats.csv', combined_arr, delimiter=',')

    for idx, line in enumerate(combined_arr):
        print(f'{idx}: min: {line[0]}; max: {line[1]}')

if __name__ == "__main__":
    main()