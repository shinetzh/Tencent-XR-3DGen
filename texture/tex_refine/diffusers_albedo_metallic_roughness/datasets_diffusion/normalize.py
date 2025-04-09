import torch
import os
import numpy as np

def latent_norm(latent_image, p=2, need_norm=False):
    if need_norm:
        latent_norm = torch.norm(latent_image, p=p, dim=-1, keepdim=True)
        latent_image = latent_image / latent_norm
    return latent_image


def get_scale_params(train_dir, configs):
    
    std_reciprocal = configs["data_config"].get("std_reciprocal", None)
    if std_reciprocal:
        print(f"get std_reciprocal: {std_reciprocal} ------")


    if os.path.exists(os.path.join(train_dir, "stats")):
        print("train: get stats---")
        stats_dir = os.path.join(train_dir, "stats")
        min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
        max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
        _range = max_values - min_values
        middle = (min_values + max_values) / 2
        print(f"self._range: {_range}, self.middle: {middle}")
    else:
        _range = None
        middle = None
    middle_range = [middle, _range]

    ## check mean, range
    mean_path = os.path.join(train_dir, "mean.npy")
    range_path = os.path.join(train_dir, "range.npy")
    if os.path.exists(mean_path):
        print("train: get mean range---")
        mean = np.load(mean_path).astype(np.float32)
        range = np.load(range_path).astype(np.float32)
        print(f"self.mean: {mean}, self.range: {range}")
    else:
        mean = None
        range = None
    mean_range = [mean, range]
    scale_type = configs["data_config"].get("scale_type", None)

    return [scale_type, std_reciprocal, mean_range, middle_range]


def unnormalize(sample, stats_dir=None, middle=None, _range=None):
    # raise Exception('This version of unnormalize is deprecated.')
    # RESCALE IMAGE -- Needs to be aligned with input normalization!
    need_reshape = False
    if len(sample.shape) == 5:
        need_reshape = True
        shape_origin = sample.shape
        plane_list = []
        for i in range(sample.shape[1]):
            plane_list.append(sample[:, i, :, :, :])
        sample = torch.concat(plane_list, dim=1)
    if middle is None or _range is None:
        min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
        max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
        _range = max_values - min_values
        middle = (min_values + max_values) / 2
        # print(sample.shape)  # ex: [4, 96, 128, 128]
    middle_tensor = torch.tensor(middle, dtype=torch.float32, device=sample.device)
    _range_tensor = torch.tensor(_range, dtype=torch.float32, device=sample.device)

    sample = sample * (_range_tensor / 2) + middle_tensor

    if need_reshape:
        sample = sample.view(*shape_origin)

    return sample

def normalize(sample, stats_dir=None, middle=None, _range=None):
    # raise Exception('This version of unnormalize is deprecated.')
    # RESCALE IMAGE -- Needs to be aligned with input normalization!
    need_reshape = False
    if len(sample.shape) == 5:
        need_reshape = True
        shape_origin = sample.shape
        plane_list = []

        for i in range(sample.shape[1]):
            plane_list.append(sample[:, i, :, :, :])
        sample = torch.concat(plane_list, dim=1)
    if middle is None or _range is None:
        min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
        max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
        _range = max_values - min_values
        middle = (min_values + max_values) / 2

    # print(sample.shape)  # ex: [4, 96, 128, 128]
    middle_tensor = torch.tensor(middle, dtype=torch.float32, device=sample.device)
    _range_tensor = torch.tensor(_range, dtype=torch.float32, device=sample.device)

    sample = (sample - middle_tensor) / (_range_tensor / 2)

    if need_reshape:
        sample = sample.view(*shape_origin)

    return sample

def normalize_latent(latent, scale_type, std_reciprocal, mean_range, middle_range):
    if scale_type == "std_scale":
        return latent * std_reciprocal
    elif scale_type == "mean_range":
        mean, range = mean_range
        return normalize(latent, middle=mean, _range=range)
    elif scale_type == "stats":
        middle, _range = middle_range
        return normalize(latent, middle=middle, _range=_range)
    else:
        return latent


def unnormalize_latent(latent, scale_type, std_reciprocal, mean_range, middle_range):
    if scale_type == "std_scale":
        return latent / std_reciprocal
    elif scale_type == "mean_range":
        mean, range = mean_range
        return unnormalize(latent, middle=mean, _range=range)
    elif scale_type == "stats":
        middle, _range = middle_range
        return unnormalize(latent, middle=middle, _range=_range)
    else:
        return latent

if __name__ == "__main__":
    atensor = torch.randn(4, 16, 48).cuda()
    norm_atensor = latent_norm(atensor, need_norm=True)
    breakpoint()