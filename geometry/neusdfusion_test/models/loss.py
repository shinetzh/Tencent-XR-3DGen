import torch.nn as nn
import torch

def tvloss(x):
    b, n, c, h_x, w_x = x.shape
    count_h = n * c * (h_x - 1) * w_x
    count_w = n * c * h_x * (w_x - 1)
    h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x-1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x-1]), 2).sum()
    return (h_tv / count_h + w_tv / count_w) / b

L2Loss = torch.nn.MSELoss(reduction="mean")