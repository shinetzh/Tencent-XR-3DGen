""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DAware_dmtet(nn.Module):
    """
    adapted for 3D aware convolution for triplane in Rodin: https://arxiv.org/pdf/2212.06135.pdf
    xy, yz, xz - hw
    implement two versions, group or rollout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, rolloutORgroup="rollout"):
        super().__init__()
        self.rolloutORgroup = rolloutORgroup
        if rolloutORgroup == "group":
            self.conv = nn.Conv2d(in_channels * 6, out_channels * 3, kernel_size, stride, padding, bias=bias, groups=3)
        elif rolloutORgroup == "rollout":
            self.conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size, stride, padding, bias=bias)
    
    def perception_3d_rollout(self, x):
        _, _, h, w = x.shape
        fea_xy, fea_yz, fea_xz = x[..., 0:w//3], x[..., w//3:(w//3) * 2], x[..., (w//3) * 2:]
        fea_xy_mean_x = torch.mean(fea_xy, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_xy_mean_y = torch.mean(fea_xy, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yz_mean_y = torch.mean(fea_yz, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_yz_mean_z = torch.mean(fea_yz, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_xz_mean_x = torch.mean(fea_xz, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_xz_mean_z = torch.mean(fea_xz, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_xy_3d_aware = torch.cat((fea_xy, fea_yz_mean_y, fea_xz_mean_x), dim=1)
        fea_yz_3d_aware = torch.cat((fea_yz, fea_xy_mean_y, fea_xz_mean_z), dim=1)
        fea_xz_3d_aware = torch.cat((fea_xz, fea_xy_mean_x, fea_yz_mean_z), dim=1)
        fea = torch.cat([fea_xy_3d_aware, fea_yz_3d_aware, fea_xz_3d_aware], dim=3)

        return fea
    
    def perception_3d_group(self, x):
        _, c, h, w = x.shape
        fea_xy, fea_yz, fea_xz = x[:, 0:c//3, ...], x[:, c//3:(c//3) * 2, ...], x[:, (c//3) * 2:, ...]
        fea_xy_mean_x = torch.mean(fea_xy, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_xy_mean_y = torch.mean(fea_xy, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yz_mean_y = torch.mean(fea_yz, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_yz_mean_z = torch.mean(fea_yz, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_xz_mean_x = torch.mean(fea_xz, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_xz_mean_z = torch.mean(fea_xz, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_xy_3d_aware = torch.cat((fea_xy, fea_yz_mean_y, fea_xz_mean_x), dim=1)
        fea_yz_3d_aware = torch.cat((fea_yz, fea_xy_mean_y, fea_xz_mean_z), dim=1)
        fea_xz_3d_aware = torch.cat((fea_xz, fea_xy_mean_x, fea_yz_mean_z), dim=1)
        fea = torch.cat([fea_xy_3d_aware, fea_yz_3d_aware, fea_xz_3d_aware], dim=1)
        return fea

    def forward(self, x):
        if self.rolloutORgroup == "group":
            triplane = self.perception_3d_group(x)
        elif self.rolloutORgroup == "rollout":
            triplane = self.perception_3d_rollout(x)
        result = self.conv(triplane)
        return result


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, conv3d=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if conv3d:
            conv = Conv3DAware_dmtet
        else:
            conv = nn.Conv2d
        self.double_conv = nn.Sequential(
            conv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv3d=False, maxpool=False):
        super().__init__()
        if maxpool:
            down_block = nn.MaxPool2d(2)
        else:
            down_block = Conv3DAware_dmtet(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool_conv = nn.Sequential(
            down_block,
            DoubleConv(in_channels, out_channels, conv3d=conv3d)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, conv3d=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, conv3d=conv3d)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, conv3d=conv3d)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv3d=False):
        super(OutConv, self).__init__()
        if conv3d:
            self.conv = Conv3DAware_dmtet(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)