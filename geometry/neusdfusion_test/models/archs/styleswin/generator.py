# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, trunc_normal_
from torch import nn

from models.archs.styleswin.basic_layers import (EqualLinear, PixelNorm,
                                 SinusoidalPositionalEmbedding, Upsample)
from models.autoencoder import ResBlock

class ToRGB(nn.Module):
    def __init__(self, in_channel, out_channel=96, upsample=True, resolution=None, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.is_upsample = upsample
        self.resolution = resolution
        ratio = 2

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = nn.Conv2d(in_channel, in_channel * ratio , kernel_size=1, groups=3)
        self.gn = nn.GroupNorm(num_groups=3, num_channels=in_channel * ratio)
        self.act = nn.SiLU()
        self.group_conv = nn.Conv2d(in_channel * ratio, out_channel, kernel_size=1, groups=3)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = self.gn(out)
        out = self.act(out)
        out = self.group_conv(out) + self.bias

        if skip is not None:
            if self.is_upsample:
                skip = self.upsample(skip)

            out = out + skip
        out = nn.functional.tanh(out)
        return out


class ToRGB2(nn.Module):
    def __init__(self, in_channel, out_channel=96, upsample=True, resolution=None, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.is_upsample = upsample
        self.resolution = resolution
        ratio = 2

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = nn.Conv2d(in_channel, in_channel * ratio , kernel_size=1, groups=3)
        self.gn = nn.GroupNorm(num_groups=3, num_channels=in_channel * ratio)
        self.act = nn.SiLU()
        self.group_conv = nn.Conv2d(in_channel * ratio, out_channel, kernel_size=1, groups=3)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = self.gn(out)
        out = self.act(out)
        out = self.group_conv(out) + self.bias

        if skip is not None:
            if self.is_upsample:
                skip = self.upsample(skip)

            out = out + skip
        out = nn.functional.tanh(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta
        return out


class StyleSwinTransformerBlock(nn.Module):
    r""" StyleSwin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        style_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, style_dim=512):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2
        self.style_dim = style_dim
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = AdaptiveInstanceNorm(dim, style_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])
        
        attn_mask1 = None
        attn_mask2 = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                            self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        
        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.norm2 = AdaptiveInstanceNorm(dim, style_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, style):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        # Double Attn
        shortcut = x
        x = self.norm1(x.transpose(-1, -2), style).transpose(-1, -2)
        
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)
        
        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, self.attn_mask2)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.transpose(-1, -2), style).transpose(-1, -2))

        return x
    
    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class StyleSwinTransformerBlockWithConv(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, style_dim=512):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2
        self.style_dim = style_dim
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = AdaptiveInstanceNorm(dim, style_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])
        
        attn_mask1 = None
        attn_mask2 = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                            self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        
        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.norm2 = AdaptiveInstanceNorm(dim, style_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)

    def forward(self, x, style):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        # Double Attn
        shortcut = x
        x = self.norm1(x.transpose(-1, -2), style).transpose(-1, -2)
        
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)
        
        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, self.attn_mask2)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        # conv
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.conv1(x1)
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        x2 = self.conv2(x2)
        x2 = x2.permute(0, 2, 3, 1).contiguous()

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.transpose(-1, -2), style).transpose(-1, -2))

        return x
    
    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows


class StyleBasicLayer(nn.Module):
    """ A basic StyleSwin layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        out_dim (int): Number of output channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        style_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, out_dim=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., upsample=None, 
                 use_checkpoint=False, style_dim=512, add_conv=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        if not add_conv:
            self.blocks = nn.ModuleList([
                StyleSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop, style_dim=style_dim)
                for _ in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                StyleSwinTransformerBlockWithConv(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop, style_dim=style_dim)
                for _ in range(depth)])

        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim)
        else:
            self.upsample = None

    def forward(self, x, latent1, latent2):
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.blocks[0], x, latent1)
            x = checkpoint.checkpoint(self.blocks[1], x, latent2)
        else:
            x = self.blocks[0](x, latent1)
            x = self.blocks[1](x, latent2)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class BilinearUpsample(nn.Module):
    """ BilinearUpsample Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, out_dim=None):
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim // 2, padding_idx=0, init_size=out_dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "wrong in PatchMerging"

        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L*4, C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # Add SPE    
        x = x.reshape(B, H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * 2 * W * 2, self.out_dim)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # LN
        flops = 4 * H * W * self.dim
        # proj
        flops += 4 * H * W * self.dim * (self.out_dim)
        # SPE
        flops += 4 * H * W * 2
        # bilinear
        flops += 4 * self.input_resolution[0] * self.input_resolution[1] * self.dim * 5
        return flops


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            768, 
            384, 
            192, 
            96 
        ] 
        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)]

        self.input = ConstantInput(in_channels[0])
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB(out_dim, plane_channel, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        # styles = self.style(noise)
        styles = noise
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2
        return skip



class Generator2(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            384, 
            384, 
            192, 
            96 
        ] 
        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)]

        self.input = ConstantInput(in_channels[0])
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB2(out_dim, plane_channel, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2
        return skip


class ToRGB3(nn.Module):
    def __init__(self, in_channel, out_channel=96, upsample=True, resolution=None, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        ratio = 2

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=3, num_channels=in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel, in_channel * ratio , kernel_size=1, groups=3)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=3, num_channels=in_channel * ratio),
            nn.SiLU(),
            nn.Conv2d(in_channel * ratio, out_channel, kernel_size=1, groups=3),
        )

        self.skip_connection = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=3)

    def forward(self, input):
        out = self.in_layers(input)
        out = self.out_layers(out)

        out = out + self.skip_connection(input)
        out = nn.functional.tanh(out)
        return out

        
class Generator3(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.plane_channel = plane_channel
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            384, 
            384, 
            192, 
            192
        ] 

        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)]

        self.input = ConstantInput(in_channels[0])
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0

        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** (i_layer),2 ** (i_layer)),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)
            num_layers += 2

        self.to_rgb = ToRGB3(in_channels[-1], plane_channel, upsample=False, resolution=size)

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        for layer in self.layers:
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            count = count + 2
        b, n, c = x.shape
        skip = self.to_rgb(x.transpose(-1, -2).reshape(b, c, self.size, self.size))
        
        return skip

        


class Generator4(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            384, 
            384, 
            192, 
            192
        ] 
        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)]

        self.input = ConstantInput(in_channels[0])
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB2(out_dim, plane_channel, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2
        return skip

        
class Generator6(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            384, 
            384, 
            192, 
            192, 
            192,
            192
        ] 
        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)] + [8, 8]

        self.input = ConstantInput(in_channels[0], size=4)
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB(out_dim, plane_channel, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        for i in range(2):
            in_channel = in_channels[i_layer + i - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer + i - start],
                               num_heads=num_heads[i_layer +i - start],
                               window_size=window_sizes[i_layer + i - start],
                               out_dim=in_channels[i_layer + i - start + 1],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer + i - start + 1]
            to_rgb = ToRGB(out_dim, plane_channel, upsample=False, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2
        return skip


class ToRGBWithConvTranspose(nn.Module):
    def __init__(self, in_channel, out_channel=96, upsample=True, resolution=None, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.is_upsample = upsample
        self.resolution = resolution
        ratio = 2

        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1, groups=3)
            # self.upsample = Upsample(blur_kernel)

        self.conv = nn.Conv2d(in_channel, in_channel * ratio , kernel_size=1, groups=3)
        self.gn = nn.GroupNorm(num_groups=3, num_channels=in_channel * ratio)
        self.act = nn.SiLU()
        self.group_conv = nn.Conv2d(in_channel * ratio, out_channel, kernel_size=1, groups=3)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = self.gn(out)
        out = self.act(out)
        out = self.group_conv(out) + self.bias

        if skip is not None:
            if self.is_upsample:
                skip = self.upsample(skip)
            out = out + skip
        out = nn.functional.tanh(out)
        return out

    
class Generator7(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            384, 
            384, 
            192, 
            96
        ] 
        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)] + [8, 8]

        self.input = ConstantInput(in_channels[0], size=4)
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGBWithConvTranspose(out_dim, plane_channel, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2
        return skip

        

class Generator8(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            384, 
            384, 
            192, 
            96
        ] 
        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)] + [8, 8]

        self.input = ConstantInput(in_channels[0], size=4)
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim, add_conv=True)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB(out_dim, plane_channel, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2
        return skip

        
class Generator9(nn.Module):
    def __init__(
        self,
        size,
        plane_channel,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        in_channels = [
            768, 
            768, 
            384, 
            384, 
            192, 
            96,
            96,
            96
        ] 
        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)] + [8, 8]

        self.input = ConstantInput(in_channels[0], size=4)
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim, add_conv=True)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB(out_dim, plane_channel, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        for i in range(2):
            in_channel = in_channels[i_layer + i - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer + i - start],
                               num_heads=num_heads[i_layer +i - start],
                               window_size=window_sizes[i_layer + i - start],
                               out_dim=in_channels[i_layer + i - start + 1],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim, add_conv=True)
            self.layers.append(layer)

            out_dim = in_channels[i_layer + i - start + 1]
            to_rgb = ToRGB(out_dim, plane_channel, upsample=False, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers

    def forward(
        self,
        noise,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2
        return skip