# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from grpc import xds_server_credentials
from sympy import xfield
import torch
import torch.nn as nn
import numpy as np
import math
import os
import copy
import sys
import json

from typing import Callable, Dict, List, Optional, Union
from easydict import EasyDict as edict
from torch.jit import Final
import torch.nn.functional as F
from timm.layers import use_fused_attn
from timm.models.vision_transformer import PatchEmbed, Mlp




class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k

            raise ValueError(
                f"There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}."
            )

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)




class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha = None,
        device = None,
        dtype = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.with_lora = False

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        if not self.with_lora:
            if y is None:
                y = x
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            _, yN, _ = y.shape
            kv = self.kv(y).reshape(B, yN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            if y is None:
                y = x
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q_lora = self.lora_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = q + q_lora
            _, yN, _ = y.shape

            kv = self.kv(y).reshape(B, yN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

            k_lora = self.lora_k(y).reshape(B, yN, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_lora = self.lora_k(y).reshape(B, yN, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k + k_lora
            v = v + v_lora
            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    def set_lora(self, rank=16):
        self.with_lora = True
        self.device = next(self.parameters()).device
        self.lora_q = LoRALinearLayer(in_features=self.dim, out_features=self.dim, rank=rank).to(self.device)
        self.lora_k = LoRALinearLayer(in_features=self.dim, out_features=self.dim, rank=rank).to(self.device)
        self.lora_v = LoRALinearLayer(in_features=self.dim, out_features=self.dim, rank=rank).to(self.device)
        
        return attention_lora_proxy(self.lora_q, self.lora_k, self.lora_v)

class attention_lora_proxy(nn.Module):
    def __init__(self, lora_q, lora_k, lora_v):
        super().__init__()
        self.lora_q = lora_q
        self.lora_k = lora_k
        self.lora_v = lora_v


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn_cross = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, time_embedding, condition_embedding):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_embedding).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        ## add by neoshang
        x = x + self.attn_cross(self.norm3(x), condition_embedding)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        condition_dim=256,
        learn_sigma=True,
        config_path=None
    ):
        super().__init__()
        if config_path:
            with open(config_path, "r") as fr:
                config = json.load(fr)
            self.config = config
            input_size = config["input_size"]
            in_channels = config["in_channels"]
            condition_dim = config["condition_dim"]
            learn_sigma = config["learn_sigma"]
            hidden_size = config.get("hidden_size", hidden_size)
            num_heads = config.get("num_heads", num_heads)
            self.zero_input = config.get("zero_input", False)
            print(f"dit.zero_input: {self.zero_input}")
        
        self.input_size = input_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        if not patch_size == 1:
            ## 'yx', 'zx', 'yz'
            self.yx_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            self.zx_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            self.yz_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            self.x_embedder = self.triplane_embedder
        else:
            self.x_embedder = nn.Sequential(
                nn.Linear(in_channels, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.condition_embedder = nn.Sequential(
            nn.Linear(condition_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        if not patch_size == 1:
            num_patches = self.yx_embedder.num_patches
        else:
            num_patches = input_size * input_size
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * 3, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_size, self.out_channels, bias=True)
        )
        if self.zero_input:
            self.zero_input_holder = torch.nn.Parameter(torch.randn(1, 4, 16, 48), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if not self.patch_size == 1:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            grid_size = int(self.yx_embedder.num_patches ** 0.5)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [grid_size, grid_size * 3])
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
            w_yx = self.yx_embedder.proj.weight.data
            nn.init.xavier_uniform_(w_yx.view([w_yx.shape[0], -1]))
            nn.init.constant_(self.yx_embedder.proj.bias, 0)

            # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
            w_zx = self.zx_embedder.proj.weight.data
            nn.init.xavier_uniform_(w_zx.view([w_zx.shape[0], -1]))
            nn.init.constant_(self.zx_embedder.proj.bias, 0)

            # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
            w_yz = self.yz_embedder.proj.weight.data
            nn.init.xavier_uniform_(w_yz.view([w_yz.shape[0], -1]))
            nn.init.constant_(self.yz_embedder.proj.bias, 0)
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [self.input_size, self.input_size * 3])
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

        ### zero-out output layers
        nn.init.constant_(self.final_layer[0].weight, 0)
        nn.init.constant_(self.final_layer[0].bias, 0)


    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def unpatchify_plane(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.yx_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def unpatchify(self, x):
        """
        x: (N, T*3, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        b, t, _ = x.shape
        t_plane = t // 3
        yx, zx, yz = x[:, :t_plane, ...], x[:, t_plane:2*t_plane, ...], x[:, 2*t_plane:3*t_plane, ...]
        yx = self.unpatchify_plane(yx)
        zx = self.unpatchify_plane(zx)
        yz = self.unpatchify_plane(yz)
        return torch.cat([yx, zx, yz], dim=-1)


    def triplane_embedder(self, x):
        b, c, h, w = x.shape
        yx, zx, yz = x[:, :, :, :h], x[:, :, :, h:2*h], x[:, :, :, 2*h:3*h]
        yx = self.yx_embedder(yx)
        zx = self.zx_embedder(zx)
        yz = self.yz_embedder(yz)
        return torch.cat([yx, zx, yz], dim=1)

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict = True, training=True):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        c: (N, squeence_length, dim) tensor of class labels
        """
        if self.zero_input:
            if training:
                zero_idx = torch.where(timestep > (1000 - 100))
                sample[zero_idx] = self.zero_input_holder
            else:
                zero_idx = torch.where(timestep == 0)
                sample[zero_idx] = self.zero_input_holder
        x = sample
        t = timestep
        if len(t.shape) < 1:
            t = t.expand(x.shape[0])
        
        b, c, h, w = sample.shape
        if not self.patch_size == 1:
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        else:
            x = x.reshape(b, c, h*w).permute(0, 2, 1)
            x = self.x_embedder(x) + self.pos_embed

        t = self.t_embedder(t)                   # (N, D)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)    # (N, squeece_length, D)

        """
        需要在dit block中加入 condition 和 x的cross attention, 加在feedforward 和 self attention 之间
        """
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, encoder_hidden_states)       # (N, T, D)

        x = self.final_layer(x)                # (N, T, patch_size ** 2 * out_channels)
        if not self.patch_size == 1:
            x = self.unpatchify(x)                   # (N, out_channels, H, W)
        else:
            x = x.permute(0, 2, 1).reshape(b, c, h, w)

        if not return_dict:
            return (x,)

        return edict({"sample": x})

    def forward_with_cfg(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None, return_dict = True):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        if not torch.is_tensor(timestep):
            dtype = torch.float32
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        model_out = self.forward(sample, timestep, encoder_hidden_states)
        return model_out

    def save_checkpoint(self, save_dir, weights_name="dit.pt"):
        # Save the model
        os.makedirs(save_dir, exist_ok=True)
        shadow_model = copy.deepcopy(self)
        shadow_model.to(torch.device("cpu"))
        torch.save(shadow_model.state_dict(), os.path.join(save_dir, weights_name))
        del shadow_model


    # def save_checkpoint(self, save_dir, weights_name="dit.pt"):
    #     # Save the model
    #     os.makedirs(save_dir, exist_ok=True)
    #     device = next(self.parameters()).device
    #     torch.save(self.to(torch.device("cpu")).state_dict(), os.path.join(save_dir, weights_name))
    #     self.to(device)

    def load_checkpoint(self, save_dir, weights_name="dit.pt"):
        # Load the model
        ckpt_path = os.path.join(save_dir, weights_name)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        print(f"dit model loaded ckpt from {ckpt_path}")

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    h, w = grid_size[0], grid_size[1]
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def DiT_XL_1(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_1(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_1(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_1(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/1': DiT_XL_1,  'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/1':  DiT_L_1,   'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/1':  DiT_B_1,   'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/1':  DiT_S_1,   'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


class lora_proxy(nn.Module):
    def __init__(self, lora_dict):
        super().__init__()
        self.layers = torch.nn.ModuleList(lora_dict.values())


def add_lora_dit(model):
    lora_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            attention_lora_proxy = module.set_lora(rank=16)
            if name == "blocks.0.attn":
                print(id(module.lora_q))
                print(id(attention_lora_proxy.lora_q))
            lora_dict[name] = attention_lora_proxy
    
    return lora_proxy(lora_dict)


if __name__ == "__main__":
    """
    test cross attention
    """
    
    # x = torch.randn([4, 1024, 512])
    # y = torch.randn([4, 197, 512])
    # attention = Attention(dim=512, num_heads=8, qkv_bias=True)
    # cross_attention = Attention(dim=512, num_heads=8, qkv_bias=True)
    # # result = attention(x)
    # result = cross_attention(x, y)

    """
    test position embedding
    """
    # pos_embedding = get_2d_sincos_pos_embed(512, 16)
    

    """
    test dit
    """
    exp_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/text_to_3d/910b_objaverse_v0.0.0"

    with open(os.path.join(exp_dir, "train_configs.json"), "r") as fr:
        json_config = json.load(fr)

    dit = DiT_models[json_config["diffusion_model"]["model_name"]](config_path=os.path.join(exp_dir, "unet/config.json"))
    dit.load_checkpoint("/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/text_to_3d/910b_objaverse_v0.0.0/checkpoint-94000/unet")
    dit = dit.cuda()
    lora_proxy = add_lora_dit(dit)
    for name, module in lora_proxy.named_modules():
        print(name)
        print(module)
        breakpoint()

    # print(dit)
    x = torch.randn([5, 4, 16, 48]).cuda()
    y = torch.randn([5, 197, 1024]).cuda()
    t = torch.randint(0, 1000, (5,), dtype=torch.long).cuda()
    output = dit(x, t, y).sample
    print(x.shape)
    print(y.shape)
    print(t.shape)
    print(output.shape)
    breakpoint()