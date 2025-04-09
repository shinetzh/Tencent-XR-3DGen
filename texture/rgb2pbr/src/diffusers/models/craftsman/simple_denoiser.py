from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.embeddings import Timesteps
import math
import os
import copy
from easydict import EasyDict as edict

from .utils_transformers.attention import ResidualAttentionBlock
from .utils_transformers.utils import init_linear, MLP


class UNetDiffusionTransformer(nn.Module):
    def __init__(
            self,
            *,
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
            init_scale: float = 0.25,
            qkv_bias: bool = False,
            skip_ln: bool = False,
            use_checkpoint: bool = False
    ):
        super().__init__()

        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        self.encoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_checkpoint=use_checkpoint
            )
            self.encoder.append(resblock)

        self.middle_block = ResidualAttentionBlock(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_checkpoint=use_checkpoint
        )

        self.decoder = nn.ModuleList()
        for _ in range(layers):
            resblock = ResidualAttentionBlock(
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                use_checkpoint=use_checkpoint
            )
            linear = nn.Linear(width * 2, width)
            init_linear(linear, init_scale)

            layer_norm = nn.LayerNorm(width) if skip_ln else None

            self.decoder.append(nn.ModuleList([resblock, linear, layer_norm]))

    def forward(self, x: torch.Tensor):

        enc_outputs = []
        for block in self.encoder:
            x = block(x)
            enc_outputs.append(x)

        x = self.middle_block(x)

        for i, (resblock, linear, layer_norm) in enumerate(self.decoder):
            x = torch.cat([enc_outputs.pop(), x], dim=-1)
            x = linear(x)

            if layer_norm is not None:
                x = layer_norm(x)

            x = resblock(x)

        return x



class SimpleDenoiser(nn.Module):
    def __init__(self,
        pretrained_model_name_or_path: Optional[str] = None,
        input_channels: int = 8,
        output_channels: int = 8,
        n_ctx: int = 197,
        width: int = 1024,
        layers: int = 12,
        heads: int = 16,
        context_dim: int = 768,
        context_ln: bool = True,
        skip_ln: bool = False,
        init_scale: float = 0.25,
        flip_sin_to_cos: bool = False,
        use_checkpoint: bool = False):
        super().__init__()
    
        init_scale = init_scale * math.sqrt(1.0 / width)

        self.backbone = UNetDiffusionTransformer(
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            skip_ln=skip_ln,
            init_scale=init_scale,
            use_checkpoint=use_checkpoint
        )
        self.ln_post = nn.LayerNorm(width)
        self.input_proj = nn.Linear(input_channels, width)
        self.output_proj = nn.Linear(width, output_channels)

        # timestep embedding
        self.time_embed = Timesteps(width, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=0)
        self.time_proj = MLP(width=width, init_scale=init_scale)

        if context_ln:
            self.context_embed = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, width),
            )
        else:
            self.context_embed = nn.Linear(context_dim, width)
        
        if pretrained_model_name_or_path:
            pretrained_ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
            _pretrained_ckpt = {}
            for k, v in pretrained_ckpt.items():
                if k.startswith('denoiser_model.'):
                    _pretrained_ckpt[k.replace('denoiser_model.', '')] = v
            pretrained_ckpt = _pretrained_ckpt
            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('denoiser_model.'):
                        _pretrained_ckpt[k.replace('denoiser_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
            self.load_state_dict(pretrained_ckpt, strict=True)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                encoder_hidden_states: torch.FloatTensor,
                cross_attention_kwargs=None,
                return_dict = True):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """
        context = encoder_hidden_states
        _, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_proj(self.time_embed(timestep)).unsqueeze(dim=1)

        # 2. conditions projector
        context = self.context_embed(context)

        # 3. denoiser
        x = self.input_proj(model_input)
        x = torch.cat([t_emb, context, x], dim=1)
        x = self.backbone(x)
        x = self.ln_post(x)
        x = x[:, -n_data:] # B, n_data, width
        sample = self.output_proj(x) # B, n_data, embed_dim


        if not return_dict:
            return (sample,)

        return edict({"sample": sample})

    def save_checkpoint(self, save_dir, weights_name="crm.pt"):
        # Save the model
        os.makedirs(save_dir, exist_ok=True)
        shadow_model = copy.deepcopy(self)
        shadow_model.to(torch.device("cpu"))
        torch.save(shadow_model.state_dict(), os.path.join(save_dir, weights_name))
        del shadow_model


    def load_checkpoint(self, save_dir, weights_name="crm.pt"):
        # Load the model
        ckpt_path = os.path.join(save_dir, weights_name)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        print(f"dit model loaded ckpt from {ckpt_path}")


if __name__ == "__main__":
    from thop import profile

    device = torch.device("cuda:0")
    """
    test cross attention
    """
    # x = torch.randn([4, 512, 768])
    # y = torch.randn([4, 197, 768])
    # attention = Attention(dim=768, num_heads=8, qkv_bias=True)
    # cross_attention = Attention(dim=768, num_heads=8, qkv_bias=True)
    # # result = attention(x)
    # result = cross_attention(x, y)
    # print(result.shape)


    """
    test position embedding
    """
    # pos_embedding = get_2d_sincos_pos_embed(512, 16)
    # pos_embed = get_1d_sincos_pos_embed_from_grid(768, np.array(range(512)))
    # print(pos_embed.shape)

    """
    test dit
    """
    crm = SimpleDenoiser().to(device)
    x = torch.randn([5, 512, 8]).to(device)
    condition = torch.randn([5, 197, 768]).to(device)
    t = torch.randint(0, 1000, (5,), dtype=torch.long).to(device)
    Flops, params = profile(crm, inputs=(x, t, condition)) # macs
    print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
    print('params: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值

    output = crm(x, t, condition).sample
    print(x.shape)
    print(condition.shape)
    print(t.shape)
    print(output.shape)