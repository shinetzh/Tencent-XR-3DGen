import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from einops import rearrange, reduce

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar("torch.tensor")

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# from models.archs.attention_utils.SelfAttention import ScaledDotProductAttention
# from models.archs.attention_utils.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
# from models.archs.attention_utils.CBAM import CBAMBlock
# from ldm.modules.diffusionmodules.util import (
#     conv_nd,
#     avg_pool_nd,
#     zero_module,
#     normalization,
# )

from models.archs.attention import *
from models.autoencoder import ResBlock, ResBlockGC

    

class StyleSwinVAE_base(nn.Module):
    def __init__(self) -> None:
        super(StyleSwinVAE_base, self).__init__()
        pass

    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        result = enc_input
        if self.plane_dim == 5:
            plane_list = []
            for i in range(self.plane_shape[0]):
                plane_list.append(result[:, i, :, :, :])
            result = torch.concat(plane_list, dim=-1)
        elif self.plane_dim == 4:
            plane_channel = result.shape[1] // 3
            result = torch.concat([result[:, 0:plane_channel ,...],
                                result[:, plane_channel:plane_channel*2 ,...],
                                result[:, plane_channel*2:plane_channel*3 ,...]], dim=-1)
            
        feature = self.in_layer(result)


        for i, module in enumerate(self.encoders_down):
            feature = module(feature)

        feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        feature = rearrange(feature, 'b c h p w -> b (p c) h w').contiguous()
        feature = self.encoder_out_layer(feature)
        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()


        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        # x = self.decoder(z)
        # x = rearrange(x, 'b (c d) h w -> b c d h w', c=3).contiguous()
        x = self.decoder(z).reshape(-1,  3, 32, 128, 128)

        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)

        result = self.decode(z)

        return  [result, data, mu, log_var, z]

    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]
      
        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, log_var) 
            #print("latent shape: ", latent.shape) # (B, dim)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]

        else:
            std = torch.exp(torch.clamp(0.5 * log_var, max=10)) + 1e-6
            gt_dist = torch.distributions.normal.Normal( torch.zeros_like(mu), torch.ones_like(std)*self.kl_std )
            sampled_dist = torch.distributions.normal.Normal( mu, std )
            #gt_dist = normal_dist.sample(log_var.shape)
            #print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist) # reversed KL
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()

        return self.kl_weight * kl_loss

    def sample(self,
               num_samples:int,
                **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z_rollout_shape = [self.z_shape[0]]
        gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, *(z_rollout_shape)), 
                                                    torch.ones(num_samples, *(z_rollout_shape)) * self.kl_std)

        z = gt_dist.sample().cuda()
        samples = self.decode(z)
        return samples, z

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_latent(self, x):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z


class StyleSwinVAE_v2_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v2_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [512])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator
        self.decoder = Generator(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=8, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        x = self.decoder(z).reshape(-1,  3, 32, 128, 128)

        return x


class StyleSwinVAE_v3_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v3_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [512])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator2
        self.decoder = Generator2(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=4, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        x = self.decoder(z).reshape(-1,  3, 32, 128, 128)

        return x



class StyleSwinVAE_v4_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v4_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [512])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator3
        self.decoder = Generator3(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=4, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)



class StyleSwinVAE_v5_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v5_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [512])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator4
        self.decoder = Generator4(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=8, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)


    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        x = self.decoder(z).reshape(-1,  3, 32, 128, 128)

        return x

    

class StyleSwinVAE_v6_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v6_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [512])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator6
        self.decoder = Generator6(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=8, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)



class StyleSwinVAE_v7_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v7_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [512])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator7
        self.decoder = Generator7(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=8, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)
        
    

class StyleSwinVAE_v8_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v8_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [512])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator8
        self.decoder = Generator8(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=8, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)



class StyleSwinVAE_v9_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v9_128, self).__init__()
        print("vae type: StyleSwinVAE_v9_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [2048])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        from models.archs.styleswin.generator import Generator9
        n_mlp = vae_config.get("n_mlp", 8)
        self.decoder = Generator9(size=self.plane_shape[-1], plane_channel=self.plane_shape[1]*3, style_dim=self.z_shape[0], n_mlp=n_mlp, channel_multiplier=2,lr_mlp=0.01,enable_full_resolution=8,mlp_ratio=4,use_checkpoint=False,qkv_bias=True,qk_scale=None,drop_rate=0,attn_drop_rate=0)
        
    


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).

    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa

    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.

    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].

        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

        
class BilinearUpsampleV10(nn.Module):
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
        x = rearrange(x, 'b d h (w c) -> (b c) h w d', c=3)
        B, _, _, C = x.shape
        L = H * W

        x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L*4, C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # Add SPE    
        x = x.reshape(B , H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha
        x = x.permute(0, 2, 3, 1).contiguous()
        x = rearrange(x, '(b c) h w d -> b d h (w c)', c=3)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size * 3))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out
    

class AdaptiveInstanceNormV10(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input.transpose(1, 2))
        out = gamma * out + beta
        return out.transpose(1, 2)

        

class StyleTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
        'linear': NormLinearAttention,
    }

    def __init__(
        self,
        style_dim,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
        layer = 0,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
            layer = layer,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            layer = layer 
        )  # is self-attn if context is none
        self.norm1 = AdaptiveInstanceNormV10(dim, style_dim)
        self.norm2 = AdaptiveInstanceNormV10(dim, style_dim)
        self.norm3 = AdaptiveInstanceNormV10(dim, style_dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, styles, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if additional_tokens is not None:
            kwargs.update({"styles": styles})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        if context is None:
            return checkpoint(
                self._forward, [x, styles], self.parameters(), self.checkpoint
            )
        else:
            return checkpoint(
                self._forward, [x, styles, context], self.parameters(), self.checkpoint
            )

    def _forward(
        self, x, styles, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        x = (
            self.attn1(
                self.norm1(x, styles[:, 0]),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x, styles[:, 1]), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        x = self.ff(self.norm3(x, styles[:, 2])) + x
        return x

        
class StyleTransformerLayer(nn.Module):
    def __init__(
        self,
        style_dim,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="linear",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
        layer = 0,
        upsample = False,
        resolution = None
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.transformer_blocks = nn.ModuleList(
            [
                StyleTransformerBlock(
                    style_dim,
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    layer=layer
                )
                for d in range(depth)
            ]
        )
        
        if upsample:
            self.upsample = BilinearUpsampleV10(resolution, in_channels, in_channels)
        else:
            self.upsample = None

    def forward(self, x, context=None, styles=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        count = 0
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, styles[:, count:count+3, :], context=context[i])
            count += 3
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x = x + x_in
        if self.upsample is not None:
            x = self.upsample(x)
        return x



class StyleSwinVAE_v10_128(StyleSwinVAE_base):
    def __init__(self, vae_config) -> None:
        super(StyleSwinVAE_v10_128, self).__init__()
        print("vae type: StyleSwinVAE_v10_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4096])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)
        style_dim = vae_config.get("style_dim", 4096)

        self.transform_depth = transform_depth
        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        # 

        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64
        
        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:1]):
            stride = 2
            modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock(
                        h_dim,
                        dropout=0,
                        out_channels=h_dim,
                        use_conv=True,
                        dims=2,
                        use_checkpoint=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()),
            )
            in_channels = h_dim
            self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[1:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        layer=feature_size[i+1]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        

        ## build decoder
        
        self.decoder_in_linear = nn.Linear(self.z_shape[0], self.z_shape[0])
        
        # num of params: 1024 // 16-> 469; 768 // 12-> 371; 512 // 8 -> 293
        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 1024, 1024]
        # hidden_dims_decoder = [768, 768, 768, 768, 768]
        hidden_dims_decoder = [512, 512, 512, 512, 512]
                #feature size:  4,   8,    16,  32,  64

        feature_size_decoder = [4, 8, 16, 32, 64]

        self.input = ConstantInput(hidden_dims_decoder[2])

        self.decoders_up = nn.ModuleList()
        num_layers = 0

        for i, (h_dim, resolution) in enumerate(zip(hidden_dims_decoder, feature_size_decoder)):
            dim_head = h_dim // num_heads
            self.decoders_up.append(StyleTransformerLayer(
                                    style_dim,
                                    h_dim,
                                    num_heads,
                                    dim_head,
                                    depth=transform_depth,
                                    context_dim=h_dim,
                                    disable_self_attn=False,
                                    use_linear=True,
                                    attn_type="linear",
                                    use_checkpoint=True,
                                    layer = resolution,
                                    upsample = True if i < len(hidden_dims_decoder) - 1 else False,
                                    resolution = (resolution, resolution)
                                    
            ))
            in_channels = h_dim
            num_layers += transform_depth * 3

        self.to_planes = nn.Sequential(
                            nn.ConvTranspose2d(in_channels * 3,
                                                in_channels * 3,
                                                kernel_size=3,
                                                stride = 2,
                                                padding=1,
                                                output_padding=1,
                                                groups=3),
                            # nn.BatchNorm2d(in_channels * 3),
                            nn.SiLU(),
                            ResBlockGC(
                                in_channels * 3,
                                dropout=0,
                                out_channels=self.plane_shape[1]*3,
                                use_conv=True,
                                dims=2,
                                use_checkpoint=False,
                                groups=3
                            ),
                            nn.Tanh())

        self.decoders_upsample = nn.Sequential(
                            nn.ConvTranspose2d(in_channels,
                                                in_channels,
                                                kernel_size=3,
                                                stride = 2,
                                                padding=1,
                                                output_padding=1),
                            nn.SiLU())

        self.decoders_conv = nn.Sequential(
                                    ResBlockGC(
                                        in_channels * 3,
                                        dropout=0,
                                        out_channels=self.plane_shape[1]*3,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                        groups=3
                                    ),
                                    nn.Tanh())


        self.n_latent = num_layers

    def decode(self, z: Tensor) -> Tensor:
        styles = self.decoder_in_linear(z)

        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            latent = styles
        x = self.input(latent)
        count = 0
        for layer in self.decoders_up:
            x = layer(x, styles = latent[:,count:count+3*self.transform_depth,:])
            count = count + 3


        x = self.decoders_upsample(x)
        x = torch.concat(torch.chunk(x,3,dim=-1),dim=1)
        x = self.decoders_conv(x)
        x = torch.concat(torch.chunk(x.unsqueeze(1),3,dim=2), dim=1)
        return x


if __name__ == "__main__":
    # vae_model = BetaVAESpacial(plane_shape=[3, 32, 256, 256], z_shape=[4, 64, 64], kl_std=0.25, kl_weight=0.001).cuda()
    # vae_model = BetaVAE(plane_shape=[96, 256, 256], z_shape=[1024], kl_std=0.25).cuda()
    # vae_model = BetaVAESpacial2(plane_shape=[3, 32, 256, 256], z_shape=[3, 64, 64], kl_std=0.25, kl_weight=0.001).cuda()
    
    # vae_model = BetaVAESpacial2_Unet(plane_shape=[3, 32, 256, 256], z_shape=[3, 64, 64], kl_std=0.25, kl_weight=0.001).cuda()

    # vae_config = {"kl_std": 0.25,
    #             "kl_weight": 0.001,
    #             "plane_shape": [3, 32, 128, 128],
    #             # "z_shape": [32, 64, 64],
    #             "z_shape": [256, 32, 32],
    #             "num_heads": 16,
    #             "transform_depth": 1}

    
    vae_config = {"kl_std": 0.25,
                "kl_weight": 0.001,
                "plane_shape": [3, 32, 128, 128],
                # "z_shape": [32, 64, 64],
                "z_shape": [4096],
                "num_heads": 16,
                "transform_depth": 1}

    # vae_model = BetaVAERolloutTransformer_v2(vae_config).cuda()
    # vae_model = BetaVAERolloutTransformer_v2_128(vae_config).cuda()
    # vae_model = BetaVAERolloutTransformer_v3(vae_config).cuda()
    vae_model = StyleSwinVAE_v5_128(vae_config).cuda()
    

    input_tensor = torch.randn(4, 3, 32, 128, 128).cuda()
    out = vae_model(input_tensor)
    loss = vae_model.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    samples = vae_model.sample(2)
    print("samples shape: {}".format(samples[0].shape))
    import pdb;pdb.set_trace()

