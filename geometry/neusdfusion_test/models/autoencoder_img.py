import enum
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from einops import rearrange, reduce

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar("torch.tensor")

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
)

from models.archs.attention import SpatialTransformer, checkpoint, SpatialTransformer4, TransformerBlock, PatchEmbed
from models.archs.fpn import FPN_down, FPN_up
from models.autoencoder import ResBlock, ResBlockGC
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution



class PlaneDecoder_v1(nn.Module):
    def __init__(self, vae_config) -> None:
        super(PlaneDecoder_v1, self).__init__()

        plane_shape = vae_config.get("plane_shape", [3, 32, 128, 128])
        z_shape = vae_config.get("z_shape", [64, 8, 8])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)
        decoder_embed_dim = vae_config.get("decoder_embed_dim", 1024)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
       
        ## build decoder
        hidden_dims_decoder = [1024, 1024, 512]
                #feature size: 8,    16,  32,  64

        feature_size_decoder = [8, 16, 32]

        self.decoder_in_layer = nn.Sequential(
                            nn.Conv2d(self.z_shape[0], decoder_embed_dim*3, 1),
                            nn.BatchNorm2d(decoder_embed_dim*3),
                            nn.SiLU()
                            )

        in_channels = decoder_embed_dim
        # print("in_channels", in_channels, "z shape:", self.z_shape)
        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder):
            modules = []
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels*3,
                                                        h_dim*3,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1,
                                                        groups=3),
                            nn.GroupNorm(3, h_dim*3),
                            # nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            modules.append(nn.Sequential(SpatialTransformer(h_dim*3,
                                            num_heads,
                                            dim_head,
                                            depth=transform_depth,
                                            context_dim=h_dim*3,
                                            disable_self_attn=False,
                                            use_linear=True,
                                            attn_type="linear",
                                            use_checkpoint=True,
                                            layer = feature_size_decoder[i]
                                            ),
                            nn.BatchNorm2d(h_dim*3),
                            nn.SiLU()))
            in_channels = h_dim
            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels*3,
                                                        in_channels*3,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1,
                                                        groups=3),
                                    nn.GroupNorm(3, in_channels*3),
                                    nn.SiLU(),
                                    ResBlock(
                                        in_channels*3,
                                        dropout=0,
                                        out_channels=self.plane_shape[1]*3,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                        group_layer_num_in=3,
                                        group_layer_num_out=3,
                                        conv_group=3,
                                    ),
                                    nn.GroupNorm(3, self.plane_shape[1]*3),
                                    nn.Tanh()))

        # print(self.decoder_in_layer, self.decoders_up)

    def forward(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # print("input z shape", z.shape)
        x = self.decoder_in_layer(z)
        # print('decoder in layer', x.shape)

        for i, module in enumerate(self.decoders_up):
            # print("decoder up, x.shape", x.shape, module)
            x = module(x)

        # print("decoder out shape", x.shape)
        x = rearrange(x, 'b (p d) h w -> b p d h w', p=3)
        return x

    
class AutoencoderKL_v1(nn.Module):
    def __init__(self, vae_config):
        super().__init__()
        self.embed_dim = vae_config['embed_dim']
        ch_mult = vae_config['ch_mult']
        ch = vae_config['ch']
        resolution = vae_config['resolution']
        z_channels = vae_config['z_channels']
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        self.num_resolutions = len(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
        
        self.encoder = Encoder(**vae_config)
        self.decoder = PlaneDecoder_v1(vae_config)
        assert vae_config["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*vae_config["z_channels"], 2*self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, vae_config["z_channels"], 1)

        logvar_init = 0.0
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        # print("decoder post quant, z.shape", z.shape)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # print("decode size", dec.shape)
        return dec, posterior

    def loss_function(self, posterior):
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        return kl_loss
    
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

        z = torch.hstack((torch.zeros(num_samples, *(self.z_shape)), torch.ones(num_samples, *(self.z_shape))))
        # print("z shape: ", z.shape)
        z = DiagonalGaussianDistribution(z).sample().cuda()
        # print(z.shape)

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
        _, posterior = self.encode(x)
        return posterior.mode()

    
    
if __name__ == "__main__":

    vae_config = {
        "kl_std" : 0.25,
        "kl_weight" : 0.001,
        "plane_shape" : [3, 32, 128, 128],
        # "z_shape" :  [256, 32, 32],
        "num_heads": 16,
        "embed_dim": 64,
        "double_z": True,
        "z_channels": 64,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [ 1,1,2,2,4,4],
        "num_res_blocks": 2,
        "attn_resolutions": [16,8],
        "dropout": 0.0
        }

    vae_model = AutoencoderKL_v1(vae_config).cuda()
    
    input_tensor = torch.randn(4, 3, 256, 256).cuda()
    predicted_planes, posterior = vae_model(input_tensor)
    loss = vae_model.loss_function(posterior)
    print("loss: {}".format(loss))
    print("reconstruct shape: {}".format(predicted_planes.shape))
    samples = vae_model.sample(2)[0]
    print("samples shape: {}".format(samples.shape))
    import pdb;pdb.set_trace()