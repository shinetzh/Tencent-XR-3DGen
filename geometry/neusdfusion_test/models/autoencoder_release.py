import enum
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from einops import rearrange, reduce

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar("torch.tensor")

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.archs.attention_utils.CBAM import CBAMBlock
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
)

from models.archs.attention import SpatialTransformer, checkpoint

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        group_layer_num_in=32,
        group_layer_num_out=32,
        conv_group = 1,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels, group_layer_num_in),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1, groups=conv_group),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, group_layer_num_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, groups=conv_group)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, groups=conv_group
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, groups=conv_group)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
             self._forward, [x], self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h



class ResBlockGC(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        groups=3
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels, groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels * 3, 3, padding=1, groups=groups)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels * 3, groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels * 3, self.out_channels, 3, padding=1, groups=groups)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, groups=groups
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, groups=groups)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
             self._forward, [x], self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h


class GroupConv(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, stride=1,padding=0) -> None:
        super(GroupConv, self).__init__()
        self.conv = nn.Conv2d(3*in_channels, 3*out_channels, kernel_size, stride, padding,groups=3)
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        data = torch.concat(torch.chunk(data,3,dim=-1),dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data,3,dim=1),dim=-1)
        return data

class GroupConvTranspose(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, stride=1,padding=0,output_padding=0) -> None:
        super(GroupConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose2d(3*in_channels, 3*out_channels, kernel_size, stride, padding,output_padding,groups=3)
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        data = torch.concat(torch.chunk(data,3,dim=-1),dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data,3,dim=1),dim=-1)
        return data

class ResBlock_g(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        group_layer_num_in=32,
        group_layer_num_out=32,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels, group_layer_num_in),
            nn.SiLU(),
            GroupConv(channels, self.out_channels, 3, padding=1)
            # conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, group_layer_num_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                GroupConv(self.out_channels, self.out_channels, 3, padding=1)
                # conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            # self.skip_connection = conv_nd(
            #     dims, channels, self.out_channels, 3, padding=1
            # )
            self.skip_connection = GroupConv(channels, self.out_channels, 3, padding=1)
        else:
            # self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
            self.skip_connection = GroupConv(channels, self.out_channels,1)
    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
             self._forward, [x], self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        h = self.skip_connection(x) + h
        return h
    

class AutoencoderKL(nn.Module):
    def __init__(self, vae_config) -> None:
        super(AutoencoderKL, self).__init__()
        print("vae type: AutoencoderKL")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_dim = vae_config.get("z_dim", 4)
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)
        hidden_dims = vae_config.get("hidden_dims", [512, 512, 512, 512])
        hidden_dims_decoder = vae_config.get("hidden_dims_decoder", [512, 512, 512, 512])

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_dim
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        feature_size = [64,  32,  16,   8, 4]
        
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
        for i, h_dim in enumerate(hidden_dims):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
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
                                                        layer=feature_size[i]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1),
                            nn.BatchNorm2d(in_channels),
                            nn.SiLU(),
                            nn.Conv2d(in_channels, out_channels=self.z_shape*2, kernel_size=1),
                            )
                
        self.decoder_in_layer = nn.Sequential(
                                    nn.Conv2d(self.z_shape, out_channels=hidden_dims_decoder[0], kernel_size=1),
                                    nn.BatchNorm2d(hidden_dims_decoder[0]),
                                    nn.SiLU(),
                                    nn.Conv2d(hidden_dims_decoder[0], out_channels=hidden_dims_decoder[0], kernel_size=1),
                                    nn.BatchNorm2d(hidden_dims_decoder[0]),
                                    nn.SiLU()
                                    )
        

        ## build decoder
        feature_size_decoder = [8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            if i > 0:
                modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                            h_dim,
                                                            kernel_size=3,
                                                            stride = 2,
                                                            padding=1,
                                                            output_padding=1),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
            modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                            num_heads,
                                            dim_head,
                                            depth=transform_depth,
                                            context_dim=h_dim,
                                            disable_self_attn=False,
                                            use_linear=True,
                                            attn_type="linear",
                                            use_checkpoint=True,
                                            layer = feature_size_decoder[i]
                                            ),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            in_channels = h_dim

            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.SiLU(),
                                    ))

        self.decoders_up_group = nn.Sequential(
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


    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        planes = rearrange(enc_input, 'b p d h w -> b d h (w p)', p=3)
        feature = self.in_layer(planes)

        for i, module in enumerate(self.encoders_down):
            feature = module(feature)

        feature = self.encoder_out_layer(feature)
        mu, log_var = torch.split(feature, self.z_shape, dim=1)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        x = self.decoder_in_layer(z) # b 512 h (w 3)

        for i, module in enumerate(self.decoders_up):
            x = module(x)

        x = rearrange(x, 'b d h (w p) -> b p d h w', p=3)
        x = rearrange(x, 'b p d h w -> b (p d) h w')

        x = self.decoders_up_group(x)
        x = rearrange(x, 'b (p d) h w -> b p d h w', p=3)
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

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]
        #print("recon, data shape: ", recons.shape, data.shape)
        #recons_loss = F.mse_loss(recons, data)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

      
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

        # # return samples
        # z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        # eps = torch.randn(num_samples, *(z_rollout_shape)).cuda()
        # z = eps * self.kl_std
        # samples = self.decode(z)
        # return samples, z

        z_rollout_shape = [self.z_shape, 8, 3*8]
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

    def get_latent(self, x, return_dist=False):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        if return_dist:
            return torch.cat((mu, log_var), dim=-1)
        
        z = self.reparameterize(mu, log_var)
        return z



from models.cnn import Encoder, Decoder
class AutoencoderCNN(nn.Module):
    def __init__(self, vae_config) -> None:
        super(AutoencoderCNN, self).__init__()
        print("vae type: AutoencoderCNN")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_dim = vae_config.get("z_dim", 4)
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)
        hidden_dims = vae_config.get("hidden_dims", [512, 512, 512, 512])
        hidden_dims_decoder = vae_config.get("hidden_dims_decoder", [512, 512, 512, 512])

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = 4
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        resolution = 128
        z_channels = 4
        in_channels = 96
        ch = 256
        out_ch = 3
        ch_mult = (1,2,2,4,4)
        num_res_blocks = 2
        attn_resolutions = ()
        dropout = 0.0
        resamp_with_conv = True
        double_z = True
        use_linear_attn = False
        attn_type = "vanilla"
        self.encoder = Encoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult,
                        num_res_blocks=num_res_blocks,
                        attn_resolutions=attn_resolutions,
                        dropout=dropout, resamp_with_conv=resamp_with_conv,
                        in_channels=in_channels, resolution=resolution,
                        z_channels=z_channels, double_z=double_z,
                        use_linear_attn=use_linear_attn, attn_type=attn_type)

        resolution = 128
        in_channels = 96
        ch = 128
        out_ch = 96
        ch_mult = (1,2,2,4,4)
        num_res_blocks = 2
        attn_resolutions = ()
        dropout = 0.0
        resamp_with_conv = True
        give_pre_end = False
        tanh_out = True
        use_linear_attn = False
        attn_type = "vanilla"
        self.decoder = Decoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult,
                        num_res_blocks=num_res_blocks,
                        attn_resolutions=attn_resolutions,
                        dropout=dropout, resamp_with_conv=resamp_with_conv,
                        in_channels=in_channels, resolution=resolution,
                        z_channels=z_channels, give_pre_end=give_pre_end,
                        tanh_out=tanh_out, use_linear_attn=use_linear_attn,
                        attn_type=attn_type)


    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        planes = rearrange(enc_input, 'b p d h w -> b (p d) h w', p=3)
        feature = self.encoder(planes)
        mu, log_var = torch.split(feature, self.z_shape, dim=1)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        x = self.decoder(z)
        x = rearrange(x, 'b (p d) h w -> b p d h w', p=3)
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

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]
        #print("recon, data shape: ", recons.shape, data.shape)
        #recons_loss = F.mse_loss(recons, data)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

      
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

        # # return samples
        # z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        # eps = torch.randn(num_samples, *(z_rollout_shape)).cuda()
        # z = eps * self.kl_std
        # samples = self.decode(z)
        # return samples, z

        z_rollout_shape = [self.z_shape, 8, 3*8]
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

    def get_latent(self, x, return_dist=False):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        if return_dist:
            return torch.cat((mu, log_var), dim=-1)
        
        z = self.reparameterize(mu, log_var)
        return z
    

class AutoencoderKLSAPE(nn.Module):
    def __init__(self, vae_config) -> None:
        super(AutoencoderKLSAPE, self).__init__()
        print("vae type: AutoencoderKLSAPE")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_dim = vae_config.get("z_dim", 4)
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)
        hidden_dims = vae_config.get("hidden_dims", [512, 512, 512, 512])
        hidden_dims_decoder = vae_config.get("hidden_dims_decoder", [512, 512, 512, 512])

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_dim
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        feature_size = [64,  32,  16,   8, 4]
        
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
        for i, h_dim in enumerate(hidden_dims):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
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
                                                        layer=feature_size[i],
                                                        sape=True
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim
        
        self.encoder_out_layer = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1),
                            nn.BatchNorm2d(in_channels),
                            nn.SiLU(),
                            nn.Conv2d(in_channels, out_channels=self.z_shape*2, kernel_size=1),
                            )
                
        self.decoder_in_layer = nn.Sequential(
                                    nn.Conv2d(self.z_shape, out_channels=hidden_dims_decoder[0], kernel_size=1),
                                    nn.BatchNorm2d(hidden_dims_decoder[0]),
                                    nn.SiLU(),
                                    nn.Conv2d(hidden_dims_decoder[0], out_channels=hidden_dims_decoder[0], kernel_size=1),
                                    nn.BatchNorm2d(hidden_dims_decoder[0]),
                                    nn.SiLU()
                                    )
        

        ## build decoder
        feature_size_decoder = [8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            if i > 0:
                modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                            h_dim,
                                                            kernel_size=3,
                                                            stride = 2,
                                                            padding=1,
                                                            output_padding=1),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
            modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                            num_heads,
                                            dim_head,
                                            depth=transform_depth,
                                            context_dim=h_dim,
                                            disable_self_attn=False,
                                            use_linear=True,
                                            attn_type="linear",
                                            use_checkpoint=True,
                                            layer = feature_size_decoder[i],
                                            sape=True
                                            ),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            in_channels = h_dim

            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.SiLU(),
                                    ))

        self.decoders_up_group = nn.Sequential(
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


    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        planes = rearrange(enc_input, 'b p d h w -> b d h (w p)', p=3)
        feature = self.in_layer(planes)

        for i, module in enumerate(self.encoders_down):
            feature = module(feature)

        feature = self.encoder_out_layer(feature)
        mu, log_var = torch.split(feature, self.z_shape, dim=1)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        x = self.decoder_in_layer(z) # b 512 h (w 3)

        for i, module in enumerate(self.decoders_up):
            x = module(x)

        x = rearrange(x, 'b d h (w p) -> b p d h w', p=3)
        x = rearrange(x, 'b p d h w -> b (p d) h w')

        x = self.decoders_up_group(x)
        x = rearrange(x, 'b (p d) h w -> b p d h w', p=3)
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

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]
        #print("recon, data shape: ", recons.shape, data.shape)
        #recons_loss = F.mse_loss(recons, data)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

      
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

        # # return samples
        # z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
        # eps = torch.randn(num_samples, *(z_rollout_shape)).cuda()
        # z = eps * self.kl_std
        # samples = self.decode(z)
        # return samples, z

        z_rollout_shape = [self.z_shape, 8, 3*8]
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

    def get_latent(self, x, return_dist=False):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''
        mu, log_var = self.encode(x)
        if return_dist:
            return torch.cat((mu, log_var), dim=-1)
        
        z = self.reparameterize(mu, log_var)
        return z
    

    
if __name__ == "__main__":

    vae_config = {"kl_std": 0.25,
                "kl_weight": 0.001,
                "plane_shape": [3, 32, 128, 128],
                "hidden_dims": [512, 512, 512, 512],
                "hidden_dims_decoder": [512, 512, 512, 512],
                "z_dim": 4,
                "num_heads": 8,
                "transform_depth": 1}

    vae_model = AutoencoderKLSAPE(vae_config).cuda()
    

    input_tensor = torch.randn(4, 3, 32, 128, 128).cuda()
    out = vae_model(input_tensor)
    loss = vae_model.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    samples = vae_model.sample(2)
    print("samples shape: {}".format(samples[0].shape))
    breakpoint()

    # vae_config = {"kl_std": 0.25,
    #             "kl_weight": 0.001,
    #             "plane_shape": [3, 32, 128, 128],
    #             "hidden_dims": [512, 512, 512, 512],
    #             "hidden_dims_decoder": [512, 512, 512, 512],
    #             "z_dim": 4,
    #             "num_heads": 8,
    #             "transform_depth": 1}

    # vae_model = AutoencoderCNN(vae_config).cuda()
    

    # input_tensor = torch.randn(4, 3, 32, 128, 128).cuda()
    # out = vae_model(input_tensor)
    # loss = vae_model.loss_function(*out)
    # print("loss: {}".format(loss))
    # print("z shape: {}".format(out[-1].shape))
    # print("reconstruct shape: {}".format(out[0].shape))
    # samples = vae_model.sample(2)
    # print("samples shape: {}".format(samples[0].shape))
    # import pdb;pdb.set_trace