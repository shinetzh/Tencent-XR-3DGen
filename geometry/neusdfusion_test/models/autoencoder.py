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

# from models.archs.attention_utils.SelfAttention import ScaledDotProductAttention
# from models.archs.attention_utils.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from models.archs.attention_utils.CBAM import CBAMBlock
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
)

from models.archs.attention import SpatialTransformer, checkpoint, SpatialTransformer2, SpatialTransformer3, SpatialTransformer4, TransformerBlock, PatchEmbed, GroupModulation, TransformerLayer
from models.archs.fpn import FPN_down, FPN_up

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
    
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BetaVAESpacial2_Unet(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 kl_std=1.0,
                 kl_weight=0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [3, 64, 64],
                 num_res_block = 4) -> None:
        super(BetaVAESpacial2_Unet, self).__init__()
        print("vae type: BetaVAESpacial2_Unet")
        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        if len(plane_shape) == 4:
            self.in_channels = plane_shape[0] * plane_shape[1]
        else:
            self.in_channels = plane_shape[0]

        self.kl_std = kl_std
        self.kl_weight = kl_weight

        #print("kl standard deviation: ", self.kl_std)
        self.plane_res_encode = plane_shape[-1] // 64
        self.pre_square = self.plane_res_encode * self.plane_res_encode

        hidden_dims = [128, 128, 128, 256, 256, 512] 
                  ### [128, 64,   32,  16,  8,   4]

        self.hidden_dims = hidden_dims

        # Build Encoder
        in_channels = self.in_channels
        self.encoder1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dims[0], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU())
        self.encoder2 = nn.Sequential(
                    nn.Conv2d(hidden_dims[0], out_channels=hidden_dims[1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[1]),
                    nn.LeakyReLU())

        in_channels = hidden_dims[1]
        modules = []
        for h_dim in hidden_dims[2:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder3 = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.pre_square, z_shape[0] * z_shape[1] * z_shape[2])  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        self.fc_var = nn.Linear(hidden_dims[-1]*self.pre_square, z_shape[0] * z_shape[1] * z_shape[2]) 


        # Build Decoder
        modules = []

        hidden_dims.reverse()
        self.decoder_input = nn.Sequential(
                    nn.Conv2d(z_shape[0], out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU())

        self.decoders = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if i in [1, 3]:
                self.decoders.append(nn.Sequential(*modules))
                modules = []
                if i == 1:
                    upfeature_channel = hidden_dims[-2]
                elif i == 3:
                    upfeature_channel = hidden_dims[-1]
                
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i] + upfeature_channel,
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        ResBlock(
                            hidden_dims[i],
                            dropout=0,
                            out_channels=hidden_dims[i + 1],
                            use_conv=False,
                            dims=2,
                            use_checkpoint=False,
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

        self.decoders.append(nn.Sequential(*modules))

        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels, # changed from 3 to in_channels
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


        #### up branch
        modules = []
        modules.append(nn.Sequential(
                        nn.Conv2d(z_shape[0], out_channels= 32, 
                                                kernel_size= 3, padding= 1),
                        ResBlock(
                            32,
                            dropout=0,
                            out_channels=hidden_dims[-2],
                            use_conv=False,
                            dims=2,
                            use_checkpoint=False,
                        ),
                        nn.BatchNorm2d(hidden_dims[-2]),
                        nn.LeakyReLU(),
                        CBAMBlock(channel=hidden_dims[-2],reduction=16,kernel_size=5),
                        # SimplifiedScaledDotProductAttention(d_model=hidden_dims[-2], h=8),
                        # ResBlock(
                        #     hidden_dims[-2],
                        #     dropout=0,
                        #     out_channels=hidden_dims[-2],
                        #     use_conv=False,
                        #     dims=2,
                        #     use_checkpoint=False,
                        # ),
                        nn.BatchNorm2d(hidden_dims[-2]),
                        nn.LeakyReLU()
                        ))
        self.up64 = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[-2],
                                        hidden_dims[-1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        nn.BatchNorm2d(hidden_dims[-1]),
                        nn.LeakyReLU(),
                        ResBlock(
                            hidden_dims[-1],
                            dropout=0,
                            out_channels=hidden_dims[-1],
                            use_conv=False,
                            dims=2,
                            use_checkpoint=False,
                        ),
                        nn.BatchNorm2d(hidden_dims[-1]),
                        nn.LeakyReLU(),
                        CBAMBlock(channel=hidden_dims[-1], reduction=16, kernel_size=5),
                        # SimplifiedScaledDotProductAttention(d_model=hidden_dims[-1], h=8),
                        # ResBlock(
                        #     hidden_dims[-1],
                        #     dropout=0,
                        #     out_channels=hidden_dims[-1],
                        #     use_conv=False,
                        #     dims=2,
                        #     use_checkpoint=False,
                        # ),
                        nn.BatchNorm2d(hidden_dims[-1]),
                        nn.LeakyReLU()
                        ))
        self.up128 = nn.Sequential(*modules)


        #print(self)

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
            result = torch.concat(plane_list, dim=1)
        result1 = self.encoder1(result)
        result2 = self.encoder2(result1)
        result = self.encoder3(result2)
        result = torch.flatten(result, start_dim=1)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result).reshape(-1, *self.z_shape)
        log_var = self.fc_var(result).reshape(-1, *self.z_shape)


        return mu, log_var, result1, result2

    def decode(self, z: Tensor) -> Tensor:
        '''
        z: latent vector: B, D (D = latent_dim*3)
        '''
        upfeature64 = self.up64(z)
        upfeature128 = self.up128(upfeature64)

        result = self.decoder_input(z)
        for i, decoder in enumerate(self.decoders):
            if i == 1:
                result = torch.cat([result, upfeature64], dim=1)
            elif i == 2:
                result = torch.cat([result, upfeature128], dim=1)
            result = decoder(result)
        result = self.final_layer(result)
        if self.plane_dim == 5:
            result = result.view(-1, *self.plane_shape)
        return result, upfeature64, upfeature128

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
        mu, log_var, result128, result64 = self.encode(data)
        z = self.reparameterize(mu, log_var)
        result, upfeature64, upfeature128 = self.decode(z)
        return  [result, data, mu, log_var, z, upfeature64, upfeature128, result64, result128]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        self.num_iter += 1
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
            # std = torch.exp(0.5 * log_var)
            # torch.autograd.set_detect_anomaly(True)
            std = torch.exp(torch.clamp(0.5 * log_var, max=10)) + 1e-6
            gt_dist = torch.distributions.normal.Normal( torch.zeros_like(mu), torch.ones_like(std)*self.kl_std )
            sampled_dist = torch.distributions.normal.Normal( mu, std )
            #gt_dist = normal_dist.sample(log_var.shape)
            #print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist) # reversed KL
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()
        

        ##### feature loss
        upfeature64, upfeature128, result64, result128 = args[5], args[6], args[7], args[8]
        feature_loss = (result64 - upfeature64).abs().mean() + (result128 - upfeature128).abs().mean()

        return self.kl_weight * kl_loss + feature_loss * 10

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
        # gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, self.latent_dim), 
        #                                             torch.ones(num_samples, self.latent_dim)*self.kl_std)

        # z = gt_dist.sample().cuda()
        # samples = self.decode(z)
        # return samples
        eps = torch.randn(num_samples, *(self.z_shape)).cuda()
        z = eps * self.kl_std
        samples, _, _ = self.decode(z)
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
        mu, log_var, _, _ = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z 


class BetaVAESpacial2(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 kl_std=1.0,
                 kl_weight=0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [3, 64, 64],
                 num_res_block = 4) -> None:
        super(BetaVAESpacial2, self).__init__()
        print("vae type: BetaVAESpacial2")
        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        if len(plane_shape) == 4:
            self.in_channels = plane_shape[0] * plane_shape[1]
        else:
            self.in_channels = plane_shape[0]

        self.kl_std = kl_std
        self.kl_weight = kl_weight

        #print("kl standard deviation: ", self.kl_std)
        self.plane_res_encode = plane_shape[-1] // 64
        self.pre_square = self.plane_res_encode * self.plane_res_encode

        modules = []
        hidden_dims = [128, 128, 128, 256, 256, 512]

        self.hidden_dims = hidden_dims

        # Build Encoder
        in_channels = self.in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.pre_square, z_shape[0] * z_shape[1] * z_shape[2])  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        self.fc_var = nn.Linear(hidden_dims[-1]*self.pre_square, z_shape[0] * z_shape[1] * z_shape[2]) 


        # Build Decoder
        modules = []

        hidden_dims.reverse()
        self.decoder_input = nn.Sequential(
                    nn.Conv2d(z_shape[0], out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU())

        for i in range(len(hidden_dims) - 1):
            if i in [1, 3]:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        ResBlock(
                            hidden_dims[i],
                            dropout=0,
                            out_channels=hidden_dims[i + 1],
                            use_conv=False,
                            dims=2,
                            use_checkpoint=False,
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels, # changed from 3 to in_channels
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


        #print(self)

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
            result = torch.concat(plane_list, dim=1)
        result = self.encoder(result)  # [B, D, 8, 8]
        result = torch.flatten(result, start_dim=1) # ([B, D*64])
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result).reshape(-1, *self.z_shape)
        log_var = self.fc_var(result).reshape(-1, *self.z_shape)


        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z: latent vector: B, D (D = latent_dim*3)
        '''
        result = self.decoder_input(z) # ([b, hidden_dim[-1], z_shape[-2], z_shape[-1])
        # result = result.view(-1, int(result.shape[-1]/self.pre_square), self.plane_res_encode, self.plane_res_encode)  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        result = self.decoder(result)
        result = self.final_layer(result) # ([32, D, resolution, resolution])
        if self.plane_dim == 5:
            result = result.view(-1, *self.plane_shape)
        return result

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
        return  [self.decode(z), data, mu, log_var, z]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        self.num_iter += 1
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
            # std = torch.exp(0.5 * log_var)
            # torch.autograd.set_detect_anomaly(True)
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
        # gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, self.latent_dim), 
        #                                             torch.ones(num_samples, self.latent_dim)*self.kl_std)

        # z = gt_dist.sample().cuda()
        # samples = self.decode(z)
        # return samples
        eps = torch.randn(num_samples, *(self.z_shape)).cuda()
        z = eps * self.kl_std
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

class BetaVAESpacial(nn.Module):
    def __init__(self,
                 kl_std=0.25,
                 kl_weight = 0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [4, 64, 64],
                 num_res_block = 4,
                 dim_head = 16,
                 transform_depth=4) -> None:
        super(BetaVAESpacial, self).__init__()

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        if len(plane_shape) == 4:
            self.in_channels = plane_shape[0] * plane_shape[1]
        else:
            self.in_channels = plane_shape[0]
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [[512, 512], [512, 512], [256, z_shape[0]]]
        self.hidden_dims = hidden_dims

        # Build Encoder
        modules = []
        in_channels = self.in_channels
        for h_dim1, h_dim2 in hidden_dims[:-1]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim1, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim1),
                    nn.LeakyReLU())
            )
            for i in range(num_res_block):
                modules.append(
                    ResBlock(
                        h_dim1,
                        dropout=0,
                        out_channels=h_dim1,
                        use_conv=False,
                        dims=2,
                        use_checkpoint=False,
                    )
                )
            modules.append(
                nn.Sequential(
                    nn.Conv2d(h_dim1, out_channels=h_dim2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim2),
                    nn.LeakyReLU())
            )
            in_channels = h_dim2
        h_dim1, h_dim2 = hidden_dims[-1]
        modules.append(
            nn.Sequential(
              nn.Conv2d(in_channels, out_channels=h_dim1, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(h_dim1),
              nn.LeakyReLU())
        )

        num_heads = h_dim1 // dim_head
        modules.append(
            SpatialTransformer(h_dim1,
                            num_heads,
                            dim_head,
                            depth=transform_depth,
                            context_dim=h_dim1,
                            disable_self_attn=False,
                            use_linear=True,
                            attn_type="softmax-xformers",
                            use_checkpoint=True,
                            )
        )

        modules.append(
            nn.Sequential(nn.Conv2d(h_dim1, out_channels=h_dim2 * 2, kernel_size=3, stride=1, padding=1))
        )

        self.encoder = nn.Sequential(*modules)


        # Build Decoder
        modules = []
        h_dim2, h_dim1 = hidden_dims[-1]
        modules.append(
            nn.Sequential(
              nn.Conv2d(h_dim1, out_channels=h_dim2, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(h_dim2),
              nn.LeakyReLU())
        )
        in_channels = h_dim2

        hidden_dims.reverse()
        for h_dim2, h_dim1 in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                  nn.Conv2d(in_channels, out_channels=h_dim1, kernel_size=3, stride=1, padding=1),
                  nn.BatchNorm2d(h_dim1),
                  nn.LeakyReLU())
            )
            for i in range(num_res_block):
                modules.append(
                    ResBlock(
                        h_dim1,
                        dropout=0,
                        out_channels=h_dim1,
                        use_conv=False,
                        dims=2,
                        use_checkpoint=False,
                    )
                )
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(h_dim1,
                                    h_dim2,
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1),
                    nn.BatchNorm2d(h_dim2),
                    nn.LeakyReLU())
            )
            in_channels = h_dim2



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels= self.in_channels, # changed from 3 to in_channels
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


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
            result = torch.concat(plane_list, dim=1)
        result = self.encoder(result)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        encode_channel = self.z_shape[0]
        mu = result[:, :encode_channel, ...]
        log_var = result[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z: latent vector: B, D (D = latent_dim*3)
        '''
        result = self.decoder(z)
        result = self.final_layer(result) # ([32, D, resolution, resolution])
        if self.plane_dim == 5:
            result = result.view(-1, *self.plane_shape)
        return result

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
        return  [self.decode(z), data, mu, log_var, z]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        mu = args[2]
        log_var = args[3]

        #print("recon, data shape: ", recons.shape, data.shape)
        #recons_loss = F.mse_loss(recons, data)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0
      
        if self.kl_std == 'zero_mean':
            latent = self.reparameterize(mu, log_var) 
            #print("latent shape: ", latent.shape) # (B, dim)
            l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
            kl_loss = l2_size_loss / latent.shape[0]

        else:
            # std = torch.exp(0.5 * log_var)
            # torch.autograd.set_detect_anomaly(True)
            std = torch.exp(torch.clamp(0.5 * log_var, max=10))
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
        # gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, self.latent_dim), 
        #                                             torch.ones(num_samples, self.latent_dim)*self.kl_std)

        # z = gt_dist.sample().cuda()
        # samples = self.decode(z)
        # return samples
        eps = torch.randn(num_samples, *(self.z_shape)).cuda()
        z = eps * self.kl_std
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

class BetaVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 kl_std=1.0,
                 kl_weight=0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [1024]) -> None:
        super(BetaVAE, self).__init__()

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        if len(plane_shape) == 4:
            self.in_channels = plane_shape[0] * plane_shape[1]
        else:
            self.in_channels = plane_shape[0]

        self.kl_std = kl_std
        self.kl_weight = kl_weight

        #print("kl standard deviation: ", self.kl_std)
        self.plane_res_encode = plane_shape[-1] // 32
        self.pre_square = self.plane_res_encode * self.plane_res_encode

        modules = []
        hidden_dims = [512, 512, 512, 512, 512]

        self.hidden_dims = hidden_dims

        # Build Encoder
        in_channels = self.in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.pre_square, self.z_shape[0])  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        self.fc_var = nn.Linear(hidden_dims[-1]*self.pre_square, self.z_shape[0]) 

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.z_shape[0], hidden_dims[-1] * self.pre_square) 

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels, # changed from 3 to in_channels
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


        #print(self)

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
            result = torch.concat(plane_list, dim=1)

        # breakpoint()

        result = self.encoder(result)  # [B, D, 2, 2]
        
        # breakpoint()

        result = torch.flatten(result, start_dim=1) # ([B, D*4])
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z: latent vector: B, D (D = latent_dim*3)
        '''
        result = self.decoder_input(z) # ([32, D*4])
        result = result.view(-1, int(result.shape[-1]/self.pre_square), self.plane_res_encode, self.plane_res_encode)  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        result = self.decoder(result)
        result = self.final_layer(result) # ([32, D, resolution, resolution])
        if self.plane_dim == 5:
            result = result.view(-1, *self.plane_shape)
        return result

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
        return  [self.decode(z), data, mu, log_var, z]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        self.num_iter += 1
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
            # std = torch.exp(0.5 * log_var)
            # torch.autograd.set_detect_anomaly(True)
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
        # gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, self.latent_dim), 
        #                                             torch.ones(num_samples, self.latent_dim)*self.kl_std)

        # z = gt_dist.sample().cuda()
        # samples = self.decode(z)
        # return samples
        eps = torch.randn(num_samples, *(self.z_shape)).cuda()
        z = eps * self.kl_std
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

class BetaVAE2(nn.Module):
    num_iter = 0
    def __init__(self,
                 vae_config) -> None:
        super(BetaVAE2, self).__init__()
        vae_type = vae_config["vae_type"]
        kl_std = vae_config["kl_std"]
        kl_weight = vae_config["kl_weight"]
        plane_shape = vae_config["plane_shape"]
        z_shape = vae_config["z_shape"]
        self.vae_geo = BetaVAE(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
        self.vae_color = BetaVAE(plane_shape=plane_shape, z_shape=z_shape, kl_std=kl_std, kl_weight=kl_weight)
        
        #### vae geo load ckpt
        vae_geo_ckpt = vae_config.get("vae_geo_ckpt", None)
        if vae_geo_ckpt:
            print("vae geo load ckpt from {}".format(vae_config["vae_geo_ckpt"]))
            vae_geo_dict = torch.load(vae_config["vae_geo_ckpt"], map_location='cuda')
            vae_geo_state_dict = {}
            for key in list(vae_geo_dict['state_dict'].keys()):
                if 'vae_model' in key:
                    key_new = key.replace("vae_model.", "")
                    vae_geo_state_dict[key_new] = vae_geo_dict['state_dict'][key]
            self.vae_geo.load_state_dict(vae_geo_state_dict, strict=True)

        #### vae color load ckpt
        vae_color_ckpt = vae_config.get("vae_color_ckpt", None)
        if vae_color_ckpt:
            print("vae color load ckpt from {}".format(vae_config["vae_color_ckpt"]))
            vae_color_dict = torch.load(vae_config["vae_color_ckpt"], map_location='cuda')
            vae_color_state_dict = {}
            for key in list(vae_color_dict['state_dict'].keys()):
                if 'vae_model' in key:
                    key_new = key.replace("vae_model.", "")
                    vae_color_state_dict[key_new] = vae_color_dict['state_dict'][key]
            self.vae_color.load_state_dict(vae_color_state_dict, strict=True)
            self.vae_geo.eval()
            self.vae_color.eval()

    def encode(self, enc_input_list):
        enc_input_geo, enc_input_color = enc_input_list
        [mu_geo, log_var_geo] = self.vae_geo.encode(enc_input_geo)
        [mu_color, log_var_color] = self.vae_color.encode(enc_input_color)

        return [[mu_geo, log_var_geo], [mu_color, log_var_color]]

    def decode(self, z_list) -> Tensor:
        z_geo, z_color = z_list
        result_geo = self.vae_geo.decode(z_geo)
        result_color = self.vae_color.decode(z_color)
        return [result_geo, result_color]

    def reparameterize(self, mu_logvar_list) -> Tensor:
        mu_logvar_geo, mu_logvar_color = mu_logvar_list
        return [self.vae_geo.reparameterize(*mu_logvar_geo), 
                self.vae_color.reparameterize(*mu_logvar_color)]

    def forward(self, data_list) -> Tensor:
        mu_logvar_list = self.encode(data_list)
        z_list = self.reparameterize(mu_logvar_list)
        return [self.decode(z_list), data_list, mu_logvar_list[0], mu_logvar_list[1], z_list]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        self.num_iter += 1
        mu_logvar_geo = args[2]
        mu_logvar_color = args[3]
        #print("recon, data shape: ", recons.shape, data.shape)
        #recons_loss = F.mse_loss(recons, data)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return self.vae_geo.loss_function(None, None, mu_logvar_geo[0], mu_logvar_geo[1], None) + \
                self.vae_color.loss_function(None, None, mu_logvar_color[0], mu_logvar_color[1], None)

    def sample(self,
               num_samples:int,
                **kwargs) -> Tensor:
        samples_geo, z_geo = self.vae_geo.sample(num_samples)
        samples_color, z_color = self.vae_color.sample(num_samples)
        return [[samples_geo, z_geo], [samples_color, z_color]]


    def generate(self, x_list, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return [self.vae_geo.forward(x_list[0])[0], self.vae_color.forward(x_list[1][0])]

    def get_latent(self, x_list):
        '''
        given input x, return the latent code
        x:  [B x C x H x W]
        return: [B x latent_dim]
        '''

        return [self.vae_geo.get_latent(x_list[0]), self.vae_color.get_latent(x_list[1])]

class BetaVAERolloutTransformer(nn.Module):
    def __init__(self,
                 kl_std=0.25,
                 kl_weight=0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [4, 64, 64],
                 dim_head = 16,
                 transform_depth=2) -> None:
        super(BetaVAERolloutTransformer, self).__init__()
        print("vae type: BetaVAERolloutTransformer")
        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [128, 256, 256, 8, 512, 512]
        self.hidden_dims = hidden_dims
        if self.plane_dim == 4:
            self.in_channels = self.plane_shape[0] // 3
        elif self.plane_dim == 5:
            self.in_channels = self.plane_shape[1]
        in_channels = self.in_channels
        # Build Encoder
        self.encoders = nn.ModuleList()
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            stride = 2
            if i == 2: ### size 64
                self.encoders.append(nn.Sequential(*modules))
                modules = []
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            if i in [3, 4, 5]:
                # todo add transformer block
                num_heads = h_dim // dim_head
                modules.append(nn.Sequential(
                    SpatialTransformer(h_dim,
                                    num_heads,
                                    dim_head,
                                    depth=transform_depth,
                                    context_dim=h_dim,
                                    disable_self_attn=False,
                                    use_linear=True,
                                    attn_type="softmax-xformers",
                                    use_checkpoint=True,
                                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                    ))
            in_channels = h_dim

        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels=self.z_shape[0] * 2, kernel_size=3, stride=stride, padding=1))
        )

        self.encoder = nn.Sequential(*modules)



        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.pre_square) 
        self.decoder_input = nn.Conv2d(self.z_shape[0], hidden_dims[-1], 1, 1, 0)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if i in [1, 3]:
              modules.append(
                  nn.Sequential(
                      nn.ConvTranspose2d(hidden_dims[i],
                                      hidden_dims[i + 1],
                                      kernel_size=3,
                                      stride = 2,
                                      padding=1,
                                      output_padding=1),
                      nn.BatchNorm2d(hidden_dims[i + 1]),
                      nn.LeakyReLU())
              )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels, # changed from 3 to in_channels
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


        #print(self)

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
        
        result = self.encoder(result)  # [B, D, 2, 2]
        encode_channel = self.z_shape[0]
        mu = result[:, :encode_channel, ...]
        log_var = result[:, encode_channel:, ...]

        # result = torch.flatten(result, start_dim=1) # ([B, D*4])

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # mu = self.fc_mu(result)
        # log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z: latent vector: B, D (D = latent_dim*3)
        '''
        result = self.decoder_input(z) # ([32, D*4])
        # result = result.view(-1, int(result.shape[-1]/self.pre_square), self.plane_res_encode, self.plane_res_encode)  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        result = self.decoder(result)
        result = self.final_layer(result) # ([32, D, resolution, resolution])

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w].unsqueeze(1),
                                result[..., plane_w : plane_w * 2].unsqueeze(1),
                                result[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w],
                                result[..., plane_w : plane_w * 2],
                                result[..., plane_w * 2 : plane_w * 3],], dim=1)
        return result

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
        # gt_dist = torch.distributions.normal.Normal(torch.zeros(num_samples, self.latent_dim), 
        #                                             torch.ones(num_samples, self.latent_dim)*self.kl_std)

        # z = gt_dist.sample().cuda()
        # samples = self.decode(z)
        # return samples
        eps = torch.randn(num_samples, *(self.z_shape)).cuda()
        z = eps * self.kl_std
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

class BetaVAERolloutTransformer_v2(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v2, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v2")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 2)


        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [128, 256, 256, 512, 1024, 1024]
        #feature size:  128, 64,  32,  16,   8,    4
                                  #z

        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())

        # Build Encoder
        self.encoders_down = nn.ModuleList()
        modules = []
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:3]):
            stride = 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU())
            )
            in_channels = h_dim
        self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[3:]):
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
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        hidden_dims_reverse = hidden_dims[::-1]

        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_reverse[1:4]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        modules = []
        for i, h_dim in enumerate(hidden_dims_reverse[4:]):
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                        nn.BatchNorm2d(h_dim),
                                        nn.SiLU(),
                                        ResBlock(h_dim,
                                                dropout=0,
                                                out_channels=h_dim,
                                                use_conv=True,
                                                dims=2,
                                                use_checkpoint=False),
                                        nn.BatchNorm2d(h_dim),
                                        nn.SiLU()))
            in_channels = h_dim

        modules.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.SiLU(),
                                    nn.Conv2d(in_channels,
                                               out_channels=self.plane_shape[1],
                                                kernel_size= 3, 
                                                padding= 1),
                                    nn.Tanh()))
        self.decoder = nn.Sequential(*modules)


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

        features_down = []
        for module in self.encoders_down:
            feature = module(feature)
            features_down.append(feature)


        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-3]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        
        result = self.decoder(z)
        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w].unsqueeze(1),
                                result[..., plane_w : plane_w * 2].unsqueeze(1),
                                result[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w],
                                result[..., plane_w : plane_w * 2],
                                result[..., plane_w * 2 : plane_w * 3],], dim=1)
        return result

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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

class BetaVAERolloutTransformer_v2_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v2_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v2")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 2)


        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [256, 256, 512, 1024, 1024]
        #feature size:  64,  32,  16,   8,    4
                             #z

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
        modules = []
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims[:2]):
            stride = 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU())
            )
            in_channels = h_dim
        self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[2:]):
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
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        hidden_dims_reverse = hidden_dims[::-1]

        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_reverse[1:4]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        modules = []
        for i, h_dim in enumerate(hidden_dims_reverse[4:]):
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                        nn.BatchNorm2d(h_dim),
                                        nn.SiLU(),
                                        ResBlock(h_dim,
                                                dropout=0,
                                                out_channels=h_dim,
                                                use_conv=True,
                                                dims=2,
                                                use_checkpoint=False),
                                        nn.BatchNorm2d(h_dim),
                                        nn.SiLU()))
            in_channels = h_dim

        modules.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.SiLU(),
                                    nn.Conv2d(in_channels,
                                               out_channels=self.plane_shape[1],
                                                kernel_size= 3, 
                                                padding= 1),
                                    nn.Tanh()))
        self.decoder = nn.Sequential(*modules)


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

        features_down = []
        for module in self.encoders_down:
            feature = module(feature)
            features_down.append(feature)


        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-3]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        
        result = self.decoder(z)
        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w].unsqueeze(1),
                                result[..., plane_w : plane_w * 2].unsqueeze(1),
                                result[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w],
                                result[..., plane_w : plane_w * 2],
                                result[..., plane_w * 2 : plane_w * 3],], dim=1)
        return result

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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

class BetaVAERolloutTransformer_v4_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v4_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)


        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32

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
        for i, h_dim in enumerate(hidden_dims[:2]):
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
        
        for i, h_dim in enumerate(hidden_dims[2:5]):
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
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim


        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
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
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 3:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """

        # breakpoint()
        
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

        # print(feature.shape)
        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16
        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            # print(feature.shape)
            if i in [2, 3]:
                features_down.append(feature)


        feature = self.encoders_up[0](feature)
        # print(feature.shape)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        # print(feature.shape)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)
        # print(feature.shape)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)

        # print(x.shape)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)

            # print(x.shape)

            if i in [0, 1]:
                feature_down.append(x)

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
                # print(x.shape)
            else:
                x = module(x)
                # print(x.shape)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)

        # print(x.shape)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

class BetaVAERolloutTransformer_v3(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v3, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v3")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 2)


        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims_encoder = [128, 256, 256, 512, 1024, 1024, 1024, 512, 256, 32]
               #feature size:  128, 64,  32,  16,   8,    4,     8,   16,  32, 64
                                                                                #z
        hidden_dims_decoder = [256, 128]
                #feature size   64, 128

        self.in_layer = nn.Sequential(ResBlock(
                            32,
                            dropout=0,
                            out_channels=128,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False
                        ),
                        nn.BatchNorm2d(128),
                        nn.SiLU())

        # Build Encoder
        self.encoders_down = nn.ModuleList()
        modules = []
        in_channels = 128
        for i, h_dim in enumerate(hidden_dims_encoder[:2]):
            stride = 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU())
            )
            in_channels = h_dim
        self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims_encoder[2:6]):
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
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim


        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_encoder[6:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 3:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        modules = []
        in_channels = z_shape[0]
        for i, h_dim in enumerate(hidden_dims_decoder):
            if i == 0:
                modules.append(nn.Sequential(
                                            ResBlock(in_channels,
                                                    dropout=0,
                                                    out_channels=h_dim,
                                                    use_conv=True,
                                                    dims=2,
                                                    use_checkpoint=False),
                                            nn.BatchNorm2d(h_dim),
                                            nn.SiLU()))
            else:
                modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                            h_dim,
                                                            kernel_size=3,
                                                            stride = 2,
                                                            padding=1,
                                                            output_padding=1),
                                            nn.BatchNorm2d(h_dim),
                                            nn.SiLU(),
                                            ResBlock(h_dim,
                                                    dropout=0,
                                                    out_channels=h_dim,
                                                    use_conv=True,
                                                    dims=2,
                                                    use_checkpoint=False),
                                            nn.BatchNorm2d(h_dim),
                                            nn.SiLU()))
            in_channels = h_dim

        modules.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                        nn.BatchNorm2d(in_channels),
                                        nn.SiLU(),
                                    nn.Conv2d(in_channels,
                                               out_channels=self.plane_shape[1],
                                                kernel_size= 3, 
                                                padding= 1),
                                    nn.Tanh()))
        self.decoder = nn.Sequential(*modules)


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

        features_down = []
        for module in self.encoders_down:
            feature = module(feature)
            features_down.append(feature)

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-3]], dim=1)
        feature = self.encoders_up[2](feature)
        feature = torch.cat([feature, features_down[-4]], dim=1)
        feature = self.encoders_up[3](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''
        result = self.decoder(z)
        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w].unsqueeze(1),
                                result[..., plane_w : plane_w * 2].unsqueeze(1),
                                result[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            result = torch.concat([result[..., 0 : plane_w],
                                result[..., plane_w : plane_w * 2],
                                result[..., plane_w * 2 : plane_w * 3],], dim=1)
        return result

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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

class BetaVAERolloutTransformer_v5_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v5_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)


        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32

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
        for i, h_dim in enumerate(hidden_dims[:2]):
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
        
        for i, h_dim in enumerate(hidden_dims[2:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer2(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim


        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer2(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer2(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 3:
                modules.append(nn.Sequential(SpatialTransformer2(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16
        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [2, 3]:
                features_down.append(feature)


        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            if i in [0, 1]:
                feature_down.append(x)

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

class BetaVAERolloutTransformer_v6_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v6_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)


        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32

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
                                    SpatialTransformer3(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim


        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer3(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(),
                                    SpatialTransformer3(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer3(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="softmax-xformers",
                                                use_checkpoint=True,
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16
        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [2, 3]:
                features_down.append(feature)


        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            if i in [0, 1]:
                feature_down.append(x)

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

# this model are nearly all transformer layers
class BetaVAERolloutTransformer_v7_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v7_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v7_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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
        
        #
        # self.spatial_modulation = nn.Linear(128*3, 128*3)

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

        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size[i+5]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
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
                                                        layer=feature_size_decoder[i]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        # spatial modulation
        # breakpoint()
        # feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        # feature = rearrange(feature, 'b c h p w -> b h w (p c)').contiguous()
        # feature = self.spatial_modulation(feature)
        # feature = rearrange(feature, 'b h w (p c) -> b h w p c', p=3).contiguous()
        # feature = rearrange(feature, 'b h w p c -> b c h (p w)', p=3).contiguous()

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16
        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [2, 3]:
                features_down.append(feature)

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        # breakpoint()
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            if i in [0, 1]:
                feature_down.append(x)

        # breakpoint()
        
        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

# add feature pyramid fusion module, to save more detailed low-level information
class BetaVAERolloutTransformer_v8_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v8_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v8_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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
        
        #
        # self.spatial_modulation = nn.Linear(128*3, 128*3)

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

        self.encoder_fpn = FPN_down([512, 512, 1024, 1024], [512, 1024, 1024])

        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size[i+5]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
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
                                                        layer=feature_size_decoder[i]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoder_fpn = FPN_up([1024, 1024, 1024, 512], [1024, 1024, 512])

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        # spatial modulation
        # breakpoint()
        # feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        # feature = rearrange(feature, 'b c h p w -> b h w (p c)').contiguous()
        # feature = self.spatial_modulation(feature)
        # feature = rearrange(feature, 'b h w (p c) -> b h w p c', p=3).contiguous()
        # feature = rearrange(feature, 'b h w p c -> b c h (p w)', p=3).contiguous()

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16

        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [0, 1, 2, 3]:
                features_down.append(feature)

        features_down = self.encoder_fpn(features_down)

        # breakpoint()

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            feature_down.append(x)

        # breakpoint()

        feature_down = self.decoder_fpn(feature_down[::-1])

        # breakpoint()

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

# output one dimensional vector as output, parameters too big: 1 billion
class BetaVAERolloutTransformer_v9_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v9_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v9_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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
        
        self.fc_mu = nn.Linear(1024 * 4 * 4 * 3, self.z_shape[0])
        self.fc_var = nn.Linear(1024 * 4 * 4 * 3, self.z_shape[0])

        self.decoder_in_layer = nn.Linear(self.z_shape[0], 1024 * 4 * 4 * 3)

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()

        # breakpoint()

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        # breakpoint()

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        # breakpoint()
        x = self.decoder_in_layer(z)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1024, h=4, w=12).contiguous()

        # x = self.decoder_in_layer(x)
        # breakpoint()

        for i, module in enumerate(self.decoders_up):
            x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

# output one dimensional vector as output
class BetaVAERolloutTransformer_v10_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v10_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v10_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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
        
        # 1024 4 4 3
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )
        
        # 1024 2 2
        
        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        self.decoder_in_linear = nn.Linear(self.z_shape[0], 1024*4)

        self.decoder_in_layer = nn.Sequential(
                                    nn.ConvTranspose2d(1024, out_channels=1024*3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(1024*3),
                                    nn.SiLU()
                                    )
        

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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
        
        features_down = []
        feature = self.in_layer(result)
        features_down.append(feature)

        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            features_down.append(feature)

        # breakpoint()

        feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        feature = rearrange(feature, 'b c h p w -> b (p c) h w').contiguous()
        feature = self.encoder_out_layer(feature)
        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()

        # breakpoint()

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        # breakpoint()
        x = self.decoder_in_linear(z)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1024, h=2, w=2).contiguous()
        # breakpoint()

        x = self.decoder_in_layer(x) # b 4096 h w
        # breakpoint()
        x = rearrange(x, 'b (p c) h w -> b p c h w', p=3).contiguous()
        x = rearrange(x, 'b p c h w -> b c h (p w)').contiguous()

        for i, module in enumerate(self.decoders_up):
            x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

# swap from w structure to v structure , output feature maps
class BetaVAERolloutTransformer_v11_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v11_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v11_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape

        self.depth = [2,8]
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]

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
        
        dim = [512, 1024, 512]
        self.blocks = nn.ModuleList()
        modules = []
        
        for i in range(len(self.depth)):
            h_dim = dim[i]
            out_dim = dim[i+1]
            depth = self.depth[i]
            modules = []
            for j in range(depth):
                dim_head = h_dim // num_heads
                modules.append(nn.Sequential(TransformerBlock(h_dim, 
                                                num_heads,
                                                dim_head,
                                                context_dim=h_dim,
                                                gated_ff=False,
                                                checkpoint=True, 
                                                attn_mode="linear"
                                                    ).cuda()))
            
            if i == 0:
                stride = 2
            elif i== 1:
                stride = 1

            modules.append(nn.Sequential(
                                nn.Conv2d(h_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=1),
                                nn.BatchNorm2d(out_dim),
                                nn.SiLU()))
            
            self.blocks.append(nn.Sequential(*modules))

        # build decoder

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoder_depth = self.depth[::-1]
        dim = [512, 1024, 512]
        self.decoder_blocks = nn.ModuleList()
        modules = []
        
        for i in range(len(self.decoder_depth)):
            h_dim = dim[i]
            out_dim = dim[i+1]
            depth = self.decoder_depth[i]
            modules = []
            for j in range(depth):
                dim_head = h_dim // num_heads
                modules.append(nn.Sequential(TransformerBlock(h_dim, 
                                                num_heads,
                                                dim_head,
                                                context_dim=h_dim,
                                                gated_ff=False,
                                                checkpoint=True, 
                                                attn_mode="linear"
                                                    ).cuda()))
            
            # if i == 0:
            #     stride = 2
            # elif i== 1:
            #     stride = 1

            stride = 2
            modules.append(nn.Sequential(
                                nn.ConvTranspose2d(h_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=1, output_padding=1),
                                nn.BatchNorm2d(out_dim),
                                nn.SiLU()))
            
            self.decoder_blocks.append(nn.Sequential(*modules))

        self.decoders_up = nn.ModuleList()

        self.decoders_up.append(nn.Sequential(
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))

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

        for blk in self.blocks:
            feature = blk(feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)


        for blk in self.decoder_blocks:
            x = blk(x)

        for i, module in enumerate(self.decoders_up):
            x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

# output one dimensional vector as output, refined transformer layers 
class BetaVAERolloutTransformer_v12_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v12_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v12_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]
        
        self.encoder_depth = [1, 1, 2, 2]

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
                                    TransformerLayer(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=self.encoder_depth[i],
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
        
        # 1024 4 4 3
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )
        # 1024 2 2
        
        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        self.decoder_in_linear = nn.Linear(self.z_shape[0], 1024*4)

        self.decoder_in_layer = nn.Sequential(
                                    nn.ConvTranspose2d(1024, out_channels=1024*3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(1024*3),
                                    nn.SiLU()
                                    )
        

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoder_depth = [2, 2, 1, 1]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(TransformerLayer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=self.decoder_depth[i],
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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
        
        # features_down = []
        feature = self.in_layer(result)
        # features_down.append(feature)

        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            # features_down.append(feature)

        # breakpoint()

        feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        feature = rearrange(feature, 'b c h p w -> b (p c) h w').contiguous()
        feature = self.encoder_out_layer(feature)
        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()

        # breakpoint()

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        # breakpoint()
        x = self.decoder_in_linear(z)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1024, h=2, w=2).contiguous()
        # breakpoint()

        x = self.decoder_in_layer(x) # b 4096 h w
        # breakpoint()
        x = rearrange(x, 'b (p c) h w -> b p c h w', p=3).contiguous()
        x = rearrange(x, 'b p c h w -> b c h (p w)').contiguous()

        for i, module in enumerate(self.decoders_up):
            x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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



# output feature map, reduce rollout, add more transformer 
class Transformer_v13_128_spatial(nn.Module):
    def __init__(self, vae_config) -> None:
        super(Transformer_v13_128_spatial, self).__init__()
        print("vae type: Transformer_v13_128_spatial")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32
        feature_size = [64,  32,  16,   8,    4,    8,   16,       32]

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
        
        # self.spatial_modulation = nn.Linear(128*3, 128*3)

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

        self.encoder_fpn = FPN_down([512, 512, 1024, 1024], [512, 1024, 1024])

        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i == 2:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=2*z_shape[0],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(2*z_shape[0]),
                                nn.SiLU()))
                in_channels = z_shape[0]
            else:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size[i+5]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))


        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
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
                                                        layer=feature_size_decoder[i]
                                                        ),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoder_fpn = FPN_up([1024, 1024, 1024, 512], [1024, 1024, 512])

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        # spatial modulation
        # breakpoint()
        # feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        # feature = rearrange(feature, 'b c h p w -> b h w (p c)').contiguous()
        # feature = self.spatial_modulation(feature)
        # feature = rearrange(feature, 'b h w (p c) -> b h w p c', p=3).contiguous()
        # feature = rearrange(feature, 'b h w p c -> b c h (p w)', p=3).contiguous()

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16

        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [0, 1, 2, 3]:
                features_down.append(feature)

        features_down = self.encoder_fpn(features_down)

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)
            feature_down.append(x)

        # breakpoint()

        feature_down = self.decoder_fpn(feature_down[::-1])

        # breakpoint()

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
            else:
                x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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











class TransformerVAE_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(TransformerVAE_128, self).__init__()
        print("vae type: TransformerVAE_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        # hidden_dims = [256, 512, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,       32

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        # 1131111
        # 1123211
        # 1122211
        self.down_stages = 4
        self.up_stages = 3
        depths = [1,1,2,2]
        up_depths = [2,1,1]
        attention_types = ["softmax-xformers", "softmax-xformers", "softmax-xformers", "softmax-xformers"]

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
        
        encoder_down_dims =  [256, 512, 1024, 1024, 1024]
        # Build Encoder_down    
        self.encoders_down = nn.ModuleList()
        in_channels = 128
        for i, h_dim in enumerate(encoder_down_dims[:1]):
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
        
        for i in range(self.down_stages):
            h_dim = encoder_down_dims[i]
            dim_head = h_dim // num_heads
            attention_type = attention_types[i]
            block = nn.ModuleList()
            for j in range(depths[i]): 
                block.append(TransformerBlock(h_dim,
                                                num_heads,
                                                dim_head,
                                                context_dim=h_dim,
                                                gated_ff=False,
                                                checkpoint=False, 
                                                attn_mode=attention_type
                                                ).cuda())
            
            patch_embed = PatchEmbed(in_chans=h_dim, out_chans= encoder_down_dims[i+1])
            # group_modulation = GroupModulation(in_chans=h_dim, out_chans= encoder_down_dims[i+1])

            # setattr(self, f"pos_drop{i + 1}", pos_drop) # later
            setattr(self, f"encoder_down_block{i + 1}", block) 
            setattr(self, f"encoder_down_patch_embed{i + 1}", patch_embed)
            # setattr(self, f"encoder_down_group_modulation{i + 1}", group_modulation)

            in_channels = h_dim

        # encoder_fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 1024, 1024], 256)

        encoder_up_dims = [1024, 1024,1024, 2*self.z_shape[0]]
        # build encoder_up
        # self.encoders_up = nn.ModuleList()
        for i in range(self.up_stages):
            h_dim = (encoder_up_dims[i])
            dim_head = h_dim // num_heads
            attention_type = attention_types[i]
            block = nn.ModuleList()
            for j in range(up_depths[i]): 
                block.append(TransformerBlock(h_dim,
                                                num_heads,
                                                dim_head,
                                                context_dim=h_dim,
                                                gated_ff=False,
                                                checkpoint=False, 
                                                attn_mode=attention_type
                                                ).cuda())
            
            patch_embed = PatchEmbed(in_chans=h_dim * 2, out_chans = encoder_up_dims[i+1], down=False)

            setattr(self, f"encoder_up_block{i + 1}", block) 
            setattr(self, f"encoder_up_patch_embed{i + 1}", patch_embed)

        ## build decoder
        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0],
                            dropout=0,
                            out_channels=512,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=1
                        ),
                        nn.BatchNorm2d(512),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        
        decoder_down_dims = [512, 1024, 1024, 1024]
       
        # decoder down
        for i in range(self.up_stages):
            h_dim = (decoder_down_dims[i])
            dim_head = h_dim // num_heads
            attention_type = attention_types[i]
            block = nn.ModuleList()
            for j in range(up_depths[i]): 
                block.append(TransformerBlock(h_dim,
                                                num_heads,
                                                dim_head,
                                                context_dim=h_dim,
                                                gated_ff=False,
                                                checkpoint=False, 
                                                attn_mode=attention_type
                                                ).cuda())
                
            patch_embed = PatchEmbed(in_chans=h_dim, out_chans= decoder_down_dims[i+1], down=True)

            setattr(self, f"decoder_down_block{i + 1}", block) 
            setattr(self, f"decoder_down_patch_embed{i + 1}", patch_embed)

        decoder_up_dims = [1024, 1024, 1024, 512, 256]
        # decoder up
        for i in range(self.down_stages):
            h_dim = (decoder_up_dims[i])

            block = nn.ModuleList()
            for j in range(depths[i]): 
                block.append(TransformerBlock(h_dim,
                                                num_heads,
                                                dim_head,
                                                context_dim=h_dim,
                                                gated_ff=False,
                                                checkpoint=False, 
                                                attn_mode=attention_type
                                                ).cuda())
            
            if i < 3:
                patch_embed = PatchEmbed(in_chans=h_dim * 2, out_chans= decoder_up_dims[i+1], down=False)
            else:
                patch_embed = PatchEmbed(in_chans=h_dim, out_chans= decoder_up_dims[i+1], down=False)


            setattr(self, f"decoder_up_block{i + 1}", block) 
            setattr(self, f"decoder_up_patch_embed{i + 1}", patch_embed)

        in_channels = decoder_up_dims[-1]
        self.decoders_up = nn.ModuleList()
        self.decoders_up.append(nn.Sequential(
                                    nn.ConvTranspose2d(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.SiLU(),
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """
        # enc_input b 3 d h w  

        # breakpoint()

        x = enc_input
        if self.plane_dim == 5:
            plane_list = []
            for i in range(self.plane_shape[0]):
                plane_list.append(x[:, i, :, :, :])
            x = torch.concat(plane_list, dim=-1)
        elif self.plane_dim == 4:
            plane_channel = x.shape[1] // 3
            x = torch.concat([x[:, 0:plane_channel ,...],
                                x[:, plane_channel:plane_channel*2 ,...],
                                x[:, plane_channel*2:plane_channel*3 ,...]], dim=-1)
        
        # result : b 
        # x = rearrange(x, 'b p d h w -> b d h (p w)') 
        x = self.in_layer(x)

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16
        features_down = []
        for i, module in enumerate(self.encoders_down):
            x = module(x)
            # if i in [2, 3]:
            #     features_down.append(feature)
        
        for i in range(self.down_stages):
            patch_embed = getattr(self, f"encoder_down_patch_embed{i + 1}")
            # pos_embed = getattr(self, f"encoder_down_pos_embed{i + 1}")
            block = getattr(self, f"encoder_down_block{i + 1}")
            # group_modulation = getattr(self, f"encoder_down_group_modulation{i + 1}")

            b, c, h, w = x.shape # 8, 512, 16, 48
            x = rearrange(x, "b c h w -> b (h w) c").contiguous() # b n d
            for blk in block:
                x = blk(x)   # x: b (3hw) d

            # patch embed
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous() 
            x = patch_embed(x)
            
            # group modulation
            # x = rearrange(x, "b c h (p w) -> b c h p w", p=3).contiguous()
            # x = rearrange(enc_input, 'b c h p w -> b (p c) h w').contiguous()
            # x = group_modulation(x)
            # x = rearrange(enc_input, 'b c h p w -> b (p c) h w').contiguous()

            features_down.append(x)
        
        # breakpoint()

        # breakpoint()
        for i in range(self.up_stages):
            patch_embed = getattr(self, f"encoder_up_patch_embed{i + 1}")
            # pos_embed = getattr(self, f"encoder_down_pos_embed{i + 1}")
            block = getattr(self, f"encoder_up_block{i + 1}")

            b, c, h, w = x.shape # 8, 512, 16, 48
            x = rearrange(x, "b c h w -> b (h w) c").contiguous() # b n d
            for blk in block:
                x = blk(x)   # x: b (3hw) d

            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous() 
            x = torch.cat([x, features_down[-(i+1)]], dim=1)
            x = patch_embed(x)

        # breakpoint()

        encode_channel = self.z_shape[0]
        mu = x[:, :encode_channel, ...]
        log_var = x[:, encode_channel:, ...]

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        # breakpoint()
        x = self.decoder_in_layer(z)
        features_down = [x]

        for i in range(self.up_stages):
            patch_embed = getattr(self, f"decoder_down_patch_embed{i + 1}")
            block = getattr(self, f"decoder_down_block{i + 1}")

            b, c, h, w = x.shape # 8, 512, 16, 48
            x = rearrange(x, "b c h w -> b (h w) c").contiguous() # b n d
            for blk in block:
                x = blk(x)   # x: b (3hw) d

            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous() 
            x = patch_embed(x)
            features_down.append(x)
        
        # breakpoint()

        for i in range(self.down_stages):
            patch_embed = getattr(self, f"decoder_up_patch_embed{i + 1}")
            # pos_embed = getattr(self, f"decoder_down_pos_embed{i + 1}")
            block = getattr(self, f"decoder_up_block{i + 1}")

            b, c, h, w = x.shape # 8, 512, 16, 48
            x = rearrange(x, "b c h w -> b (h w) c").contiguous() # b n d
            for blk in block:
                x = blk(x)   # x: b (3hw) d

            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous() 

            if i < 3:
                x = torch.cat([x, features_down[-(i+1)]], dim=1)
            x = patch_embed(x)

        for i, module in enumerate(self.decoders_up):
            x = module(x)

        # breakpoint()

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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





















# output one dimensional vector as output
class BetaVAERolloutTransformer_v10_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v10_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v10_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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

        self.decoder_in_linear = nn.Linear(self.z_shape[0], 1024*4)

        self.decoder_in_layer = nn.Sequential(
                                    nn.ConvTranspose2d(1024, out_channels=1024*3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(1024*3),
                                    nn.SiLU()
                                    )
        

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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
                                    ResBlock(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        # breakpoint()

        feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        feature = rearrange(feature, 'b c h p w -> b (p c) h w').contiguous()
        feature = self.encoder_out_layer(feature)
        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()

        # breakpoint()

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        # breakpoint()
        x = self.decoder_in_linear(z)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1024, h=2, w=2).contiguous()
        # breakpoint()

        x = self.decoder_in_layer(x) # b 4096 h w
        # breakpoint()
        x = rearrange(x, 'b (p c) h w -> b p c h w', p=3).contiguous()
        x = rearrange(x, 'b p c h w -> b c h (p w)').contiguous()


        for i, module in enumerate(self.decoders_up):
            x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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


class BetaVAETransformer_v1_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAETransformer_v1_128, self).__init__()
        print("vae type: BetaVAETransformer_v1_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4096])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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

        self.decoder_in_linear = nn.Linear(self.z_shape[0], 1024*4)

        self.decoder_in_layer = nn.Sequential(
                                    nn.ConvTranspose2d(1024, out_channels=1024*3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(1024*3),
                                    nn.SiLU()
                                    )
        

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
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

        # breakpoint()

        feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        feature = rearrange(feature, 'b c h p w -> b (p c) h w').contiguous()
        feature = self.encoder_out_layer(feature)
        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()

        # breakpoint()

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        x = self.decoder_in_linear(z)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1024, h=2, w=2).contiguous()

        x = self.decoder_in_layer(x) # b 4096 h w
        x = rearrange(x, 'b (p c) h w -> b p c h w', p=3).contiguous()
        x = rearrange(x, 'b p c h w -> b c h (p w)').contiguous()


        for i, module in enumerate(self.decoders_up):
            x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
        
        x = rearrange(x, 'b d c h w -> b (d c) h w')
        x = self.decoders_up_group(x)
        x = rearrange(x, 'b (d c) h w -> b d c h w', d=3)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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



class BetaVAETransformer_v2_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAETransformer_v2_128, self).__init__()
        print("vae type: BetaVAETransformer_v2_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4096])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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
        
        self.in_layer = nn.Sequential(ResBlock_g(
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
                    GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU(),
                    ResBlock_g(
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
                                    GroupConv(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
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
        
        # self.encoder_out_layer = nn.Sequential(
        #                             nn.Conv2d(1024*3, out_channels=1024*3, kernel_size=3, stride=2, padding=1, groups=3),
        #                             nn.BatchNorm2d(1024*3),
        #                             nn.SiLU()
        #                             )
        self.encoder_out_layer = nn.Sequential(
                                    nn.Conv2d(1024*3, out_channels=1024, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.SiLU()
                                    )

        self.fc_mu = nn.Linear(4096, self.z_shape[0])
        self.fc_var = nn.Linear(4096, self.z_shape[0])

        self.decoder_in_linear = nn.Linear(self.z_shape[0], 1024*4)

        # self.decoder_in_layer = nn.Sequential(
        #                             GroupConvTranspose(1024, out_channels=1024*3, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                             nn.BatchNorm2d(1024*3),
        #                             nn.SiLU()
        #                             )
        
        self.decoder_in_layer = nn.Sequential(
                                    nn.ConvTranspose2d(1024, out_channels=1024*3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(1024*3),
                                    nn.SiLU()
                                    )
        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(GroupConvTranspose(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock_g(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.decoders_up.append(nn.Sequential(*modules))

        self.decoders_up.append(nn.Sequential(
                                    GroupConvTranspose(in_channels,
                                                        in_channels,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.SiLU(),
                                    ResBlock_g(
                                        in_channels,
                                        dropout=0,
                                        out_channels=self.plane_shape[1],
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                    nn.BatchNorm2d(self.plane_shape[1]),
                                    nn.Tanh()))


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

        # breakpoint()

        feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        # print("1", feature.shape)
        feature = rearrange(feature, 'b c h p w -> b (p c) h w').contiguous()
        # print("2", feature.shape)
        feature = self.encoder_out_layer(feature)
        # print("3", feature.shape)
        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()
        # print("4", feature.shape)

        # breakpoint()

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        # breakpoint()
        x = self.decoder_in_linear(z)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1024, h=2, w=2).contiguous()
        # breakpoint()

        x = self.decoder_in_layer(x) # b 4096 h w
        # breakpoint()
        x = rearrange(x, 'b (p c) h w -> b p c h w', p=3).contiguous()
        x = rearrange(x, 'b p c h w -> b c h (p w)').contiguous()


        for i, module in enumerate(self.decoders_up):
            x = module(x)

        if self.plane_dim == 5:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w].unsqueeze(1),
                                x[..., plane_w : plane_w * 2].unsqueeze(1),
                                x[..., plane_w * 2 : plane_w * 3].unsqueeze(1),], dim=1)
        elif self.plane_dim == 4:
            plane_w = self.plane_shape[-1]
            x = torch.concat([x[..., 0 : plane_w],
                                x[..., plane_w : plane_w * 2],
                                x[..., plane_w * 2 : plane_w * 3],], dim=1)
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

    




#  BetaVAETransformer_v3_128: based on v1, with group conv transpose at last layer of the decoder
class BetaVAETransformer_v3_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAETransformer_v3_128, self).__init__()
        print("vae type: BetaVAETransformer_v3_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4096])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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

        self.decoder_in_linear = nn.Linear(self.z_shape[0], 1024*4)

        self.decoder_in_layer = nn.Sequential(
                                    nn.ConvTranspose2d(1024, out_channels=1024*3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(1024*3),
                                    nn.SiLU()
                                    )
        

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            # if i > 0 and i < 4:
            #     in_channels = in_channels * 2
            dim_head = h_dim // num_heads
            modules.append(nn.Sequential(nn.ConvTranspose2d(in_channels,
                                                        h_dim,
                                                        kernel_size=3,
                                                        stride = 2,
                                                        padding=1,
                                                        output_padding=1),
                            nn.BatchNorm2d(h_dim),
                            nn.SiLU()))
            if i < 4:
                modules.append(nn.Sequential(SpatialTransformer(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                layer = feature_size_decoder[i+3]
                                                ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(ResBlock(
                                        h_dim,
                                        dropout=0,
                                        out_channels=h_dim,
                                        use_conv=True,
                                        dims=2,
                                        use_checkpoint=False,
                                    ),
                                nn.BatchNorm2d(h_dim),
                                nn.SiLU()))
                in_channels = h_dim
            self.decoders_up.append(nn.Sequential(*modules))

        self.to_planes = nn.Sequential(
                            nn.ConvTranspose2d(in_channels * 3,
                                                in_channels * 3,
                                                kernel_size=3,
                                                stride = 2,
                                                padding=1,
                                                output_padding=1,
                                                groups=3),
                            nn.BatchNorm2d(in_channels * 3),
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

        # breakpoint()

        feature = rearrange(feature, 'b c h (p w) -> b c h p w', p=3).contiguous()
        feature = rearrange(feature, 'b c h p w -> b (p c) h w').contiguous()
        feature = self.encoder_out_layer(feature)
        feature = rearrange(feature, 'b c h w -> b (c h w)').contiguous()

        # breakpoint()

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64

        x = self.decoder_in_linear(z)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1024, h=2, w=2).contiguous()

        x = self.decoder_in_layer(x) # b 4096 h w
        x = rearrange(x, 'b (p c) h w -> b p c h w', p=3).contiguous()
        x = rearrange(x, 'b p c h w -> b c h (p w)').contiguous()


        for i, module in enumerate(self.decoders_up):
            x = module(x)

        x = torch.concat(torch.chunk(x,3,dim=-1),dim=1)
        x = self.to_planes(x)
        x = torch.concat(torch.chunk(x.unsqueeze(1),3,dim=2), dim=1)
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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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

        
        




class Transformer_v13_128_spatial(nn.Module):
    def __init__(self, vae_config) -> None:
        super(Transformer_v13_128_spatial, self).__init__()
        print("vae type: Transformer_v13_128_spatial")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

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
                            32*3,
                            dropout=0,
                            out_channels=128*3,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=3,
                            group_layer_num_out=3,
                            conv_group=3,
                        ),
                        nn.BatchNorm2d(128*3),
                        nn.SiLU())
        
        # Build Encoder
        self.encoders_down = nn.ModuleList()
        in_channels = 128

        # for i, h_dim in enumerate(hidden_dims[:1]):
        #     stride = 2
        #     modules = []
        #     modules.append(
        #         nn.Sequential(
        #             nn.AvgPool2d(2, 1),
        #             nn.BatchNorm2d(h_dim),
        #             nn.SiLU(),
        #             ResBlock(
        #                 h_dim,
        #                 dropout=0,
        #                 out_channels=h_dim,
        #                 use_conv=True,
        #                 dims=2,
        #                 use_checkpoint=False,
        #             ),
        #             nn.BatchNorm2d(h_dim),
        #             nn.SiLU()),
        #     )
            # in_channels = h_dim
            # self.encoders_down.append(nn.Sequential(*modules))
        
        for i, h_dim in enumerate(hidden_dims[0:5]):
            dim_head = h_dim // num_heads
            self.encoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels*3, out_channels=h_dim*3, kernel_size=1, groups=3),                      
                                    nn.GroupNorm(3, h_dim*3),
                                    nn.SiLU(),
                                    nn.AvgPool2d(kernel_size=2),   
                                    SpatialTransformer4(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        seq_len=feature_size[i],
                                                        sape = True
                                                        ),
                                    # nn.BatchNorm2d(h_dim),
                                    nn.GroupNorm(3, h_dim*3),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.encoder_fpn = FPN_down([512*3, 512*3, 1024*3, 1024*3], [512*3, 1024*3, 1024*3], conv_group=3)

        self.encoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[5:]):
            modules = []
            if i > 0:
                in_channels = in_channels * 2
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
            # if i == 2:
                # modules.append(nn.Sequential(ResBlock(
                #                         h_dim,
                #                         dropout=0,
                #                         out_channels=2*z_shape[0],
                #                         use_conv=True,
                #                         dims=2,
                #                         use_checkpoint=False,
                #                     ),
                #                 nn.BatchNorm2d(2*z_shape[0]),
                #                 nn.SiLU()))
                # in_channels = z_shape[0]
            # else:
            modules.append(nn.Sequential(SpatialTransformer4(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                seq_len = feature_size[i+5],
                                                sape=True
                                                ),
                                # nn.BatchNorm2d(h_dim),
                                nn.GroupNorm(3, h_dim*3),
                                nn.SiLU()))
            
            in_channels = h_dim
            self.encoders_up.append(nn.Sequential(*modules))

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
                #feature size:  16,    8,   4,   8,    16,  32,  64

        feature_size_decoder = [16, 8, 4, 8, 16, 32, 64]

        self.decoder_in_layer = nn.Sequential(ResBlock(
                            self.z_shape[0]*3,
                            dropout=0,
                            out_channels=512*3,
                            use_conv=True,
                            dims=2,
                            use_checkpoint=False,
                            group_layer_num_in=3,
                            group_layer_num_out=3,                            
                            conv_group=3
                        ),
                        nn.GroupNorm(3, 512*3),
                        nn.SiLU())
        
        self.decoders_down = nn.ModuleList()
        in_channels = 512
        for i, h_dim in enumerate(hidden_dims_decoder[0:3]):
            dim_head = h_dim // num_heads
            stride = 2
            self.decoders_down.append(nn.Sequential(
                                    nn.Conv2d(in_channels*3, out_channels=h_dim*3, kernel_size=1, groups=3),
                                    nn.GroupNorm(3, h_dim*3),
                                    nn.SiLU(),
                                    nn.AvgPool2d(kernel_size=2), 
                                    SpatialTransformer4(h_dim,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=h_dim,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="linear",
                                                        use_checkpoint=True,
                                                        seq_len=feature_size_decoder[i],
                                                        sape=True
                                                        ),
                                    nn.GroupNorm(3, h_dim*3),
                                    nn.SiLU()
                                    ))
            in_channels = h_dim

        self.decoder_fpn = FPN_up([1024*3, 1024*3, 1024*3, 512*3], [1024*3, 1024*3, 512*3], conv_group=3)

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[3:]):
            modules = []
            if i > 0 and i < 4:
                in_channels = in_channels * 2
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
            # if i < 4:
            modules.append(nn.Sequential(SpatialTransformer4(h_dim,
                                                num_heads,
                                                dim_head,
                                                depth=transform_depth,
                                                context_dim=h_dim,
                                                disable_self_attn=False,
                                                use_linear=True,
                                                attn_type="linear",
                                                use_checkpoint=True,
                                                seq_len = feature_size_decoder[i+3],
                                                sape=True
                                                ),
                                nn.GroupNorm(3, h_dim*3),
                                nn.SiLU()))
            in_channels = h_dim
            # else:
            #     modules.append(nn.Sequential(ResBlock(
            #                             h_dim,
            #                             dropout=0,
            #                             out_channels=h_dim,
            #                             use_conv=True,
            #                             dims=2,
            #                             use_checkpoint=False,
            #                         ),
            #                     nn.BatchNorm2d(h_dim),
            #                     nn.SiLU()))
                # in_channels = h_dim

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


    def encode(self, enc_input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :enc_input: (Tensor) Input tensor to encoder [B x D x resolution x resolution]
        :return: (Tensor) List of latent codes
        """

        if self.plane_dim == 5:
            x = rearrange(enc_input, 'b p d h w -> b (p d) h w')

        feature = self.in_layer(x)

        # breakpoint()
        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16

        features_down = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            if i in [0, 1, 2, 3]:
                features_down.append(feature)

        features_down = self.encoder_fpn(features_down)

        # breakpoint()

        feature = self.encoders_up[0](feature)
        feature = torch.cat([feature, features_down[-1]], dim=1)
        feature = self.encoders_up[1](feature)
        feature = torch.cat([feature, features_down[-2]], dim=1)
        feature = self.encoders_up[2](feature)

        # breakpoint()
        feature = rearrange(feature, 'b (p d) h w -> b p d h w', p=3)
        feature = rearrange(feature, 'b p d h w -> b d h (p w)')
        encode_channel = self.z_shape[0]
        mu = feature[:, :encode_channel, ...]
        log_var = feature[:, encode_channel:, ...]

        # breakpoint()

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        # breakpoint()
        z = rearrange(z, 'b d h (p w) -> b p d h w', p=3)
        z = rearrange(z, 'b p d h w -> b (p d) h w')

        x = self.decoder_in_layer(z)

        # breakpoint()
        feature_down = [x]
        for i, module in enumerate(self.decoders_down):
            x = module(x)

        # breakpoint()

        feature_down = self.decoder_fpn(feature_down[::-1])

        # breakpoint()

        for i, module in enumerate(self.decoders_up):
            if i in [1, 2, 3]:
                x = torch.cat([x, feature_down[-i]], dim=1)
                x = module(x)
                print("decoder up layer x shape", x.shape, i)
            else:
                x = module(x)
                print("decoder up layer x shape", x.shape, i)

        # breakpoint()
        x = rearrange(x, 'b (p d) h w -> b p d h w', p=3)
        # x = rearrange(x, 'b p d h w -> b p d h w')

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

        z_rollout_shape = [self.z_shape[0], self.z_shape[1], self.z_shape[2] * 3]
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
    vae_model = BetaVAETransformer_v2_128(vae_config).cuda()
    

    input_tensor = torch.randn(4, 3, 32, 128, 128).cuda()
    out = vae_model(input_tensor)
    loss = vae_model.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    samples = vae_model.sample(2)
    print("samples shape: {}".format(samples[0].shape))
    import pdb;pdb.set_trace()