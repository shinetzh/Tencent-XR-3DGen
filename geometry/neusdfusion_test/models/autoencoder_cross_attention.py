import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, reduce
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
)

from models.archs.attention import SpatialTransformer, checkpoint

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

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
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels if out_channels else channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

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



class CrossAttention3DAware(nn.Module):
    """
    adapted for 3D aware convolution for triplane in Rodin: https://arxiv.org/pdf/2212.06135.pdf
    """
    def __init__(self, ch,
                    num_heads,
                    dim_head,
                    depth=1,
                    context_dim=-1,
                    disable_self_attn=False,
                    use_linear=True,
                    attn_type="softmax",
                    use_checkpoint=True):
        super().__init__()
        self.conv = conv_nd(2, ch, ch, 3, padding=1)
        self.transformer = SpatialTransformer(ch,
                                            num_heads,
                                            dim_head,
                                            depth=depth,
                                            context_dim=context_dim,
                                            disable_self_attn=disable_self_attn,
                                            use_linear=use_linear,
                                            attn_type=attn_type,
                                            use_checkpoint=use_checkpoint)
    

    def perception_3d_sdf(self, x):
        _, _, h, w = x.shape
        fea_yx, fea_zx, fea_yz = x[..., 0:w//3], x[..., w//3:(w//3) * 2], x[..., (w//3) * 2:]

        context_yx = x[..., w//3:]
        context_zx = torch.cat([fea_yx, fea_yz], dim=3)
        context_yz = torch.cat([fea_yx, fea_zx], dim=3)

        context_yx = rearrange(context_yx, "b c h w -> b (h w) c").contiguous()
        context_zx = rearrange(context_zx, "b c h w -> b (h w) c").contiguous()
        context_yz = rearrange(context_yz, "b c h w -> b (h w) c").contiguous()

        h_yx = self.transformer(fea_yx, context_yx)
        h_zx = self.transformer(fea_zx, context_zx)
        h_yz = self.transformer(fea_yz, context_yz)

        h = torch.cat([h_yx, h_zx, h_yz], dim=3)
        hx = self.conv(x) + h
        return hx

        # return self.transformer(x)

    def forward(self, x):
        triplane =  self.perception_3d_sdf(x)
        return triplane


class Conv3DAware(nn.Module):
    """
    adapted for 3D aware convolution for triplane in Rodin: https://arxiv.org/pdf/2212.06135.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size, stride, padding)
        self.out_channels = out_channels
    
    def perception_3d_sdf(self, x):
        _, _, h, w = x.shape
        fea_yx, fea_zx, fea_yz = x[..., 0:w//3], x[..., w//3:(w//3) * 2], x[..., (w//3) * 2:]
        fea_yx_mean_y = torch.mean(fea_yx, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_yx_mean_x = torch.mean(fea_yx, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_zx_mean_z = torch.mean(fea_zx, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_zx_mean_x = torch.mean(fea_zx, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yz_mean_y = torch.mean(fea_yz, dim=3, keepdim=True).repeat(1, 1, 1, w//3)
        fea_yz_mean_z = torch.mean(fea_yz, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yx_3d_aware = torch.cat((fea_yx, fea_zx_mean_x, fea_yz_mean_y), dim=1)
        fea_zx_3d_aware = torch.cat((fea_zx, fea_yx_mean_x, fea_yz_mean_z), dim=1)
        fea_yz_3d_aware = torch.cat((fea_yz, fea_yx_mean_y, fea_zx_mean_z), dim=1)
        fea = torch.cat([fea_yx_3d_aware, fea_zx_3d_aware, fea_yz_3d_aware], dim=3)

        return fea

    def forward(self, x):
        triplane =  self.perception_3d_sdf(x)
        result = self.conv(triplane)
        return result



class SpacialVAECrossAttention(nn.Module):
    def __init__(self,
                 kl_std=1.0,
                 kl_weight=0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [4, 64, 64],
                 model_channels = 64,
                 num_head_channels = 32,
                 channel_mult = [1, 2, 2],
                 transformer_depth = [2, 4, 5],
                 attention_resolutions = [4, 2],
                 num_res_block = 2,
                 use_checkpoint=True) -> None:
        super(SpacialVAECrossAttention, self).__init__()

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight
        
        self.num_res_blocks = [num_res_block] * len(channel_mult)


        ###### encoder
        self.encoder = nn.ModuleList([conv_nd(2, plane_shape[1], model_channels, 3, padding=1)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level] + 1):
                layers = [ResBlock(
                            ch,
                            dropout=0,
                            out_channels=mult * model_channels,
                            use_conv=False,
                            dims=2,
                            use_checkpoint=use_checkpoint,
                        )]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if ds == 4:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                        layers.append(
                            CrossAttention3DAware(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=ch,
                                disable_self_attn=False,
                                use_linear=True,
                                attn_type="softmax",
                                use_checkpoint=use_checkpoint,
                            )
                        )
                    else:
                        layers.append(Conv3DAware(in_channels=ch, 
                                                  out_channels=ch, 
                                                  kernel_size=3, 
                                                  stride=1, 
                                                  padding=1)
                                        )
                self.encoder.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.encoder.append(Downsample(ch, True, dims=2, out_channels=out_ch))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.encoder.append(conv_nd(2, ch, z_shape[0] * 2, 1, padding=0))
        ch = z_shape[0]

        ###### decoder
        self.decoder = nn.ModuleList([conv_nd(2, ch, model_channels, 1, padding=0)])
        ch = model_channels
        transformer_depth = transformer_depth[::-1]
        for level, mult in list(enumerate(channel_mult)):
            for i in range(self.num_res_blocks[level] + 1):
                layers = [ResBlock(
                            ch,
                            dropout=0,
                            out_channels=mult * model_channels,
                            use_conv=False,
                            dims=2,
                            use_checkpoint=use_checkpoint)
                        ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if ds == 4:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                        layers.append(
                            CrossAttention3DAware(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=ch,
                                disable_self_attn=False,
                                use_linear=True,
                                attn_type="softmax",
                                use_checkpoint=use_checkpoint,
                            )
                        )
                    else:
                        layers.append(Conv3DAware(in_channels=ch, 
                                                  out_channels=ch, 
                                                  kernel_size=3, 
                                                  stride=1, 
                                                  padding=1)
                                        )
                self.decoder.append(nn.Sequential(*layers))
            if level and i == self.num_res_blocks[level]:
                out_ch = ch
                self.decoder.append(Upsample(ch, True, dims=2, out_channels=out_ch))
                ds //= 2

        self.decoder.append(nn.Sequential(
                                normalization(ch),
                                nn.SiLU(),
                                conv_nd(2, ch, plane_shape[1], 1, padding=0),
                                nn.Tanh()
                                ))

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
        for module in self.encoder:
            result = module(result)
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
        result = z
        for module in self.decoder:
            result = module(result)

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

if __name__ == "__main__":
    # vae_model = SpacialRolloutVAE(plane_shape = [3, 32, 256, 256], z_shape=[4, 64, 192])
    vae_model = SpacialVAECrossAttention(plane_shape = [3, 32, 256, 256], z_shape=[6, 64, 192], use_checkpoint=True)
    
    vae_model = vae_model.cuda()
    input_tensor = torch.randn(2, 3, 32, 256, 256).float().cuda()
    out = vae_model(input_tensor)
    loss = vae_model.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    samples = vae_model.sample(2)
    print("samples shape: {}".format(samples[0].shape))
    loss.backward()
    import pdb;pdb.set_trace()