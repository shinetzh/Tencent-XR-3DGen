import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, reduce

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


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



class Conv3DAwareTranspose(nn.Module):
    """
    adapted for 3D aware convolution for triplane in Rodin: https://arxiv.org/pdf/2212.06135.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels * 3,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      stride = stride,
                                      padding=padding,
                                      output_padding=output_padding)
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
        result = self.conv_transpose(triplane)
        return result


class SpacialVAERolloutAware3d(nn.Module):
    def __init__(self,
                 kl_std=0.25,
                 kl_weight=0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [4, 64, 64]) -> None:
        super(SpacialVAERolloutAware3d, self).__init__()

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        self.reso = plane_shape[-1]
        #print("kl standard deviation: ", self.kl_std)
        self.plane_res_encode = self.reso // 8
        self.pre_square = self.plane_res_encode * self.plane_res_encode

        hidden_dims = [128, 512, 128, 512, 128]
        self.hidden_dims = hidden_dims
        if self.plane_dim == 4:
            self.in_channels = self.plane_shape[0] // 3
        elif self.plane_dim == 5:
            self.in_channels = self.plane_shape[1]
        in_channels = self.in_channels
        # Build Encoder
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            if i in [1, 3]:
                stride = 2
            else:
                stride = 1
            modules.append(
                nn.Sequential(
                    Conv3DAware(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(Conv3DAware(in_channels, out_channels=self.z_shape[0] * 2, kernel_size=3, stride=stride, padding=1))
        )

        self.encoder = nn.Sequential(*modules)



        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.pre_square) 
        self.decoder_input = Conv3DAware(self.z_shape[0], hidden_dims[-1], 1, 1, 0)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if i in [1, 3]:
              modules.append(
                  nn.Sequential(
                      Conv3DAwareTranspose(hidden_dims[i],
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
                        Conv3DAware(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            Conv3DAware(hidden_dims[-1], out_channels= self.in_channels, # changed from 3 to in_channels
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



class SpacialVAERollout(nn.Module):
    def __init__(self,
                 kl_std=0.25,
                 kl_weight=0.001,
                 plane_shape = [3, 32, 256, 256],
                 z_shape = [4, 64, 64]) -> None:
        super(SpacialVAERollout, self).__init__()

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        hidden_dims = [512, 512, 512, 512, 512]
        self.hidden_dims = hidden_dims
        if self.plane_dim == 4:
            self.in_channels = self.plane_shape[0] // 3
        elif self.plane_dim == 5:
            self.in_channels = self.plane_shape[1]
        in_channels = self.in_channels
        # Build Encoder
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            if i in [1, 3]:
                stride = 2
            else:
                stride = 1
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
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



# if __name__ == "__main__":
#     # vae_model = SpacialRolloutVAE(plane_shape = [3, 32, 256, 256], z_shape=[4, 64, 192])
#     vae_model = SpacialVAERolloutAware3d(plane_shape = [3, 32, 256, 256], z_shape=[4, 64, 192])
    
#     vae_model = vae_model.cuda()
#     input_tensor = torch.randn(2, 3, 32, 256, 256).float().cuda()
#     out = vae_model(input_tensor)
#     loss = vae_model.loss_function(*out)
#     print("loss: {}".format(loss))
#     print("z shape: {}".format(out[-1].shape))
#     print("reconstruct shape: {}".format(out[0].shape))
#     samples = vae_model.sample(2)
#     print("samples shape: {}".format(samples[0].shape))
#     import pdb;pdb.set_trace()