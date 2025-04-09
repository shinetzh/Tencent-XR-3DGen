import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar("torch.tensor")

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        if len(latents.shape) == 4:
            latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        if latents.shape == 4:
            return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
        else:
            return quantized_latents.contiguous(), vq_loss

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(nn.Module):

    def __init__(self, config) -> None:
        super(VQVAE, self).__init__()

        in_channels = config.get("in_channels", 3)
        embedding_dim = config.get("embedding_dim", 64)
        num_embeddings = config.get("num_embeddings", 512)
        hidden_dims = config.get("hidden_dims", None)
        beta = config.get("beta", 0.25)
        img_size = config.get("img_size", 64)


        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        result = self.decode(quantized_inputs)
        return [result, input, vq_loss, encoding]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]




class BetaVAE_VQ(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self, config) -> None:
        super(BetaVAE_VQ, self).__init__()

        plane_shape = config.get("plane_shape", [3, 32, 256, 256])
        num_embeddings = config.get("num_embeddings", 4096)
        beta = config.get("beta", 0.25)
        z_shape = config.get("z_shape", [1024])

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        if len(plane_shape) == 4:
            self.in_channels = plane_shape[0] * plane_shape[1]
        else:
            self.in_channels = plane_shape[0]

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
        # self.fc_mu = nn.Linear(hidden_dims[-1]*self.pre_square, self.z_shape[0])  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        # self.fc_var = nn.Linear(hidden_dims[-1]*self.pre_square, self.z_shape[0]) 
        self.fc_layer = nn.Linear(hidden_dims[-1]*self.pre_square, self.z_shape[0])

        # vq layer
        if len(z_shape) == 1:
            embedding_dim = 1
        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        beta)

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
        result = self.encoder(result)  # [B, D, 2, 2]
        result = torch.flatten(result, start_dim=1) # ([B, D*4])
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        encoding = self.fc_layer(result)

        return encoding

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


    def forward(self, data: Tensor, **kwargs) -> Tensor:
        encoding = self.encode(data)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        result = self.decode(quantized_inputs)

        return  [result, data, vq_loss, encoding]

    # only using VAE loss
    def loss_function(self,
                      *args) -> dict:
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        # recons_loss = F.mse_loss(recons, input)

        # loss = recons_loss + vq_loss
        # return {'Reconstruction_Loss': recons_loss,
        #         'VQ_Loss':vq_loss}
        return vq_loss


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
    # vae_model = BetaVAE(plane_shape=[96, 256, 256], z_shape=[1024], kl_std=0.25).cuda()

    vae_config = {"plane_shape": [3, 32, 256, 256],
                "z_shape": [1024],
                "num_embeddings": 4096,
                "beta": 0.25}

    vae_model = BetaVAE_VQ(vae_config).cuda()

    input_tensor = torch.randn(4, 3, 32, 256, 256).cuda()
    out = vae_model(input_tensor) # [result, data, vq_loss, encoding]
    loss = vae_model.loss_function(*out)
    print("loss: {}".format(loss))
    print("z shape: {}".format(out[-1].shape))
    print("reconstruct shape: {}".format(out[0].shape))
    import pdb;pdb.set_trace()

    # vq_vqe_config = {"in_channels": 3,
    #             "embedding_dim": 64,
    #             "num_embeddings": 512,
    #             "img_size": 64,
    #             "beta": 0.25
    #             }
    # vae_model = VQVAE(vq_vqe_config).cuda()

    # input_tensor = torch.randn(4, 3, 64, 64).cuda()
    # out = vae_model(input_tensor) # [result, input, vq_loss, encoding]
    # loss = vae_model.loss_function(*out)
    # print("loss: {}".format(loss))
    # print("z shape: {}".format(out[-1].shape))
    # print("reconstruct shape: {}".format(out[0].shape))
    # import pdb;pdb.set_trace()