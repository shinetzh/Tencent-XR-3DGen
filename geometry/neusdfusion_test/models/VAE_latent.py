class BetaVAERolloutTransformer_v4_latent_128(nn.Module):
    def __init__(self, vae_config) -> None:
        super(BetaVAERolloutTransformer_v4_latent_128, self).__init__()
        print("vae type: BetaVAERolloutTransformer_v4_128")

        kl_std = vae_config.get("kl_std", 0.25)
        kl_weight = vae_config.get("kl_weight", 0.001)
        plane_shape = vae_config.get("plane_shape", [3, 32, 256, 256])
        z_shape = vae_config.get("z_shape", [4, 64, 64])
        num_heads = vae_config.get("num_heads", 16)
        transform_depth = vae_config.get("transform_depth", 1)

        print("z_shape: " + str(z_shape))
        print("plane_shape: " + str(plane_shape))

        self.plane_dim = len(plane_shape) + 1
        self.plane_shape = plane_shape
        self.z_shape = z_shape
        
        self.kl_std = kl_std
        self.kl_weight = kl_weight

        #hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024, 2*self.z_shape[0]]
        hidden_dims = [512, 512, 512, 512, 512, 512, self.z_shape[0]]
        #feature size:  64,  32,  16,   8,    4,    8,   16,   32

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

        self.in_layer_01 = nn.Sequential(ResBlock(
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

        self.encoders_down_01 = nn.ModuleList()
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
            self.encoders_down_01.append(nn.Sequential(*modules))
        
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
            
            self.encoders_down_01.append(nn.Sequential(
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
        
        self.fc_mu = nn.Linear(24576, self.z_shape[0])  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        self.fc_var = nn.Linear(24576, self.z_shape[0]) 

        print("feature map size: " + str(hidden_dims[-1]))

        ## build decoder
        hidden_dims_decoder = [1024, 1024, 512, 512]
        #feature size:  16,    8,   4,   8,    16,  32,  64

        self.decoder_in_layer = nn.Linear(self.z_shape[0], 24576)

        in_channels = 512

        self.combine = nn.Sequential(
                                    nn.Conv2d(512*2, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.SiLU(),
                                    SpatialTransformer(512,
                                                        num_heads,
                                                        dim_head,
                                                        depth=transform_depth,
                                                        context_dim=512,
                                                        disable_self_attn=False,
                                                        use_linear=True,
                                                        attn_type="softmax-xformers",
                                                        use_checkpoint=True,
                                                        ),
                                    nn.BatchNorm2d(512),
                                    nn.SiLU()
                                    )

        self.decoders_up = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims_decoder[0:]):
            modules = []
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
        feature_01 = self.in_layer_01(result)

        # hidden_dims = [512, 512, 1024, 1024, 1024, 1024, 1024]
        # #feature size:  64,  32,  16,   8,    4,    8,   16
        #features_down = []
        #features_down_01 = []
        #f_large = []
        for i, module in enumerate(self.encoders_down):
            feature = module(feature)
            #print(str(i) + " : " + str(feature.shape))
            #if i in [2, 3]:
            #    features_down.append(feature)
        
        for i, module in enumerate(self.encoders_down_01):
            feature_01 = module(feature_01)
            #print(str(i) + " : " + str(feature_01.shape))
            #if i in [2, 3]:
            #    features_down_01.append(feature_01)


        #print(feature.shape)
        feature = torch.cat([feature, feature_01], dim=1)
        feature = self.combine(feature)
        #print(feature.shape)
        feature = torch.flatten(feature, start_dim=1) # ([B, D*4])
        #print(feature.shape)

        #encode_channel = self.z_shape[0]
        #mu = feature[:, :encode_channel, ...]
        #log_var = feature[:, encode_channel:, ...]

        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''
        z
        '''

        # hidden_dims_decoder = [1024, 1024, 1024, 1024, 1024, 512, 512]
        #         #feature size:  16,    8,   4,   8,    16,  32,  64
        x = self.decoder_in_layer(z)
        x = x.view(-1, 512, 4, 12)

        for i, module in enumerate(self.decoders_up):
            x = module(x)
            #print(str(i) + " : " + str(x.shape))


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

        #print(fe[0].shape)
        #print(fe[1].shape)

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