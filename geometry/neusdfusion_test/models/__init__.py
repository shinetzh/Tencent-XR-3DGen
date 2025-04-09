#!/usr/bin/python3

from models.sdf_model import SdfModel
from models.autoencoder import BetaVAE, BetaVAE2 ,BetaVAESpacial, BetaVAESpacial2, BetaVAESpacial2_Unet
from models.autoencoder import BetaVAERolloutTransformer_v2, BetaVAERolloutTransformer_v2_128, BetaVAERolloutTransformer_v3, BetaVAERolloutTransformer_v4_128, BetaVAERolloutTransformer_v10_128
from models.autoencoder import BetaVAERolloutTransformer_v2, BetaVAERolloutTransformer_v2_128, BetaVAERolloutTransformer_v3, BetaVAERolloutTransformer_v4_128, BetaVAERolloutTransformer_v5_128, BetaVAERolloutTransformer_v6_128, BetaVAERolloutTransformer_v7_128, TransformerVAE_128, BetaVAERolloutTransformer_v8_128, BetaVAERolloutTransformer_v9_128, BetaVAERolloutTransformer_v10_128, BetaVAERolloutTransformer_v11_128, BetaVAERolloutTransformer_v12_128, Transformer_v13_128_spatial
from models.autoencoder_vq import BetaVAE_VQ
from models.autoencoder_spacial import SpacialVAERolloutAware3d, SpacialVAERollout
from models.autoencoder_cross_attention import SpacialVAECrossAttention
from models.archs.encoders.conv_pointnet import UNet

from models.diffusion import *
from models.archs.diffusion_arch import * 
#from diffusion import *
from models.sdf_model import SdfModel
from models.sdf_model_nopn import SdfModelNopn
from models.sdf_model_yh import SdfModelYh, SdfModelYhColor, SdfColorModel, SdfGeoModel
from models.combined_model import CombinedModel



from models.autoencoder import BetaVAETransformer_v1_128, BetaVAETransformer_v2_128, BetaVAETransformer_v3_128
# StyleSwin 6: add two extra layers to the decoder
# StyleSwin 7: use ConvTranspose to do upsampling
# StyleSwin 8: add conv after swin
# StyleSwin 9: merge v6 and v8
from models.autoencoder_style import StyleSwinVAE_v2_128, StyleSwinVAE_v3_128, StyleSwinVAE_v4_128, StyleSwinVAE_v5_128, StyleSwinVAE_v6_128, StyleSwinVAE_v7_128, StyleSwinVAE_v8_128, StyleSwinVAE_v9_128
from models.autoencoder_style import StyleSwinVAE_v10_128


from models.autoencoder_img import AutoencoderKL_v1