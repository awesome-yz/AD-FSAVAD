import torch
import torch.nn as nn
from .backbone import Encoder, Decoder
from .diffusion import Diffusion
from .transformer import Transformer, SpatialAttention, TemporalAttention, FeedForward
from .unet import UNet
from utils import *
from einops import rearrange
import imageio 


class Generator(nn.Module):
    def __init__(self,*, encoder_args, decoder_args, diff_args, sp_attn_args, tp_attn_args, ff_args):
        super().__init__()
        # self.patch_height, self.patch_width = tuple(patch_dim)
        # self.heads = heads
        # self.dim = att_dim
        self.encoder = Encoder(**encoder_args)
        # self.diffusion = Diffusion(**diff_args)
        self.decoder = Decoder(**decoder_args)
        # self.transformer = Transformer(transformer_args, sp_attn_args, tp_attn_args)
        self.sp_attn = SpatialAttention(**sp_attn_args)
        self.tp_attn = TemporalAttention(**tp_attn_args)
        self.ff = FeedForward(**ff_args)

    def forward(self, imgs):
        """
        imgs has shape `[batch_size, in_channels, height, width]`
        """
        sp_z= []
        import pdb; pdb.set_trace()
        for i,x in enumerate(imgs):
            # imageio.imwrite(os.path.join('./results/trial_2', f'img_{i}.png'), img_conversion(x))
            z = self.encoder(x)
            z = self.sp_attn(z) 
            sp_z.append(z)

        sp_z = torch.cat(sp_z, dim=0)
        tp_z = self.tp_attn(sp_z)
        ff_z = self.ff(tp_z)
        img_out = self.decoder(ff_z, None)
        imageio.imwrite(os.path.join('./results/trial_2', f'img_decoded.png'), img_conversion(img_out))
        # z_t, t, noise = self.diffusion(tp_z)
        # img_out = self.decoder(z_t, t)
        # imageio.imwrite(os.path.join('./results/trial', f'img_decoded.png'), img_conversion(img_out))
        # return img_out, z_t, noise
        return img_out
        