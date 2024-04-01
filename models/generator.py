import torch
import torch.nn as nn
from .backbone import Encoder, Decoder
from .diffusion import Diffusion
from .transformer import Transformer, SpatialAttention, TemporalAttention, FeedForward
from .autoencoder import AutoencoderKL
from .unet import UNet
from utils import *
from einops import rearrange
import imageio 
from ldm.diffusionmodules.model import Encoder, Decoder


class Generator(nn.Module):
    def __init__(self,*, encoder_decoder_args, sp_attn_args, tp_attn_args, ff_args, chkpt_path=None):
        super().__init__()
        # self.patch_height, self.patch_width = tuple(patch_dim)
        # self.heads = heads
        # self.dim = att_dim
        # self.encoder = Encoder(**encoder_args)
        # self.diffusion = Diffusion(**diff_args)
        # self.decoder = Decoder(**decoder_args)
        # self.transformer = Transformer(transformer_args, sp_attn_args, tp_attn_args)
        # self.autoencoder = AutoencoderKL(**autoencoder_args)
        if chkpt_path:
            self.encoder = self.init_chkpts(Encoder(**encoder_decoder_args), chkpt_path)
            self.decoder = self.init_chkpts(Decoder(**encoder_decoder_args), chkpt_path)
        else:
            self.encoder = Encoder(**encoder_decoder_args)
            self.decoder = Decoder(**encoder_decoder_args)
        self.sp_attn = SpatialAttention(**sp_attn_args)
        self.tp_attn = TemporalAttention(**tp_attn_args)
        self.ff = FeedForward(**ff_args)

    def init_chkpts(self, model, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        model.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        return model

    def forward(self, imgs):
        """
        imgs has shape `[batch_size, in_channels, height, width]`
        """
        sp_z= []
        for i,x in enumerate(imgs):
            imageio.imwrite(os.path.join('./results/trial_2', f'img_{i}.png'), img_conversion(x))
            # z = self.encoder(x)
            z = self.encoder(x)
            # z_decoded = self.decoder(z)
            z = self.sp_attn(z) 
            sp_z.append(z)
        sp_z = torch.cat(sp_z, dim=0)
        tp_z = self.tp_attn(sp_z)
        ff_z = self.ff(tp_z)
        img_out = self.decoder(ff_z)
        # imageio.imwrite(os.path.join('./results/trial_2', f'img_decoded.png'), img_conversion(z_decoded))
        # z_t, t, noise = self.diffusion(tp_z)
        # img_out = self.decoder(z_t, t)
        # imageio.imwrite(os.path.join('./results/trial', f'img_decoded.png'), img_conversion(img_out))
        # return img_out, z_t, noise
        return img_out
        