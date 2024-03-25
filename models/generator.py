import torch
import torch.nn as nn
from .backbone import Encoder, Decoder
from .diffusion import Diffusion
from .transformer import Transformer, SpatialAttention, TemporalAttention
from .unet import UNet
from utils import *
from einops import rearrange
import imageio 


class Generator(nn.Module):
    def __init__(self,*, encoder_args, decoder_args, diff_args, sp_attn_args, tp_attn_args, n_ch, att_dim, patch_dim, heads):
        super().__init__()
        self.patch_height, self.patch_width = tuple(patch_dim)
        self.heads = heads
        self.dim = att_dim
        self.encoder = Encoder(**encoder_args)
        self.diffusion = Diffusion(**diff_args)
        self.decoder = Decoder(**decoder_args)
        self.sp_attn = SpatialAttention(**sp_attn_args)
        self.tp_attn = TemporalAttention(**tp_attn_args)
        # self.transformer = Transformer(**transformer_args)
        self.embedding = nn.Conv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=1, stride=1, padding=0)
        # self.dim_expansion = nn.Conv1d(img_ch, n_ch, 3, padding=1)
        # self.dim_reduction = nn.Conv1d(self.dim * self.dim, self.dim, 3, padding=1)
        # self.unet = UNet(**unet_args)
        self.to_patch_embedding = patch_embedding(self.patch_height, self.patch_width, self.dim, channels=n_ch)
        assert ((self.patch_height * self.patch_width) % self.heads == 0) or (att_dim % self.heads == 0)

    def forward(self, imgs):
        """
        imgs has shape `[batch_size, in_channels, height, width]`
        """
        import pdb; pdb.set_trace()
        sp_z= []
        for x in imgs:
            # imageio.imwrite(os.path.join('./results/trial', f'img_{i}.png'), img_conversion(x))
            z = self.encoder(x)
            z = self.sp_attn(z)
            sp_z.append(z)
      
        sp_z = torch.cat(sp_z, dim=0)
        tp_z = self.tp_attn(sp_z)
        z_t, t, noise = self.diffusion(tp_z)
        img_out = self.decoder(z_t, t)
        imageio.imwrite(os.path.join('./results/trial', f'img_decoded.png'), img_conversion(img_out))
        return img_out, z_t, noise
        
        # img_z = img_z.unsqueeze(0) # (1 , 3, 4096)
        # img_z = self.dim_expansion(img_z)
        # img_z = img_z.permute(0, 2, 1) # (1, 4096, 64)
        # img_z = self.dim_reduction(img_z)   # (b, dim/64, t) 
        # img_z = img_z.view(1, -1, 32, 32 ) 
        # p_emb = self.to_patch_embedding(img_z) # t (h w) (p1 p2 c)
        # out = self.transformer(p_emb)
        # out = out.permute(0, 2, 1) 
        # x_t, t, noise = self.diffusion(out.unsqueeze(0))
        # img_out = self.unet(x_t, t)
        # img_out = self.decoder(x_t, t)
        # imageio.imwrite(os.path.join('./results/trial', f'img_decoded.png'), img_conversion(img_out))
        # return img_out,x_t, noise