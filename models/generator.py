import torch
import torch.nn as nn
from .encoder import Encoder
from .diffusion import Diffusion
from .transformer import Transformer
from utils import *
from einops import rearrange


class Generator(nn.Module):
    def __init__(self,*, encoder_args, diff_args, transformer_args, att_dim, patch_dim, heads):
        super().__init__()
        self.patch_height, self.patch_width = tuple(patch_dim)
        self.heads = heads
        self.dim = att_dim
        self.encoder = Encoder(**encoder_args)
        self.diffusion = Diffusion(**diff_args)
        self.transformer = Transformer(**transformer_args)
        self.to_patch_embedding = patch_embedding(self.patch_height, self.patch_width, self.dim)
        assert ((self.patch_height * self.patch_width) % self.heads == 0) or (att_dim % self.heads == 0)

    def forward(self, imgs):
        """
        imgs has shape `[batch_size, in_channels, height, width]`
        """
        import pdb; pdb.set_trace()
        img_z = []
        # img_z = torch.zeros(imgs[0].shape, device=imgs.device)
        for i, x in enumerate(imgs):
            # add noise to image
            # x_t, t, noise = self.diffusion(x)
            z = self.encoder(x)
            img_z.append(z)
        import pdb; pdb.set_trace()
        img_z = torch.cat(img_z, dim=0)
        p_emb = self.to_patch_embedding(img_z) # t (h w) (p1 p2 c)
        out = self.transformer(p_emb)
        return 