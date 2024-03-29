import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum



class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=64, dropout=0.0 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not(heads ==1 and dim_head ==dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv3d(dim, inner_dim *3, kernel_size=(3,1,1), stride= 1, padding=0,bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()

    def forward(self, x):
        T, H, W, C = x.shape
        x = self.norm(x)
        x = rearrange(x, '(B T) H W C -> B C T H W', B=1)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'B (h C) ... -> B h C (...)', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'B h C ... -> B (h C) ...')
        out = self.to_out(out)
        return out.reshape(-1, C, H, W)


class SpatialAttention(nn.Module):
    def __init__(self, dim, sp_dim, heads=1, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads 
        project_out = not(heads==1 and dim_head ==dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(sp_dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()
        

    def forward(self, x):
        b, c, ph, pw = x.shape # (batch, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # position_wise attention branch
        q, k, vp = map(lambda t: rearrange(t, 'b (h c) ph pw -> b h (ph pw) c', h=self.heads), qkv)
        dots_position = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_position = self.attend(dots_position)
        attn_position = self.dropout(attn_position)
        out_position = torch.matmul(attn_position, vp)
        out_position = self.to_out(out_position)
        out_position = rearrange(out_position, 'b h (ph pw) c -> b ph pw (h c)', ph=ph, pw=pw) 
        
        # channel_wise attention branch
        q, k, vc = map(lambda t: rearrange(t, 'b (h c) ph pw -> b h c (ph pw)', h=self.heads), qkv)
        dots_channel = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_channel = self.attend(dots_channel)
        attn_channel = self.dropout(attn_channel)
        out_channel = torch.matmul(attn_channel, vc)
        out_channel = self.to_out(out_channel)
        out_channel = rearrange(out_channel, 'b h c (ph pw) -> b ph pw (h c)', ph=ph, pw=pw)

        # adding both position_wise and channel_wise attention maps
        out = out_position + out_channel
        return out

       
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=3, padding=1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, transformer_args, sp_attn_args, tp_attn_args):
        super().__init__()
        import pdb; pdb.set_trace()
        self.dim, self.mlp_dim, self.ff_dropout = transformer_args.values()
        self.norm = nn.LayerNorm(self.dim)
        self.transformer = nn.Sequential(
            SpatialAttention(**sp_attn_args),
            TemporalAttention(**tp_attn_args),
            FeedForward(self.dim, self.mlp_dim, dropout=self.ff_dropout)
        )

    def forward(self, x):
        import pdb; pdb.set_trace()
        x = self.transformer(x) + x
        x = self.norm(x)
        return x

