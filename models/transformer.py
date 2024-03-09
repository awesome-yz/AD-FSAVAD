import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum



class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=64, dropout=0.0 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not(heads ==1 and dim_head ==dim)
        self.b = 1
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim *3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()

    def forward(self, x):
        import pdb; pdb.set_trace()
        shape = x.shape
        x = rearrange(x, '(b t) n d -> b (t n) d', b=self.b)
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = q * self.scale
        sim_qk = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = self.attend(sim_qk)
        out = einsum('b h i j, b h j d -> b h i d', attn, v )
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attn = self.attend(dots)
        # out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        out = self.to_out(out)
        return out.view(*shape)



class SpatialAttention(nn.Module):
    def __init__(self, dim, heads, dropout, dim_head=64):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        project_out = not(heads ==1 and dim_head ==dim)
        self.proj_dim = dim * 2
        self.proj_conv = nn.Conv2d(dim, self.proj_dim, kernel_size=3, padding=3, stride=3)
        self.norm = nn.BatchNorm2d(self.proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.proj_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, self.proj_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        shape = x.shape
        x = self.norm(self.proj_conv(x))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d), b h n d', h= self.heads), qkv)
        dots =torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out.view(*shape)
        return out

       
class FeedForward(nn.Module):
    def __init__(self,):
        super().__init__()

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head, dropout=0.0):
        super().__init__()
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            TemporalAttention(dim, heads, self.dim_head, dropout),
            SpatialAttention(dim, heads, self.dim_head, dropout),
            
            # FeedForward()
        ])

    def forward(self, x):
        for temp_attn, spact_attn, in self.layers:
            x = temp_attn(x) + x
            x = spact_attn(x) + x
        return self.norm(x)

