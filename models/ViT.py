import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

from typing import Tuple
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, n_patches: int, emb_size: int, patch_size: Tuple[int, int], hybrid: bool = False):
        super().__init__()
        if hybrid:
            self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        else:
            #self.projection = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size[0], s2=patch_size[1])
            self.projection = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e')
            )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn(n_patches+1, emb_size)) # n_patches+1 if class_token
        
    def forward(self, x: Tensor):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class ReconstructDDF(nn.Module):
    def __init__(self, in_channels: int, patch_size: Tuple[int, int], emb_size: int, height: int, width: int):
        super().__init__()
        self.conv = nn.Sequential(
            Rearrange('b (h w) e -> b e (h) (w)', h=height//patch_size[0], w=width//patch_size[1]),
            nn.ConvTranspose2d(emb_size, in_channels, kernel_size=patch_size, stride=patch_size)
            )

    def forward(self, x):
        return self.conv(x[:,1:,:])

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# class RecurrentFeedForwardBlock(nn.Module):
#     def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
#         super().__init__()
#         self.feedforward = nn.Sequential(
#             nn.Linear(emb_size, expansion * emb_size),
#             nn.GELU(),
#             nn.Dropout(drop_p),
#             nn.Linear(expansion * emb_size, emb_size),
#         )
#     def forward(self, x, prev_state):




class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ViT(nn.Sequential):
    def __init__(self, input_shape: Tuple[int, int, int] = [2, 1600, 128], patch_size: Tuple[int, int] = [16, 16], depth: int = 12, hybrid=False):    
        in_channels, height, width = input_shape
        emb_size = patch_size[0] * patch_size[1] * in_channels
        n_patches = height * width // (patch_size[0] * patch_size[1])
        super().__init__(
            PatchEmbedding(in_channels, n_patches, emb_size, patch_size, hybrid),
            TransformerEncoder(depth, emb_size=emb_size),
            ReconstructDDF(in_channels, patch_size, emb_size, height, width)
        )

class ViTNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = [2, 1600, 128], patch_size: Tuple[int, int] = [16, 16], depth: int = 12, hybrid=False): 
        super(ViTNet, self).__init__()        
        self.vit = ViT(input_shape, patch_size, depth, hybrid)

    def forward(self, x):
        deformation_matrix = self.vit(x)
        warped_image = warp_image(x[:,1:2,:,:], deformation_matrix)
        return warped_image, deformation_matrix

def warp_image(x, ddf):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + ddf

    # scale grid to [-1,1]
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    output = F.grid_sample(x, vgrid)
    return output
