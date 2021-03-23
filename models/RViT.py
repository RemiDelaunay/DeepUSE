import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Tuple
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import copy


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
            #nn.Linear(emb_size,emb_size),
            Rearrange('b (h w) e -> b e (h) (w)', h=height//patch_size[0], w=width//patch_size[1]),
            nn.ConvTranspose2d(emb_size, in_channels, kernel_size=patch_size, stride=patch_size)
            )

    def forward(self, x):
        return self.conv(x[:,1:,:])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor) -> Tensor:
        residual1 = src
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src)[0]
        src = residual1 + self.dropout1(src2)
        
        residual2 = src
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual2 + self.dropout2(src2)
        return src

class RecurrentTransformerEncoderLayer1(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super(RecurrentTransformerEncoderLayer1, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.lstm = nn.LSTMCell(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RecurrentTransformerEncoderLayer1, self).__setstate__(state)

    def forward(self, src: Tensor, prev_state=None) -> Tensor:
        residual1 = src
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src)[0]
        src = residual1 + self.dropout1(src2)
        
        residual2 = src
        src = self.norm2(src)
        current_state = self.lstm(torch.squeeze(src), prev_state)
        src2 = self.linear2(self.dropout(current_state[0]))
        src = residual2 + self.dropout2(src2)
        return src, current_state

class RecurrentTransformerEncoderLayer2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super(RecurrentTransformerEncoderLayer2, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.lstm = nn.LSTMCell(d_model, dim_feedforward)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear3 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RecurrentTransformerEncoderLayer2, self).__setstate__(state)

    def forward(self, src: Tensor, prev_state=None) -> Tensor:
        current_state = self.lstm(torch.squeeze(src), prev_state)
        src3 = self.linear3(current_state[0].unsqueeze(0))

        residual1 = src
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src)[0]
        src = residual1 + self.dropout1(src2)
        
        residual2 = src
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual2 + self.dropout2(src2) + src3
        return src, current_state

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: Tensor) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output)

        return output

class RecurrentTransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers):
        super(RecurrentTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: Tensor, prev_state=None) -> Tensor:
        output = src
        current_state_list = []
        if prev_state is None:
            prev_state = [None for mod in self.layers]
        
        for mod, prev_state in zip(self.layers, prev_state):
            output, current_state = mod(output, prev_state)
            current_state_list += [current_state]


        return output , current_state_list

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        residual1 = tgt
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = residual1 + self.dropout1(tgt2)
        

        residual2 = tgt
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = residual2 + self.dropout2(tgt2)
        
        residual3 = tgt
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual3 + self.dropout3(tgt2)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(output, memory)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class RViTNet1(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = [2, 1600, 128], patch_size: Tuple[int, int] = [16, 16], depth: int = 12, hybrid=False): 
        super(RViTNet1, self).__init__()
        in_channels, height, width = input_shape
        emb_size = patch_size[0] * patch_size[1] * in_channels
        n_patches = height * width // (patch_size[0] * patch_size[1])  
        nhead = 8    
        self.embedding = PatchEmbedding(in_channels, n_patches, emb_size, patch_size, hybrid)
        encoderLayer = RecurrentTransformerEncoderLayer1(emb_size, nhead)
        self.encoder = RecurrentTransformerEncoder(encoderLayer, depth)
        # decoderLayer = RecurrentTransformerEncoderLayer1(emb_size, nhead)
        # self.decoder = RecurrentTransformerEncoder(decoderLayer, depth)
        self.reconstruct = ReconstructDDF(in_channels, patch_size, emb_size, height, width)

    def forward(self, src, prev_state=None):
        src_patch = self.embedding(src)
        #features = self.encoder(src_patch)
        # output, current_state = self.decoder(features, prev_state)
        output, current_state = self.encoder(src_patch, prev_state)
        deformation_matrix = self.reconstruct(output)
        warped_image = warp_image(src[:,1:2,:,:], deformation_matrix)
        return warped_image, deformation_matrix, current_state

class RViTNet2(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = [2, 1600, 128], patch_size: Tuple[int, int] = [16, 16], depth: int = 12, hybrid=False): 
        super(RViTNet2, self).__init__()
        in_channels, height, width = input_shape
        emb_size = patch_size[0] * patch_size[1] * in_channels
        n_patches = height * width // (patch_size[0] * patch_size[1])  
        nhead = 8    
        self.embedding = PatchEmbedding(in_channels, n_patches, emb_size, patch_size, hybrid)
        encoderLayer = RecurrentTransformerEncoderLayer2(emb_size, nhead)
        self.encoder = RecurrentTransformerEncoder(encoderLayer, depth)
        self.reconstruct = ReconstructDDF(in_channels, patch_size, emb_size, height, width)

    def forward(self, src, prev_state=None):
        src_patch = self.embedding(src)
        output, current_state = self.encoder(src_patch, prev_state)
        deformation_matrix = self.reconstruct(output)
        warped_image = warp_image(src[:,1:2,:,:], deformation_matrix)
        return warped_image, deformation_matrix, current_state

class ViTNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = [2, 1600, 128], patch_size: Tuple[int, int] = [16, 16], depth: int = 12, hybrid=False): 
        super(ViTNet, self).__init__()
        in_channels, height, width = input_shape
        emb_size = patch_size[0] * patch_size[1] * in_channels
        n_patches = height * width // (patch_size[0] * patch_size[1])  
        nhead = 8    
        self.embedding = PatchEmbedding(in_channels, n_patches, emb_size, patch_size, hybrid)
        encoderLayer = TransformerEncoderLayer(emb_size, nhead)
        self.encoder = TransformerEncoder(encoderLayer, depth)
        self.reconstruct = ReconstructDDF(in_channels, patch_size, emb_size, height, width)

    def forward(self, src):
        src_patch = self.embedding(src)
        output = self.encoder(src_patch)
        deformation_matrix = self.reconstruct(output)
        warped_image = warp_image(src[:,1:2,:,:], deformation_matrix)
        return warped_image, deformation_matrix

class TransformerNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = [2, 1600, 128], patch_size: Tuple[int, int] = [16, 16], depth: int = 6, hybrid=False): 
        super(TransformerNet, self).__init__()
        in_channels, height, width = input_shape
        emb_size = patch_size[0] * patch_size[1] * in_channels
        n_patches = height * width // (patch_size[0] * patch_size[1])
        nhead = 8
        self.embedding = PatchEmbedding(in_channels, n_patches, emb_size, patch_size, hybrid)     
        encoderLayer = TransformerEncoderLayer(emb_size, nhead)
        self.encoder = TransformerEncoder(encoderLayer, depth)
        decoderLayer = TransformerDecoderLayer(emb_size,nhead)
        self.decoder = TransformerDecoder(decoderLayer, depth)
        self.reconstruct = ReconstructDDF(in_channels, patch_size, emb_size, height, width)

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = torch.zeros_like(src)
        src_patch = self.embedding(src)
        tgt_patch = self.embedding(tgt)
        memory = self.encoder(src_patch)
        output = self.decoder(tgt_patch, memory)
        deformation_matrix = self.reconstruct(output)
        warped_image = warp_image(src[:,1:2,:,:], deformation_matrix)
        return warped_image, deformation_matrix

class Encoder_vit(nn.Module):
    def __init__(self, in_channel=2, num_channel_initial=32):
        super(Encoder_vit, self).__init__()
        self.in_channel = in_channel
        self.num_channel_initial = num_channel_initial
        self.ch = [int(self.num_channel_initial*(2**i)) for i in range(5)]

        self.down_res_0 = DownResBlock(in_channels=self.in_channel, out_channels=self.ch[0], kernel_size=3)
        self.down_res_1 = DownResBlock(self.ch[0], self.ch[1])
        self.down_res_2 = DownResBlock(self.ch[1], self.ch[2])
        self.down_res_3 = DownResBlock(self.ch[2], self.ch[3])
        self.bottleneck = convBNrelu(self.ch[3],self.ch[4])

    def forward(self, x):
        _, down0 = self.down_res_0(x)
        _, down1 = self.down_res_1(down0)
        _, down2 = self.down_res_2(down1)
        _, down3 = self.down_res_3(down2)
        conv4 = self.bottleneck(down3)
        return conv4