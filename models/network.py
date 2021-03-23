import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import *
from utils import warp_image

use_cuda = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, in_channel=2, num_channel_initial=16):
        super(Encoder, self).__init__()
        self.in_channel = in_channel
        self.num_channel_initial = num_channel_initial
        self.ch = [int(self.num_channel_initial*(2**i)) for i in range(5)]

        self.down_res_0 = DownResBlock(in_channels=self.in_channel, out_channels=self.ch[0], kernel_size=3)
        self.down_res_1 = DownResBlock(self.ch[0], self.ch[1])
        self.down_res_2 = DownResBlock(self.ch[1], self.ch[2])
        self.down_res_3 = DownResBlock(self.ch[2], self.ch[3])
        
    def forward(self, x):
        conv0, down0 = self.down_res_0(x)
        conv1, down1 = self.down_res_1(down0)
        conv2, down2 = self.down_res_2(down1)
        conv3, down3 = self.down_res_3(down2)
        return [down3,conv3,conv2,conv1,conv0]

class Decoder(nn.Module):
    def __init__(self, num_channel_initial=16):
        super(Decoder, self).__init__()
        self.num_channel_initial = num_channel_initial
        self.ch = [int(self.num_channel_initial*(2**i)) for i in range(5)]

        self.bottleneck = convBNrelu(self.ch[3],self.ch[4])
        self.up_res_0 = UpResBlock(self.ch[4],self.ch[3])
        self.up_res_1 = UpResBlock(self.ch[3],self.ch[2])
        self.up_res_2 = UpResBlock(self.ch[2],self.ch[1])
        self.up_res_3 = UpResBlock(self.ch[1],self.ch[0])
        self.ddf_summand = ddf_summand(self.ch)

    def forward(self, encoded):
        decoded = [self.bottleneck(encoded[0])]
        decoded += [self.up_res_0(decoded[0],encoded[1])]
        decoded += [self.up_res_1(decoded[1],encoded[2])]
        decoded += [self.up_res_2(decoded[2],encoded[3])]
        decoded += [self.up_res_3(decoded[3],encoded[4])]
        self.ddf = self.ddf_summand(decoded, encoded[4].size()[2:4])
        return self.ddf

class Recurrent_decoder(nn.Module):
    def __init__(self, num_channel_initial=16):
        super(Recurrent_decoder, self).__init__()
        self.num_channel_initial = num_channel_initial
        self.ch = [int(self.num_channel_initial*(2**i)) for i in range(5)]

        self.bottleneck = ConvLSTMCell(self.ch[3],self.ch[4])
        self.up_res_0 = LstmUpBlock(self.ch[4],self.ch[3])
        self.up_res_1 = LstmUpBlock(self.ch[3],self.ch[2])
        self.up_res_2 = LstmUpBlock(self.ch[2],self.ch[1])
        self.up_res_3 = LstmUpBlock(self.ch[1],self.ch[0])
        self.ddf_summand = ddf_summand(self.ch)

    def forward(self, encoded,prev_state):
        decoded = [self.bottleneck(encoded[0],prev_state[0])]
        decoded += [self.up_res_0(decoded[0][0],encoded[1],prev_state[1])]
        decoded += [self.up_res_1(decoded[1][0],encoded[2],prev_state[2])]
        decoded += [self.up_res_2(decoded[2][0],encoded[3],prev_state[3])]
        decoded += [self.up_res_3(decoded[3][0],encoded[4],prev_state[4])]

        hidden_list = [decoded[i] for i in range(0,len(decoded))]
        ddf_list = [decoded[i][0] for i in range(0,len(decoded))]
        self.ddf = self.ddf_summand(ddf_list, encoded[4].size()[2:4])
        return self.ddf, hidden_list

class ReUSENet(nn.Module):
    def __init__(self, in_channel=2, num_channel_initial=16):
        super(ReUSENet, self).__init__()
        self.encoder = Encoder(in_channel, num_channel_initial)
        self.decoder = Recurrent_decoder(num_channel_initial)

    def forward(self, x, prev_state):
        features = self.encoder(x)
        deformation_matrix, hidden_list = self.decoder(features, prev_state)
        return deformation_matrix, hidden_list

class USENet(nn.Module):
    def __init__(self, in_channel=2, num_channel_initial=16):
        super(USENet, self).__init__()
        self.encoder = Encoder(in_channel, num_channel_initial)
        self.decoder = Decoder(num_channel_initial)

    def forward(self, x):
        features = self.encoder(x)
        deformation_matrix = self.decoder(features)
        return deformation_matrix
