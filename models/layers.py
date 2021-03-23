import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    Adapted from "RVOS: End-to-End Recurrent Network for Video Object Segmentation"
    """
    def __init__(self, input_size, hidden_size, kernel_size=3, padding=1):
        super(ConvLSTMCell,self).__init__()
        self.use_gpu = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_hidden], 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)
        state = [hidden,cell]
        return state

class convBNrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class deconvBNrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels, kernel_size, stride=2,padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class convBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1, bias=False),
            nn.BatchNorm2d(out_channels,track_running_stats=False))
        
    def forward(self, x):
        return self.conv(x)

class ddf_summand(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=2, padding=1, bias=True)
                                    for in_channels in in_channels_list])

    def forward(self, x, size_out):
        x1_resize = []
        for i, _ in enumerate(self.convs):
            x1 = self.convs[i](x[4-i])
            x1_resize.append(F.interpolate(x1, size=size_out, mode='bilinear', align_corners=True))
        return torch.sum(torch.stack(x1_resize,dim=4), dim=4)

class DownResBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_0 = convBNrelu(in_channels, out_channels, kernel_size)
        self.conv_1 = convBNrelu(out_channels, out_channels, kernel_size)
        self.conv_2 = convBN(out_channels, out_channels, kernel_size)
        self.acti = nn.ReLU()
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        x1 = self.conv_0(x)
        x2 = self.conv_1(x1)
        x3 = self.conv_2(x2)
        x3 += x1
        x3 = self.acti(x3)
        down = self.down(x3)
        return x1, down

class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.transposed = deconvBNrelu(in_channels,out_channels, kernel_size)
        self.conv_0 = convBNrelu(out_channels, out_channels, kernel_size)
        self.conv_1 = convBN(out_channels, out_channels, kernel_size)
        self.acti = nn.ReLU()

    def forward(self, x, input_skip):
        add_up = self.transposed(x)
        add_up += input_skip
        add_up += additive_up_sampling(x, input_skip)
        x1 = self.conv_0(add_up)
        x2 = self.conv_1(x1)
        x2 += x1
        x2 = self.acti(x2)
        return x2

class LstmUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.transposed = deconvBNrelu(in_channels,out_channels, kernel_size)
        self.lstm = ConvLSTMCell(out_channels, out_channels, kernel_size)

    def forward(self, x, input_skip, hidden_state_temporal): 
        add_up = self.transposed(x)
        add_up += input_skip
        add_up += additive_up_sampling(x, input_skip)
        x1 = self.lstm(add_up, hidden_state_temporal)
        return x1

def additive_up_sampling(input, input_skip):
    upsampled = F.interpolate(input,size=input_skip.size()[2:4], mode='bilinear', align_corners=True)
    upsampled_split = torch.chunk(upsampled, 2, dim=1)
    upsampled_stack = torch.stack(upsampled_split, dim=1)
    upsampled_final = torch.sum(upsampled_stack, dim=1)
    return upsampled_final

