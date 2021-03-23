import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import six
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import hilbert2

def transfer_to_device(x, device):
    """Transfers pytorch tensors or lists of tensors to GPU. This
        function is recursive to be able to deal with lists of lists.
    """
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = transfer_to_device(x[i], device)
    else:
        x = x.to(device)
    return x


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        return config_file


def get_scheduler(optimizer, configuration, last_epoch=-1):
    """Return a learning rate scheduler.
    """
    if configuration['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=configuration['lr_decay_iters'], gamma=configuration['lr_gamma'], last_epoch=last_epoch)
    elif configuration['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=configuration['lr_gamma'], patience=10, threshold=0.0001, min_lr=0.000005)
    elif configuration['lr_policy'] == 'cyclic':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0000001, max_lr=0.05, step_size_up=3000, mode='triangular2', last_epoch=last_epoch, cycle_momentum=False)   
    else:
        return NotImplementedError('learning rate policy [{0}] is not implemented'.format(configuration['lr_policy']))
    return scheduler


def stack_all(list, dim=0):
    """Stack all iterables of torch tensors in a list (i.e. [[(tensor), (tensor)], [(tensor), (tensor)]])
    """
    return [torch.stack(s, dim) for s in list]

def get_patches(x, x_wind=143):
    kh, dh = (x_wind*2)+1, 1
    patches = x.unfold(2, kh, dh)
    patches = torch.squeeze(patches,dim=1).permute(0,1,3,2)
    return patches

def get_strain(disp, x_wind=143):
    d = x_wind*2+1
    Uxx_list = []
    disp = get_patches(disp,x_wind=x_wind)
    depthX = torch.linspace(1,d,d)
    depthX = torch.stack([depthX,torch.ones_like(depthX)]).float().permute(1,0).cuda()
    depthX = depthX.unsqueeze(0).repeat(disp.shape[1],1,1)
    XtX = depthX.permute(0,2,1).bmm(depthX)
    for i in range(len(disp)):
        # Cholesky decomposition
        XtY = depthX.permute(0,2,1).bmm(disp[i,...])
        betas_cholesky, _ = torch.solve(XtY, XtX)
        Uxx = torch.squeeze(betas_cholesky[:,0,:])
        # pad to original size
        Uxx_list += [F.pad(Uxx, (0,0,x_wind, x_wind))]
    return torch.stack(Uxx_list).unsqueeze(1)

def to_bmode(image):
    return np.log(np.abs(np.imag(hilbert2(np.squeeze(image))))+0.01)

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        _, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def warp_image(x, ddf):
    B, _, H, W = x.size()
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