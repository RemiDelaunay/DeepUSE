import torch
import torch.optim as optim

from models.base_model import BaseModel
from models.network import ReUSENet
from utils.losses import GradNorm, NCC, LNCC
from utils import get_strain, warp_image

class ReUSENetModel(BaseModel):
    def __init__(self, configuration):
        """Initialize the model
        """
        super().__init__(configuration)
        self.loss_names = ['similarity_mean', 'smooth_mean', 'total', 'consistency_strain_mean']
        self.network_names = ['ReUSENet']
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        self.net = ReUSENet(in_channel=2, num_channel_initial=configuration['num_channel_initial'])
        self.net = self.net.to(self.device)
        if self.is_train:
            self.optimizer = optim.Adam(self.net.parameters(), lr=configuration['lr'])
            self.optimizers = [self.optimizer]
            self.alpha = configuration['alpha']
            self.beta = configuration['beta']
            self.corr_kernel = configuration['corr_kernel']

    def forward(self):
        """Run forward pass.
        """
        self.disp_list = []
        self.strain_list = []
        self.strain_compensated_list = []
        self.warped_img_list = []
        self.prev_state = [None, None, None, None, None]
        channel_size = self.input.size()[1]
        self.batch_fixed = self.input[:,0:1,:,:]
        for t in range(channel_size-1):
            self.batch_moving = self.input[:,t+1:t+2,:,:]
            input_ = torch.cat([self.batch_fixed, self.batch_moving],dim=1)

            self.disp_map, self.prev_state = self.net(input_, self.prev_state)
            self.warped_img = warp_image(self.batch_moving, self.disp_map)

            self.strain_list += [get_strain(self.disp_map[:, 1:2, :, :])]
            self.warped_img_list += [self.warped_img]
            self.disp_list += [self.disp_map]

            # compute motion-compensated strain
            self.strain_compensated_list += [warp_image(self.strain_list[t],self.disp_map)]
           
    def backward(self):
        """Calculate losses during training; called in every training iteration.
        """
        self.loss_similarity = [LNCC(warped_img, self.batch_fixed, self.corr_kernel) for warped_img in self.warped_img_list]
        self.loss_similarity_mean = torch.mean(torch.stack(self.loss_similarity))
        self.loss_smooth = [GradNorm(disp_map) for disp_map in self.disp_list]
        self.loss_smooth_mean = torch.mean(torch.stack(self.loss_smooth))
        if len(self.strain_compensated_list) > 1:
            self.loss_consistency_strain = [LNCC(self.strain_compensated_list[t-1][:,:,143:-143,:], self.strain_compensated_list[t][:,:,143:-143,:], self.corr_kernel) for t in range(1, len(self.strain_compensated_list))]
            self.loss_consistency_strain_mean = torch.mean(torch.stack(self.loss_consistency_strain))
            self.loss_total = 1 - self.loss_similarity_mean + self.loss_smooth_mean * self.alpha + (1 - self.loss_consistency_strain_mean) * self.beta
        else:
            self.loss_total = 1 - self.loss_similarity_mean + self.loss_smooth_mean * self.alpha

    def backward_val(self):
        """Calculate losses during testing; called in every training iteration.
        """
        self.loss_similarity = [NCC(warped_img, self.batch_fixed) for warped_img in self.warped_img_list]
        self.loss_similarity_mean = torch.mean(torch.stack(self.loss_similarity))
        self.loss_smooth = [GradNorm(disp_map) for disp_map in self.disp_list]
        self.loss_smooth_mean = torch.mean(torch.stack(self.loss_smooth))
        if len(self.strain_compensated_list) > 1:
            self.loss_consistency_strain = [NCC(self.strain_compensated_list[t-1][:,:,143:-143,:], self.strain_compensated_list[t][:,:,143:-143,:]) for t in range(1, len(self.strain_compensated_list))]
            self.loss_consistency_strain_mean = torch.mean(torch.stack(self.loss_consistency_strain))
            self.loss_total = 1 - self.loss_similarity_mean + self.loss_smooth_mean * self.alpha + (1 - self.loss_consistency_strain_mean) * self.beta
        else:
            self.loss_total = 1 - self.loss_similarity_mean + self.loss_smooth_mean * self.alpha
        
    def optimize_parameters(self):
        """Calculate gradients and update network weights.
        """
        self.loss_total.backward() # calculate gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()