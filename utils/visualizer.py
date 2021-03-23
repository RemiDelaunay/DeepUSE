import torch
import os
import sys
import math
import visdom
import numpy as np
from subprocess import Popen, PIPE

import utils

class Visualizer():
    """This class includes several functions that can display images and print logging information.
    """

    def __init__(self, configuration):
        """Initialize the Visualizer class.

        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.display_id = 0
        self.name = configuration['name']
        self.ncols = 0

        if not os.path.exists(self.configuration['log_path']):
            os.mkdir(self.configuration['log_path'])
        self.vis = visdom.Visdom(log_to_filename=self.configuration['log_path']+'visdom_'+self.configuration['name'], offline=True)
        # if not self.vis.check_connection():
        #     self.create_visdom_connections()
        #self.training_log = self.vis.text('')

    def reset(self):
        """Reset the visualization.
        """
        pass


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at the default port.
        """
        cmd = sys.executable + ' -m visdom.server'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def plot_training_losses(self, epoch, counter_ratio, losses):
        """Display the current losses on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
            losses: Training losses stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([losses[k] for k in self.loss_plot_data['legend']])
        x = np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1)
        y = np.array(self.loss_plot_data['Y'])
        self.vis.line(
            X=x,
            Y=y,
            opts={
                'title': 'training loss over time',
                'legend': self.loss_plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)


    def plot_validation_losses(self, epoch, counter_ratio, metrics):
        """Display the current validation metrics on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            losses: Validation metrics stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.val_plot_data['X'].append(epoch+ counter_ratio)
        self.val_plot_data['Y'].append([metrics[k] for k in self.val_plot_data['legend']])
        x = np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1)
        y = np.array(self.val_plot_data['Y'])
        self.vis.line(
            X=x,
            Y=y,
            opts={
                'title': 'validation loss over time',
                'legend': self.val_plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'metric'},
            win=self.display_id+1)

    def show_validation_images(self, images):
        """Display validation images. The images have to be in the form of a tensor with
        [(image, label, prediction), (image, label, prediction), ...] in the 0-th dimension.
        """
        # zip the images together so that always the image is followed by label is followed by prediction
        images = images.permute(1,0,2,3)
        images = images.reshape((images.shape[0]*images.shape[1],images.shape[2],images.shape[3]))

        # add a channel dimension to the tensor since the excepted format by visdom is (B,C,H,W)
        images = images[:,None,:,:]

        self.vis.images(images, win=self.display_id+3, nrow=3)
        
    def display_validation_images(self, image_list, image_id):
        import matplotlib.pyplot as plt
        m = 6
        n = 1 #math.ceil((len(image_list)/10))
        fig, axes = plt.subplots(n,m, figsize=(20, 3), sharex=True, sharey=True)
        fig.suptitle(image_id[0])
        for image, ax in zip(image_list[0:11] ,axes.flatten()):
            ax.imshow(image.cpu().numpy(), aspect='auto', cmap='jet')
        plt.show()
        self.vis.matplot(plt, win=image_id[0])
    
    def save_validation_images(self, image_list, similarity_list, image_id, result_type='strain'):
        import matplotlib.pyplot as plt
        m = len(image_list)//3
        n = 3
        similarity_list = similarity_list*n
        fig, axes = plt.subplots(n,m, figsize=(40, 3*n), sharex=True, sharey=True)
        fig.suptitle(image_id[0])
        for image, ax, similarity in zip(image_list ,axes.flatten(),similarity_list):
            try:
                image = image.cpu().numpy()
            except AttributeError:
                image = image
            pcm = ax.imshow(image, aspect='auto', cmap='jet')
            fig.colorbar(pcm,ax=ax)
            ax.title.set_text(np.round(similarity.cpu().numpy(),2)) 
        fig.savefig(self.configuration['log_path']+image_id+'_'+result_type+'.png')
        plt.clf()


    def print_losses(self, epoch, max_epochs, iter, max_iters, losses, image_id, name='training'):
        """Print current losses on console.
        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
        message = '[epoch: {}/{}, iter: {}/{}, image_id: {}] '.format(epoch, max_epochs, iter, max_iters, image_id)
        for k, v in losses.items():
            message += '{0}: {1:.6f} '.format(k, v)
        print(message) 
        self.vis.text(message, win=name, append=True)
