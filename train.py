import argparse
import time
import math
import numpy as np
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from dataset import create_dataloader
from utils import parse_configuration
from models import create_model
from utils.visualizer import Visualizer

"""Performs training of a specified model.
    
Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
def train(config_file, export=False):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    train_dataset = create_dataloader(configuration['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    print('The number of training samples = {0}'.format(train_dataset_size))

    val_dataset = create_dataloader(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params'])   # create a visualizer that displays images and plots
    # Tensorboard writer
    training_log = SummaryWriter(configuration['visualization_params']['log_path']+'tensorboard/training/')
    validation_log = SummaryWriter(configuration['visualization_params']['log_path']+'tensorboard/validation/')

    starting_epoch = configuration['model_params']['load_checkpoint']
    num_epochs = configuration['model_params']['max_epochs'] + 1

    # iter number for summary writer
    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()

        train_iterations = len(train_dataset)
        validation_iterations = len(val_dataset)
        train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']

        model.train()

        loss_total = []
        loss_smooth = []
        loss_consistency = []
        loss_similarity = []

        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            image_list, image_id = data
            for _, image in enumerate(image_list):
                model.set_input(image)         # unpack data from dataset and apply preprocessing
                model.forward()
                model.backward()

                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                losses = model.get_current_losses()
                visualizer.print_losses(epoch, num_epochs, i, train_iterations, losses, image_id, name='training')

                # add loss to tensorboard
                loss_total.append(losses['total'])
                loss_smooth.append(losses['smooth_mean'])
                loss_similarity.append(losses['similarity_mean'])
                loss_consistency.append(losses['consistency_strain_mean'])

        training_log.add_scalar('loss/total',
                    np.mean(np.stack(loss_total)),
                    epoch)
        
        training_log.add_scalar('loss/smooth',
                    np.mean(np.stack(loss_smooth)),
                    epoch)

        training_log.add_scalar('loss/similarity',
                    np.mean(np.stack(loss_similarity)),
                    epoch)

        training_log.add_scalar('loss/consistency',
                    np.mean(np.stack(loss_consistency)),
                    epoch)

        training_log.add_scalar('learning rate',
                    model.lr,
                    epoch)

        training_log.flush()

        if epoch % configuration['validation_freq'] == 0:
           # model.eval()
            loss_total = []
            loss_smooth = []
            loss_consistency = []
            loss_similarity = []
            print('************************VALIDATION***************************')
            for i, data in enumerate(val_dataset):
                image_list, image_id = data
                for _, image in enumerate(image_list):
                    model.set_input(image)
                    model.test()
                    model.backward_val()

                    losses = model.get_current_losses()
                    visualizer.print_losses(epoch, num_epochs, i, math.floor(validation_iterations / train_batch_size), losses, image_id, name='validation')
                    
                    # add loss to tensorboard
                    loss_total.append(losses['total'])
                    loss_smooth.append(losses['smooth_mean'])
                    loss_similarity.append(losses['similarity_mean'])
                    loss_consistency.append(losses['consistency_strain_mean'])
            validation_log.add_scalar('loss/total',
                        np.mean(np.stack(loss_total)),
                        epoch)
            
            validation_log.add_scalar('loss/smooth',
                        np.mean(np.stack(loss_smooth)),
                        epoch)

            validation_log.add_scalar('loss/similarity',
                        np.mean(np.stack(loss_similarity)),
                        epoch)

            validation_log.add_scalar('loss/consistency',
                        np.mean(np.stack(loss_consistency)),
                        epoch)
            validation_log.flush()

        if epoch == num_epochs:
            model.save_networks(epoch)
            model.save_optimizers(epoch)

        if epoch % configuration['model_update_freq'] == 0:
            print('Saving model at the end of epoch {0}'.format(epoch))
            model.save_networks(epoch)
            model.save_optimizers(epoch)

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time))
        if configuration['model_params']['lr_policy'] == 'plateau':
            model.update_learning_rate(np.mean(np.stack(loss_total))) # update learning rates every epoch
        else:
            model.update_learning_rate() # update learning rates every epoch

    if export:
        print('Exporting model')
        model.eval()
        custom_configuration = configuration['train_dataset_params']
        custom_configuration['loader_params']['batch_size'] = 1 # set batch size to 1 for tracing
        dl = create_dataloader(custom_configuration)
        sample_input = next(iter(dl)) # sample input from the training dataset
        model.set_input(sample_input)
        model.export()

    return model.get_hyperparam_result()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)