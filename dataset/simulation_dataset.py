import torch
from torch.utils import data
import scipy.io as io
import numpy as np
import pandas as pd
import random

class SimulationDataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, config):
        'Initialization'
        self.list_data_id = pd.read_csv(config['partition_path'],header=None)[0].values.tolist()
        self.use_sequence = config['use_sequence']
        self.dataset_path = config['dataset_path']
        self.interframe = config['interframe']
        self.is_train = config['is_train']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_data_id)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_data_id[index]
        data = io.loadmat(self.dataset_path + ID + '.mat')
        channel_size = np.shape(data['img'])[0]
        if self.is_train:
            if self.use_sequence:
                img = data['img']
                sequence  = [torch.Tensor(img)]
            else:
                idx = random.randint(0,3)
                up =  random.randint(1,channel_size-idx-1)
                img = np.stack([data['img'][idx,...], data['img'][idx+up,...]])
                sequence  = [torch.Tensor(img)]
        else:
            img = data['img']
            sequence  = [torch.Tensor(img)]
        return sequence, ID

    def get_custom_dataloader(self, custom_configuration):
        """Get a custom dataloader (e.g. for exporting the model).
            This dataloader may use different configurations than the
            default train_dataloader and val_dataloader.
        """
        custom_collate_fn = getattr(self.dataset, "collate_fn", None)
        if callable(custom_collate_fn):
            custom_dataloader = data.DataLoader(self.dataset, **self.configuration['loader_params'], collate_fn=custom_collate_fn)
        else:
            custom_dataloader = data.DataLoader(self.dataset, **self.configuration['loader_params'])
        return custom_dataloader