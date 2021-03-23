# from dataset.base_dataset import get_transform
# from datasets.base_dataset import BaseDataset
import torch
from torch.utils import data
import scipy.io as io
import numpy as np
import pandas as pd
import random
from scipy.signal import hilbert2


class InvivoDataset(data.Dataset):
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
                idx = random.randint(0,channel_size - 1 - self.interframe)
                up = self.interframe
                img = data['img'][idx:idx+up,100:-300,:]
                sequence  = [torch.Tensor(img).unsqueeze(1)]
            else:
                idx = random.randint(0,channel_size - 1 - self.interframe)
                up =  random.randint(1,self.interframe)
                img = np.stack([data['img'][idx,100:-300,:], data['img'][idx+up,100:-300,:]])
                sequence  = [torch.Tensor(img).unsqueeze(1)]
        else:
            n_split = round(channel_size/self.interframe)
            img = np.array_split(data['img'][:,100:-300,:], n_split, axis=0)
            sequence = [torch.Tensor(image).unsqueeze(1) for image in img]
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