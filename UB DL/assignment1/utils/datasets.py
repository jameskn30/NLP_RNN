import numpy as np
from tqdm import tqdm
import gc
import torch
from torch.utils.data import Dataset, DataLoader
import logging

#add utils module to sys path
import sys
sys.path.append('../')

class EnergyDataset(Dataset):

    def __init__(self, df, target, cols, log_level = logging.DEBUG):
        super().__init__()
        # cols = ['Global_active_power', 'Global_reactive_power','Voltage',
        # 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
        # 'Sub_metering_3', 'Month', 'Day', 'Year', 'Hour', 'Minute',]
        df.reset_index(inplace = True)

        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

        test_df = df.reset_index()
        features = test_df[cols].drop(target, axis = 1)
        labels = test_df[target]

        self.X, self.y = self._make_sequence(features, labels, timestep = 100)
        self.logger.debug(f'X shape =  {self.X.shape}')
        self.logger.debug(f'y shape =  {self.y.shape}')
        self.logger
        gc.collect()
    
    def _make_sequence(self, features, labels, timestep):
        '''
        @params:
            N: lenght of array
        @returns:
            X: features, shape = (d, timestep)
            y: labels, shape = (d)
            NOTE: d = N - timestep
        '''
        X = []
        y = [] 
        N = len(features)
        if timestep >= N: return to_tensor(features[:-1]), to_tensor(labels[-1]) 

        self.logger.info('building sequences ... ')
        for i in tqdm(range(N - timestep)):
            X.append(features[i: i + timestep])
            y.append(labels[i + timestep])
        self.logger.info('done building sequences')

        to_tensor = lambda x: torch.tensor(np.array(x)).to(torch.float)
        
        return to_tensor(X), to_tensor(y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)
