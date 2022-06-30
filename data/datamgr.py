
from abc import abstractmethod
from data.dataset import *
#import pandas as pd
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SetDataManager(DataManager):
    def __init__(self, n_way, n_support, n_query, n_episode=100):
        super(SetDataManager, self).__init__()
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.n_support = n_support
        
        #self.trans_loader = TransformLoader()

    def get_data_loader(self, data_file,aug):
        transform = None
        dataset = SetDataset(data_file, self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=10, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        
        return data_loader




