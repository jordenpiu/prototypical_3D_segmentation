import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse
import sys
sys.argv=['']
del sys
from copy import deepcopy

from data.datamgr import SetDataManager
from methods.protonet import ProtoNet
from models.vnet import *

CUDA_VISIBLE_DEVICES=0

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_n_way', default=2, type=int,
                        help='class num to classify for training')
    parser.add_argument('--test_n_way', default=2, type=int,
                        help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=1, type=int,
                        help='number of query data in each class, same as n_query')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')
    parser.add_argument('--save_freq', default=20,
                        type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0,
                        type=int, help='Starting epoch')
    parser.add_argument('--stop_epoch', default=-1,
                        type=int, help='Stopping epoch')
    parser.add_argument('--resume', action='store_true',
                        help='continue from previous trained model with largest epoch')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/pointnet2_11/ModelNet40_01/',
                        help='continue from previous trained model with largest epoch')
    #args = parser.parse_args(sys.argv)
    return parser.parse_args()
    

def get_old_tasks(loader,num_batches):
    old_tasks = []
    for sample in range(num_batches):
        input_batch, target_batch = next(iter(loader))
        for image in input_batch:
            old_tasks.append(image)
    return old_tasks

class EWC(object):
    
    def __init__(self, model: nn.Module, dataset: list, loader):

        self.model = model #pretrained model
        self.data = dataset #samples from the old task or tasks
        self.loader = loader
        self.dataset = []
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.means = {}
        for n, p in deepcopy(self.params).items():
            self.means[n] = p.data.cuda()
        
        self.precision_matrices = self.diag_fisher()
        
    def diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.cuda()

        self.model.eval()
        """
        for input in self.dataset:
            self.model.zero_grad()
            #input = input.squeeze(1)
            input = input.cuda()
            #output,_ = self.model.forward(input).view(1, -1)
            output, _ = self.model.parse_feature(input, is_feature = False)
            output = output.to(torch.float)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()
            """
        for sample in range(num_batches):
            for i in enumerate(self.loader):
                self.model.zero_grad()
                self.model.train()
                self.model.train_loop(sample,self.loader,optimizer = torch.optim.Adam(model.parameters()))
                #image, label = self.loader.__getitem__(i)
                #self.dataset.append(image)
        
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

if __name__ == "__main__":
    #old task load 
    np.random.seed(10)  
    base_file = 'hippo.json'    
    params = parse_args()
    base_datamgr = SetDataManager(params.train_n_way,params.n_shot, params.n_query, 100)
    base_loader = base_datamgr.get_data_loader(base_file, None)

    num_batches = 10
    old_tasks = get_old_tasks(base_loader, num_batches)
    print("length old_tasks = ", len(old_tasks))
    feature = VNet(in_channels=1,classes= params.train_n_way)
    model = ProtoNet(params.train_n_way, params.n_shot, feature)
    tmp = torch.load(params.checkpoint_dir+'best_model.tar')
    state = tmp['state']
    model.load_state_dict(state)
    model = model.cuda()
    """
    model.eval()

    for i in range(num_batches):
        model.zero_grad()
        model.train()
        model.train_loop(i,base_loader,optimizer = torch.optim.Adam(model.parameters()) )
    """
    
    ewc = EWC(model, old_tasks,base_loader)




