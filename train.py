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
    parser.add_argument('--checkpoint_dir', default='./checkpoint/3d_model',
                        help='continue from previous trained model with largest epoch')
    #args = parser.parse_args(sys.argv)
    return parser.parse_args()

def train(base_loader, val_loader, model, optimizer, start_epoch, stop_epoch, params):
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimizer, please define the optimizer.')

    max_acc = 0

    for epoch in range(start_epoch, stop_epoch):
        model.train() #model is called by reference, no need of return
        model.train_loop(epoch, base_loader, optimizer)
        model.eval()
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader)
        with open ("VAL_LOGS.txt", "a+") as f:
            f.write('%d Test Acc = %f \n' %(epoch, acc))
        if acc > max_acc:  # for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir,
                                   '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == "__main__":
    np.random.seed(10)  #random seed
    # data lists
    base_file = './data/taskA_train.json'
    val_file = './data/taskB_val.json'
    optimization = 'Adam' #optimizer

    params = parse_args()
    params.stop_epoch = 100  # total epochs
    params.start_epoch = 0
    
    base_datamgr = SetDataManager(params.train_n_way,params.n_shot, params.n_query, 100)
    base_loader = base_datamgr.get_data_loader(base_file, None)
    val_datamgr = SetDataManager(params.test_n_way,params.n_shot, params.n_query, 100)
    val_loader = base_datamgr.get_data_loader(base_file, None)
    feature = VNet(in_channels=1,classes= params.train_n_way)
    model = ProtoNet(params.train_n_way, params.n_shot, feature) # model Prototypical Networks
    model = model.cuda()
    print('---Network architecture--- \n', model)
    
    # training part
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    
    model = train(base_loader, val_loader, model, optimization, params.start_epoch, params.stop_epoch, params)
    
