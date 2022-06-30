#from pyexpat import model
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse

from data.datamgr import SetDataManager
from methods.protonet import ProtoNet 

#from models.pointnet import PointNetEncoder
#from models.dgcnn import DGCNN_Cls_Encoder
from models.vnet import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_n_way', default=2, type=int,
                        help='class num to classify for training')
    parser.add_argument('--test_n_way', default=1, type=int,
                        help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=1, type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')
    parser.add_argument('--n_query', default=1, type=int,
                        help='number of query data in each class, same as n_query')
    parser.add_argument('--save_freq', default=20,
                        type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0,
                        type=int, help='Starting epoch')
    parser.add_argument('--stop_epoch', default=-1,
                        type=int, help='Stopping epoch')
    parser.add_argument('--resume', action='store_true',
                        help='continue from previous trained model with largest epoch')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/3d_model/best_model.tar',
                        help='continue from previous trained model with largest epoch')

    return parser.parse_args()


def test(test_loader, model, params):
    max_acc = 0
    model.eval()
    acc = model.test_loop(test_loader)

if __name__ == '__main__':
    np.random.seed(10)  #random seed
    test_file = '/home/pkpc/PROJECT/3d_protonet_working/data/taskB_test.json'
    params = parse_args()

    #n_query = max(1, int(16* params.test_n_way / params.train_n_way))

    datamgr = SetDataManager(params.test_n_way,params.n_shot, params.n_query, 100)
    dataloader = datamgr.get_data_loader(test_file, None)
    x = dataloader.dataset.__getitem__(0)
    print(x[1][0].shape)
    #feature = PointNetEncoder(channel=6)
    
    feature = VNet(in_channels=1,classes= params.test_n_way)
    model = ProtoNet(params.test_n_way, params.n_shot, feature)
    model = model.cuda()
    
    tmp = torch.load(params.checkpoint_dir)
    state = tmp['state']
    model.load_state_dict(state)
    
    model = test(dataloader, model, params)
    


