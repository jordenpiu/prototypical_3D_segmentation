#meta template 

import numpy as np
import torch.nn as nn
from abc import abstractmethod
from torch.autograd import Variable
import torch
class MetaTemplate(nn.Module):
    def __init__(self,n_way, n_support, feature):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1
        self.feature = feature

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way *(self.n_support + self.n_query), *x.size()[2:])
            x = x.transpose(2,1)
            x = x.to(torch.float)
            z_all = self.feature.forward(x)
            # embed()
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
            
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query
    


    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)


    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            #if self.change_way:
            #    self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.data.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
                                                                        i, len(train_loader), avg_loss / float(i + 1)))
                
                with open ("TRAIN_LOGS.txt", "a+") as f:
                    f.write('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}\n'.format(epoch,
                                                                        i, len(train_loader), avg_loss / float(i + 1)))
                

    def test_loop(self, test_loader, record=None):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            #if self.change_way:
            #    self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean
