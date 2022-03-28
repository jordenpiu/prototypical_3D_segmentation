#define prototypical operation on few-shots 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import *
from vnet import *
import matplotlib.pyplot as plt

#supporti, supportl,queryi,queryl, chosen_classes= get_data(data_dir, n_ways, n_shot,n_query, mode = "train")
supporti = torch.randn(5, 1, 64, 64, 64) #support image
supportl = torch.randn(5, 1, 64, 64, 64) #support label
queryi = torch.randn(5, 1, 64, 64, 64)   #query image
queryl = torch.randn(5, 1, 64, 64, 64)   #query label

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def getFeatures( fts, mask):
    """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
    """
    #print(fts.shape)
    #print(mask.shape)
    fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
    masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
        / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
    return masked_fts

def getPrototype(fg_fts):
    """
    Average the features to obtain the prototype

    Args:
        fg_fts: lists of list of foreground features for each way/shot
            expect shape: Wa x Sh x [1 x C]
        bg_fts: lists of list of background features for each way/shot
            expect shape: Wa x Sh x [1 x C]
    """
    n_ways, n_shots = len(fg_fts), len(fg_fts[0])
    fg_prototypes = [sum(way) / n_shots for way in fg_fts]
    #bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
    return fg_prototypes #, bg_prototype

def calDist(fts, prototype, scaler=20):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x H x W
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
    return dist

def forward( supp_imgs, supp_masks, qry_masks, qry_imgs):
    n_ways = len(supp_imgs)
    n_shots = len(supp_imgs[0])
    n_queries = len(qry_imgs[0])
    batch_size =  n_shots #+ n_queries
    img_size = supp_imgs[0][0].shape

    print("n_ways=", n_ways)
    print("n_shots=", n_shots)
    print("n_queries=", n_queries)
    print("batch_size=", batch_size)
    print("img_size=", img_size)
    
    supp_masks = supp_masks
    qry_masks = qry_masks

    ###### Extract features ######

    x = torch.cat([supp_imgs.view(n_ways * n_shots,1, *supp_imgs.size()[2:]),
                    qry_imgs.view(n_ways * n_queries,1,*qry_imgs.size()[2:])], 0)
    #print(x.shape)  #10, 1, 64, 64, 64      
    
    encoder =  VNet(batch=n_ways, in_channels=1)

    img_fts = encoder(x)  
    #print(img_fts.shape) #10, 5, 64, 64, 64
        
    fts_size = img_fts.shape[-3:]  
    #print(fts_size) #64, 64, 64

    supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
        n_ways*n_shots,batch_size,-1,*fts_size)  # Wa x Sh x B x C x H' x W'
    #print(supp_fts.shape) #5, 1, 5, 64, 64, 64
    #print(supp_fts[2].shape) # 1,5,64,64,64

    qry_fts = img_fts[n_ways * n_shots* batch_size:].view(
        n_ways*n_queries,batch_size,-1,*fts_size)   # N x B x C x H' x W'
    #print(qry_fts.shape) #5, 1, 5, 64, 64, 64
    #print(qry_fts[3].shape)  #1, 5, 64, 64, 64

    outputs = []
    for epi in range(batch_size):
        ######extract prototypes#######
        supp_fg_fts = [[getFeatures(supp_fts[way, shot, [epi]],supp_masks[way, shot])for shot in range(n_shots)] for way in range(n_ways)]
        #print(supp_fg_fts[0][0].shape)  #[1, 64]

        ###obtain the prototypes####
        supp_prototypes = getPrototype(supp_fg_fts)
        #print(supp_prototypes[0][0].shape) #65

        ###compute the distance ######
        dist = [calDist(qry_fts[3][:epi], prototype) for prototype in supp_prototypes]
        #print(len(dist))
        pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
        print(pred.shape)  #0, 5, 5, 64, 64, 64
        outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

forward(supporti, supportl,queryl,queryi)