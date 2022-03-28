import numpy as np 
import torch
import os 
import json
import random
import SimpleITK as sitk

data_dir  = './dataset' 
n_ways = 5
n_shot = 1
n_query = 1


def resize_image( image, img_size=(64, 64, 64), **kwargs):
    assert isinstance(image, (np.ndarray, np.generic))
    assert image.ndim - 1 == len(img_size) or image.ndim == len(
        img_size
    ), "Example size doesnt fit image size"

    rank = len(img_size)

    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.0))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    return np.pad(image[slicer], to_padding, **kwargs)


def get_data(data_dir,n_ways, n_shot,n_query,mode):
    data_dir = data_dir 
    n_ways = n_ways
    n_shot = n_shot 
    n_query = n_query
    classes = os.listdir(data_dir)
    classes_idx = list(range(0,len(classes)))
    chosen_classes = random.sample(classes_idx,n_ways)
    supporti = np.zeros((n_ways,n_shot,64,64,64), dtype=np.float32)
    supportl = np.zeros((n_ways,n_shot,64,64,64), dtype=np.float32)
    queryi = np.zeros((n_ways,n_query,64,64,64), dtype=np.float32)
    queryl = np.zeros((n_ways,n_query,64,64,64), dtype=np.float32)

    class_cnt = 0  

    for i in chosen_classes:
        imgnames = os.listdir(os.path.join(data_dir,classes[i],"labels"))
        indexs = list(range(0,len(imgnames)))
        chosen_index = random.sample(indexs,n_shot+n_query)
        j = 0
        for k in chosen_index:
            image_path = os.path.join(data_dir,classes[i],"images",imgnames[k])
            label_path = os.path.join(data_dir,classes[i], "labels",imgnames[k])
            image = sitk.ReadImage(image_path)
            image = sitk.GetArrayFromImage(image)
            label = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(label)
            image = resize_image(image, (64,64,64))
            label = resize_image(label, (64,64,64))
            #print(image.shape)
            #print(label.shape)
            if j < n_shot:
                supporti[j] = image 
                supportl[j][0] = label 
            else:
                queryi[j-n_shot] = image
                queryl[j-n_shot] = label
            j += 1
        class_cnt += 1

    #numpy to torch
    supporti = torch.from_numpy(supporti)
    supportl = torch.from_numpy(supportl)

    queryi = torch.from_numpy(queryi)
    queryl = torch.from_numpy(queryl)

                    
    return supporti, supportl,queryi,queryl ,chosen_classes




###################rough work #############
"""
supporti, supportl,queryi,queryl, chosen_classes= get_data(data_dir, n_ways, n_shot,n_query, mode = "train")
        
        x = torch.cat([supp_imgs.view(n_ways * n_shots,1, *supp_imgs.size()[2:]),
                        qry_imgs.view(n_ways * n_queries,1,*qry_imgs.size()[2:])], 0)
        print(x.shape)
        encoder =  VNet(batch=n_ways, in_channels=1)
        z= encoder(x)
        print(z.shape)
        z_dim = z.size(-1)
        print(z_dim)

        z_proto = z[:n_ways*n_shots].mean(1) #.view(n_ways, n_shots,z_dim,z_dim,z_dim).mean(1)
        print(z_proto.shape)  
        zq = z[n_ways*n_shots:]
        print(zq.shape)

        dists = euclidean_dist(zq, z_proto)
        print(dists)
        

print(supporti.shape)
print(supportl.shape)
print(queryi.shape)
print(queryl.shape)
print(chosen_classes)
"""
