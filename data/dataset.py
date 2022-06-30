import os  
import numpy as np
import torch 
import json 
import SimpleITK as sitk

def resize_image( image, img_size=(32, 32, 32), **kwargs):
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




class SetDataset:
    def __init__(self, data_file, batch_size):
        with open(data_file, 'r') as f:
            self.sub_meta = json.load(f)
            #self.sub_meta[1][1][0]
         
        self.cl_list = self.sub_meta.keys()
        
        shuffle = True 
        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size  = batch_size, 
            shuffle = shuffle,
            num_workers = 0, 
            pin_memory = False
        )
        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl):
        self.sub_meta = sub_meta
        self.cl = cl
            
    def __getitem__(self, i):
        #print( '%d -%d' %(self.cl,i))
        image_path = self.sub_meta[i][0]
        label_path = self.sub_meta[i][1]
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)
        image = resize_image(image, (32,32,32))
        label = resize_image(label, (32,32,32))
        image = image.astype(int)
        label = label.astype(int)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        image = torch.unsqueeze(image, dim= 1)
        label = torch.unsqueeze(label, dim= 1)
        #print(image.shape)
        #print(label.shape)
        return image, label

    def __len__(self):
        return len(self.sub_meta)    
        
#SetDataset(sub_meta,10)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
            
