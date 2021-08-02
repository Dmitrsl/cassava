import torch
import cv2
import numpy as np


def get_img(path):
    im_bgr = cv2.imread(str(path))
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

class CassavaDataset(torch.utils.data.Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        #path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        # root / 'test_images/'

        path = self.data_root / self.df.iloc[index]['image_id']
        img  = get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label == True:
            return img, torch.from_numpy(np.asarray(target)).long()
        else:
            return img



class CassavaTTADataset(torch.utils.data.Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        self.ttas = [self.transform, TTA1, TTA2, TTA3]
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        #path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        # root / 'test_images/'

        path = self.data_root / self.df.iloc[index]['image_id']
        img  = get_img(path)
     
        imglist = [tta(image=image)['image'] for tta in self.ttas]  # update

        img = torch.stack(imglist)
            
        # do label smoothing
        if self.output_label == True:
            return img, torch.from_numpy(np.asarray(target)).long()
        else:
            return img