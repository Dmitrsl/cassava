# from collections import deque
import yaml
import os
import gc
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from collections import OrderedDict
import multiprocessing
import torch
from catalyst.data import DynamicBalanceClassSampler, BalanceClassSampler
try:
    from dataset import CassavaDataset
except:
    from utils.dataset import CassavaDataset

try:
    from transforms import get_train_transforms, get_valid_transforms, get_inference_transforms
except:
    from utils.transforms import get_train_transforms, get_valid_transforms, get_inference_transforms

ROOT = Path(os.getcwd()) / 'cassava-leaf-disease-classification'
NUM_CORES = multiprocessing.cpu_count()

def imbalance_sampler(labels):
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(labels))
    return sampler

def get_loaders(fold=2, bs=16, img_size=528, extra=False, balanser='simple'):

    train = pd.read_csv(ROOT / 'train_cv7_add.csv')

    train_fold = train[(train[f"fold_{fold}"] == 'train') & (train.source != 'extra')]
    valid_fold = train[train[f"fold_{fold}"] == 'test']
    extra_fold = train[train[f"fold_{fold}"] == 'train']

    train_ds = CassavaDataset(train_fold, ROOT / 'train/', transforms=get_train_transforms(img_size = img_size))
    valid_ds = CassavaDataset(valid_fold, ROOT / 'train/', transforms=get_valid_transforms(img_size = img_size))
    extra_ds = CassavaDataset(extra_fold, ROOT / 'train/', transforms=get_inference_transforms(img_size = img_size))

    if balanser == 'dinamic':
        sampler = DynamicBalanceClassSampler(torch.from_numpy(train_fold.label.values).long(), exp_lambda=0.9, max_d=5,)
    if balanser == 'simple':
        sampler = BalanceClassSampler(torch.from_numpy(train_fold.label.values).long())
        #sampler = imbalance_sampler(torch.from_numpy(train_fold.label.values)),

    train_loader = torch.utils.data.DataLoader(
                train_ds, 
                batch_size=bs,
                num_workers=NUM_CORES,
                shuffle=True,
                #sampler = sampler, #imbalance_sampler(torch.from_numpy(train_fold.label.values)),
                pin_memory=False,
            )
      
    valid_loader = torch.utils.data.DataLoader(
                valid_ds, 
                batch_size=bs,
                num_workers=NUM_CORES,
                shuffle=False,
                pin_memory=False,

            ) 

    extra_loader = torch.utils.data.DataLoader(
                extra_ds, 
                batch_size=bs,
                num_workers=NUM_CORES,
                shuffle=False,
                # sampler = sampler, #imbalance_sampler(torch.from_numpy(train_fold.label.values)),
                pin_memory=False,
            )

    return train_loader, valid_loader, extra_loader



if __name__ == "__main__":

    print(next(iter(get_loaders(0)[2]))[1])

    