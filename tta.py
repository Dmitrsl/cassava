from utils.prepare_data import get_loaders
from utils.models import CassavaNet, get_params
from utils.settings import seed_everything
from utils.dataset import CassavaDataset
from utils.transforms import get_train_transforms, get_valid_transforms, get_inference_transforms


from tqdm.auto import tqdm 
from pathlib import Path
import os
import numpy as np 
import pandas as pd
import torch
import catalyst
import multiprocessing
import collections

n_tta = 3
model_name = 'tf_efficientnet_b2_ns'
fold = 1
ROOT = Path(os.getcwd())/ 'cassava-leaf-disease-classification'
OUTPUT_ROOT = ROOT / 'out'
SEED = 2021
seed_everything(SEED)
NUM_CORES = multiprocessing.cpu_count() - 2
BS = 8
img_size=528

train = pd.read_csv(ROOT / 'train_cv7_add.csv')
valid_fold = train[train[f"fold_{fold}"] == 'test']
valid_ds = CassavaDataset(valid_fold, ROOT / 'train/', transforms=get_valid_transforms(img_size = img_size))
infer_ds = CassavaDataset(valid_fold, ROOT / 'train/', transforms=get_inference_transforms(img_size = img_size))
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=bs, num_workers=NUM_CORES, shuffle=False, pin_memory=False,) 
infer_loader = torch.utils.data.DataLoader(infer_ds, batch_size=bs, num_workers=NUM_CORES, shuffle=False, pin_memory=False,) 



device = catalyst.utils.get_device()

model = CassavaNet(5, model_name).to(device)

logdir = f"{OUTPUT_ROOT}/.logs_{model_name}_{fold}_stage_2_1/checkpoints/"

model_dict = torch.load(f'{logdir}/best.pth', map_location=device)['model_state_dict']
model.load_state_dict(model_dict)

val_preds = []
tta_preds = []

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    labels = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:

        imgs = imgs[0].to(device).float()

        labels_batch = imgs[1]
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        labels += labels_batch
        
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    labels = np.concatenate(labels, axis=0)

    return image_preds_all, labels

model.eval()

with torch.no_grad():
    for _ in range(n_tta):
        tta_preds, tta_labels = inference_one_epoch(model, valid_loader, device)
        tta_preds += inference_one_epoch(model, infer_loader, device)

tta_preds = np.mean(tta_preds, axis=0) 
no_tta_preds, no_tta_labels = inference_one_epoch(model, valid_loader, device)