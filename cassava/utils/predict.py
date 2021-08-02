import torch
from tqdm.auto import tqdm 
import numpy as np
import os
import re
from scipy.stats import gmean, hmean
from pathlib import Path


from utils.models import CassavaNet
from utils.settings import seed_everything
from utils.dataset import CassavaDataset
from utils.transforms import get_train_transforms, get_valid_transforms, get_inference_transforms


def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    labels = []
    
    #pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in enumerate(data_loader):

        img = imgs[0].to(device).float()

        labels_batch = imgs[1]

        image_preds = model(img)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        labels += [labels_batch]
        
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    labels = np.concatenate(labels, axis=0)

    return image_preds_all, labels


def tta_predict(n_tta, model, infer_loader, valid_loader, device, func='gmean'):
    
    tta_preds = []
    model.eval()

    with torch.no_grad():

        no_tta_preds, no_tta_labels = inference_one_epoch(model, valid_loader, device)

        for _ in range(n_tta):
            tta, _ = inference_one_epoch(model, infer_loader, device)
            tta_preds += [tta]
            tta_preds += [no_tta_preds]

    if func == 'gmean':
        tta_preds = gmean(tta_preds, axis=0) 
    else:
        tta_preds = np.mean(tta_preds, axis=0) 
    return tta_preds


ROOT = Path(os.getcwd())/ 'cassava-leaf-disease-classification'
# img_size=528
# BS = 16
# NUM_CORES = multiprocessing.cpu_count() - 2

def get_features(fold, models_list, cat_models_list=None, n_tta=4, func='gmean', BS=16,
                 NUM_CORES=4, img_size=528, device=None):
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    valid_ds = CassavaDataset(fold, ROOT / 'train/', transforms=get_valid_transforms(img_size=img_size))
    infer_ds = CassavaDataset(fold, ROOT / 'train/', transforms=get_inference_transforms(img_size=img_size))

    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BS, num_workers=NUM_CORES, shuffle=False, pin_memory=False,) 
    infer_loader = torch.utils.data.DataLoader(infer_ds, batch_size=BS, num_workers=NUM_CORES, shuffle=False, pin_memory=False,) 

    preds = []
    for model in tqdm(models_list):
        model_name = model[:re.search(r'\_\d\_', model).start()]
        model_net = CassavaNet(5, model_name).to(device)
        model_dict = torch.load(f'final_models/{model}', map_location=device)['model_state_dict']
        model_net.load_state_dict(model_dict)
        tta_preds = tta_predict(n_tta, model_net, infer_loader, valid_loader, device, func=func)
        preds += [tta_preds]
        
    return gmean(preds, axis=0)
        
