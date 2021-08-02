'''
train
'''
from pathlib import Path
import os
import torch
import multiprocessing

from torch import nn
import pandas as pd
import collections
import catalyst

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback
from catalyst.dl.callbacks import MixupCallback, CutmixCallback
from catalyst.dl.callbacks import CriterionCallback, OptimizerCallback
from catalyst.contrib.nn import RAdam, Lookahead, AdamP, Lamb, QHAdamW, Ralamb

from pytorch_toolbelt import losses as L

from utils.prepare_data import get_loaders
from utils.models import CassavaNet, get_params
from utils.settings import seed_everything
from utils.schedulers import CosineAnnealingWarmupRestarts
from utils.loss import TaylorCrossEntropyLoss

N_FOLDS = 7
ROOT = Path(os.getcwd())/ 'cassava-leaf-disease-classification'
OUTPUT_ROOT = ROOT / 'out'

SEED = 2021
seed_everything(SEED)

NUM_CORES = multiprocessing.cpu_count() - 2

BS = 6
LR = 1e-4
num_epochs = 20
img_size=528

def train_one_model(model_name='tf_efficientnet_b2_ns'):
    '''
    main
    '''
    for fold in range(1, N_FOLDS)[0:]:

        print(f'FOLD_{fold}')

        loaders = collections.OrderedDict()
        loaders["train"], loaders["valid"], _ = get_loaders(fold=fold, bs=BS, img_size=img_size, extra=False, balanser='dinamic')

        device = catalyst.utils.get_device()

        model = CassavaNet(5, model_name).to(device)
        log = f"{OUTPUT_ROOT}/.logs_{model_name}_{fold}/checkpoints"
        model_dict = torch.load(f'{log}/best.pth', map_location=device)['model_state_dict']
        model.load_state_dict(model_dict)
        param = get_params(model, lr=LR)



        # RAdam
        optimizer = Lookahead(AdamP(param))
        
        #optimizer = torch.optim.AdamW(param)
        #scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=5, cycle_mult=1.5, max_lr=0.001, min_lr=0.000001, warmup_steps=1, gamma=0.75)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, 
                                            pct_start=0.1, 
                                            #total_steps=10,
                                            steps_per_epoch=2, epochs=10,
                                            #base_momentum=0.83, max_momentum=0.98
                                            )
        # criterion = L.JointLoss(L.FocalLoss(), L.SoftCrossEntropyLoss(reduction="mean", smooth_factor=0.1), 0.2,  0.8)
        # criterion = L.JointLoss(L.FocalLoss(), nn.CrossEntropyLoss(), 0.2, 0.8)
        # criterion = nn.CrossEntropyLoss()
        criterion =  L.JointLoss(L.FocalLoss(), TaylorCrossEntropyLoss(n=2, smoothing=0.05), 0.2,  0.8)

        #criterion = L.SoftCrossEntropyLoss(reduction="mean", smooth_factor=0.1).to(device)
        #criterion = nn.CrossEntropyLoss()# .to(device)

        logdir = f"{OUTPUT_ROOT}/.logs_{model_name}_{fold}_stage_2_1"

        runner = SupervisedRunner(device=device, model=model)

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=[AccuracyCallback(),
                        OptimizerCallback(accumulation_steps=10),
                        ],
            logdir=logdir,
            num_epochs=num_epochs,
            main_metric= "accuracy01",
            minimize_metric=False,
            fp16=True,
            verbose=True,
            load_best_on_end=True,)

        

        model = CassavaNet(5, model_name).to(device)
        log = f"{logdir}/checkpoints/best.pth"
        model_dict = torch.load(log, map_location=device)['model_state_dict']
        model.load_state_dict(model_dict)

        logdir = f"{OUTPUT_ROOT}/.logs_{model_name}_{fold}_stage_2_2"

        for param in model.backbone.parameters():
            param.requires_grad = False

        for param in model.backbone.classifier.parameters():
            param.requires_grad = True

        
        criterion =  L.JointLoss(L.FocalLoss(), TaylorCrossEntropyLoss(n=2, smoothing=0.05), 0.4,  0.6)

        optimizer = Lookahead(AdamP(model.parameters(), lr=LR, weight_decay=1e-6))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, 
                                            pct_start=0.1, 
                                            #total_steps=10,
                                            steps_per_epoch=3, epochs=10,
                                            #base_momentum=0.83, max_momentum=0.98
                                            )
        
        runner = SupervisedRunner(device=device, model=model)

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=[
                AccuracyCallback(),
                OptimizerCallback(accumulation_steps=8),
                ],
            logdir=logdir,
            num_epochs=30,
            main_metric= "accuracy01",
            minimize_metric=False,
            fp16=True,
            verbose=True,
            load_best_on_end=True,)

        batch = next(iter(loaders["valid"]))
        # # saves to `logdir` and returns a `ScriptModule` class
        runner.trace(model=model, batch=batch, logdir=logdir, fp16=True)
 


def main():
    train_one_model()

if __name__ == '__main__':
    main()
    