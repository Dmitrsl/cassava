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

N_FOLDS = 7
ROOT = Path(os.getcwd())/ 'cassava-leaf-disease-classification'
OUTPUT_ROOT = ROOT / 'out'

SEED = 2021
seed_everything(SEED)

NUM_CORES = multiprocessing.cpu_count() - 2

BS = 2
LR = 1e-3
num_epochs = 30
img_size = 512

def train_one_model(model_name='tf_efficientnet_b5_ap'):
    '''
    main
    '''
    for fold in range(N_FOLDS)[:]:

        print(f'FOLD_{fold}')

        loaders = collections.OrderedDict()
        loaders["train"], loaders["valid"], _ = get_loaders(fold=fold, bs=BS, img_size=img_size, extra=False, balanser='simple')

        device = catalyst.utils.get_device()

        model = CassavaNet(5, model_name).to(device)
        param = get_params(model, lr=LR)

        # RAdam
        optimizer = Lookahead(torch.optim.AdamW(param))
        optimizer = torch.optim.AdamW(param)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
        #criterion = L.JointLoss(L.FocalLoss(), L.SoftCrossEntropyLoss(reduction="mean", smooth_factor=0.1), 0.2,  0.8)
        #criterion = L.SoftCrossEntropyLoss(reduction="mean", smooth_factor=0.1).to(device)
        criterion = nn.CrossEntropyLoss() # .to(device)


        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Tmax=10)

        logdir = f"{OUTPUT_ROOT}/.logs_{model_name}_{fold}"
        
        runner = SupervisedRunner(device=device, model=model)

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=[
                    #MixupCallback(alpha=1.0),
                    #CutmixCallback(alpha=0.1),
                    
                    AccuracyCallback(),
                # F1ScoreCallback(
                #     input_key="targets_one_hot",
                #     activation="Softmax")
                OptimizerCallback(accumulation_steps=32),
                ],
            logdir=logdir,
            num_epochs=num_epochs,
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
    