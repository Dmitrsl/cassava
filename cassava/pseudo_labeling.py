import os
import torch
import numpy as np
import torch
import multiprocessing

from utils.models import CassavaNet
from utils.dataset import CassavaTTADataset

NUM_CORES = multiprocessing.cpu_count() - 2

BS =  6

transforms_test = A.Compose([
    A.CenterCrop(img_size,img_size,p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

TTA1 = A.Compose([
    A.CenterCrop(img_size,img_size,p=1),
    A.HorizontalFlip(p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

TTA2 = A.Compose([
    A.CenterCrop(img_size,img_size,p=1),
    A.VerticalFlip(p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

TTA3 = A.Compose([
    A.CenterCrop(img_size,img_size,p=1),
    A.Transpose(p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

extra_ds = CassavaDataset(, ROOT / 'train/', transforms_test)

extra_loader = torch.utils.data.DataLoader(
            extra_ds, 
            batch_size=bs,
            num_workers=NUM_CORES,
            shuffle=False,
            sampler = sampler, #imbalance_sampler(torch.from_numpy(train_fold.label.values)),
            pin_memory=False,
        )

model = CassavaNet(5, model_name).to(device)
log = f"{OUTPUT_ROOT}/.logs_{model_name}_{fold}_stage_2_2/checkpoints/best.pth"
model_dict = torch.load(log, map_location=device)['model_state_dict']
model.load_state_dict(model_dict)

for images,image_names in tqdm(loader):
        images = images.cuda()
        with torch.no_grad():
            batch_size, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            output= F.softmax(model(images),dim=1)
            output = output.view(batch_size, n_crops,-1).mean(1)
            pred = output.argmax(1).cpu().numpy()
