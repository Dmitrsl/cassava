from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomGamma,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, ISONoise,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize, JpegCompression, IAAPerspective,
    RandomSunFlare, RandomShadow, RandomResizedCrop, CoarseDropout, RandomGridShuffle
)
from albumentations.pytorch import ToTensorV2

mean=[0.485, 0.456, 0.406], 
std=[0.229, 0.224, 0.225]

def get_train_transforms(img_size = 528):
    return Compose([
            # CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(img_size, img_size),
            SmallestMaxSize(img_size,interpolation=2, p=1.),

            #CenterCrop(img_size, img_size, p=1.),
            RandomResizedCrop(img_size, img_size, interpolation=2),


            #RandomResizedCrop(img_size, img_size),
            Transpose(p=0.15),
            HorizontalFlip(p=0.4),
            VerticalFlip(p=0.2),
            ShiftScaleRotate(p=0.2),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.09),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.12),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # CoarseDropout(p=0.5),
            # Cutout(p=0.5),
            CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=2,
                               min_height=2, min_width=2,p=0.38),
            RandomGridShuffle(grid=(3, 3), p=0.24),  

            GridDistortion(p=0.235), # +
            GaussNoise(p=0.11), # +
            #JpegCompression(quality_lower=80, p=0.1),
            #IAAPerspective(scale=(0.02, 0.05), p=0.44), #add
            # ISONoise(p=0.25), # add
            MedianBlur(p=0.2), # +
            RandomGamma(gamma_limit=(85, 115), p=0.3), # +

            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms(img_size = 528):
    return Compose([
            SmallestMaxSize(img_size,interpolation=2, p=1.),

            CenterCrop(img_size, img_size, p=1.),
            # CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            # Resize(CFG['img_size'], CFG['img_size']),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),

            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms(img_size = 528):
    return Compose([
            RandomResizedCrop(img_size, img_size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


