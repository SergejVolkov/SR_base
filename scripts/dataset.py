import os
import numpy as np
import cv2
import torch
import albumentations as albu
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.tensor as Tensor

from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import Subset


def imshow(img: Tensor) -> None:
    img = torch.clamp(img / 2 + 0.5, 0, 1)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Dataset(BaseDataset):
    """DIV2K Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        scale (int): upscaling parameter
        augmentation (albumentations.Compose): data transfromation

    """

    def __init__(
            self,
            images_dir,
            scale,
            augmentation=None,
            in_aug=None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.in_aug = in_aug
        self.normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.scale = scale

    def __getitem__(self, i):
        # read data
        gt = cv2.imread(self.images_fps[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        if self.augmentation is not None:
            gt = self.augmentation(image=gt)["image"]

        h, w, _ = gt.shape
        in_image = cv2.resize(gt, (w // self.scale, h // self.scale), interpolation=cv2.INTER_CUBIC)
        if self.in_aug is not None:
            in_image = self.in_aug(image=in_image)["image"]

        gt = self.normalization(gt)
        in_image = self.normalization(in_image)

        return in_image, gt

    def __len__(self):
        return len(self.ids)


def get_training_augmentation(crop_size: int):
    return albu.Compose([
        albu.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        albu.HorizontalFlip(p=0.5),
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.5,
        # ),

        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=7, p=1),
        #         albu.MotionBlur(blur_limit=7, p=1),
        #     ],
        #     p=0.5,
        # ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.5,
        # )
    ])


def get_validation_augmentation(crop_size: int):
    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_size, min_width=crop_size, always_apply=True),
        albu.CenterCrop(height=crop_size, width=crop_size, always_apply=True)
    ])


def get_input_image_augmentation(blur_limit: int):
    return albu.Blur(blur_limit=blur_limit, p=1)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = 'data/DIV2K/'

train_dir = os.path.join(DATA_DIR, 'DIV2K_train_HR')
valid_dir = os.path.join(DATA_DIR, 'DIV2K_valid_HR')
test_dir = os.path.join(DATA_DIR, 'DIV2K_test_HR')

# Load datasets
train_batch_size = 32
valid_batch_size = 1

crop_size = 64
scale = 4

valid_img_size = 1140
# blur_limit = 3

train_set = Dataset(train_dir, scale=scale,
                    augmentation=get_training_augmentation(crop_size))
# train_set = Subset(train_set, list(range(128)))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                           shuffle=True, num_workers=12)

valid_set = Dataset(valid_dir, scale=scale,
                    augmentation=get_validation_augmentation(valid_img_size))
valid_set = Subset(valid_set, list(range(10)))
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=valid_batch_size,
                                           shuffle=False, num_workers=0)


# Look at images we have

# not_aug_set = Dataset(train_dir, scale=scale,
#                       in_aug=get_input_image_augmentation())
#
# image_in, image_out = not_aug_set[0]  # get some sample
# imshow(image_in)
# imshow(image_out)

# Visualize augmented images

# for i in range(3):
#     image_in, image_out = train_set[0]
#     imshow(image_in)
#     imshow(image_out)
