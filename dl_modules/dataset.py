import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.tensor as Tensor
import dl_modules.transforms as trf
import dl_modules.realsr as realsr
import cm_modules.utils as utils
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import Subset

# import torch.nn.functional as F
# from cm_modules.utils import imwrite


def imshow(img: Tensor) -> None:
    img = torch.clamp(img / 2 + 0.5, 0, 1)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Dataset(BaseDataset):
    """Images Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        scale (int): downscaling parameter
        normalization (torchvision.transforms.transform): image normalization
        transform (torchvision.transforms.transform): image transform (typically crop)
        augmentation (albumentations.Compose): data transfromation
        downscaling (str): downscaling method (possible 'bicubic', 'kernel', 'none')
        aspect_ratio (float): change pixel aspect ratio of lr image to width / heigth
        extra_scale (float): additional lr scaling for non-integer SR upscaling

    """

    def __init__(
            self,
            images_dir,
            scale,
            normalization=None,
            transform=None,
            augmentation=None,
            downscaling='bicubic',
            aspect_ratio=1.0,
            extra_scale=1.0
    ):
        self.ids = [name for name in os.listdir(images_dir) if
                    name.lower().endswith('.png') or
                    name.lower().endswith('.jpg') or
                    name.lower().endswith('.jpeg') or
                    name.lower().endswith('.gif') or
                    name.lower().endswith('.bmp')]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.transform = transform
        self.augmentation = augmentation
        if normalization is None:
            self.normalization = get_normalization()
        else:
            self.normalization = normalization
        self.scale = scale
        self.downscaling = downscaling
        self.ar = aspect_ratio
        self.es = extra_scale

    def __getitem__(self, i):
        # read data
        gt = cv2.imread(self.images_fps[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            gt = self.transform(image=gt)["image"]

        in_image = gt

        if self.augmentation is not None:
            in_image = self.augmentation(image=in_image)["image"]

        in_image = self.normalization(in_image)
        gt = self.normalization(gt)

        if self.downscaling == 'bicubic':
            in_image = utils.scale(in_image, aspect_ratio=self.ar,
                                   extra_scale=self.es / self.scale)
        elif self.downscaling == 'kernel':
            in_image = utils.scale(in_image, aspect_ratio=self.ar,
                                   extra_scale=self.es)
            in_image = realsr.apply_kernel(in_image, kernel_storage)

        return in_image, gt

    def __len__(self):
        return len(self.ids)


class ValidDataset(BaseDataset):
    """Images Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        hr_dir (str): path to HR images folder
        lr_dir (str): path to LR images folder
        normalization (torchvision.transforms.transform): image normalization
        transform (torchvision.transforms.transform): ground truth transform

    """

    def __init__(
            self,
            hr_dir,
            lr_dir,
            normalization=None,
            transform=None
    ):
        self.ids = os.listdir(hr_dir)
        self.hr_fps = [os.path.join(hr_dir, image_id) for image_id in self.ids]
        self.lr_fps = [os.path.join(lr_dir, image_id) for image_id in self.ids]

        self.transform = transform
        if normalization is None:
            self.normalization = get_normalization()
        else:
            self.normalization = normalization

    def __getitem__(self, i):
        # read data
        gt = cv2.imread(self.hr_fps[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        in_image = cv2.imread(self.lr_fps[i])
        in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            gt = self.transform(image=gt)["image"]
        gt = self.normalization(gt)
        in_image = self.normalization(in_image)
        return in_image, gt

    def __len__(self):
        return len(self.ids)


def get_normalization() -> torch.nn.Module:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def init_data():
    global train_set, train_loader, valid_set, valid_loader, noise_set, noise_loader, kernel_storage
    train_set = Dataset(train_dir, scale=scale,
                        transform=trf.get_training_transform(crop_size),
                        augmentation=trf.get_input_image_augmentation(),
                        downscaling='kernel',
                        aspect_ratio=aspect_ratio,
                        extra_scale=extra_scale)
    if train_set_size != 0:
        train_set = Subset(train_set, list(range(train_set_size)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                               shuffle=True, num_workers=6)

    valid_set = ValidDataset(hr_dir=valid_hr_dir, lr_dir=valid_lr_dir)
    if valid_set_size != 0:
        valid_set = Subset(valid_set, list(range(valid_set_size)))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=valid_batch_size,
                                               shuffle=False, num_workers=0)

    noise_set = Dataset(noise_dir, scale=scale,
                        normalization=realsr.get_noise_normalization(),
                        transform=trf.get_training_transform(int(round(crop_size / scale * extra_scale * aspect_ratio)),
                                                             int(round(crop_size / scale * extra_scale))),
                        downscaling='none')
    noise_loader = torch.utils.data.DataLoader(noise_set, batch_size=train_batch_size,
                                               shuffle=True, num_workers=0)
    kernel_storage = realsr.Kernels(kernel_dir, scale=scale, count=realsr.kernel_count)

    # Look at images we have

    # not_trf_set = Dataset(train_dir, scale=scale,
    #                       augmentation=get_input_image_augmentation())
    #
    # image_in, image_out = not_trf_set[0]  # get some sample
    # imshow(image_in)
    # imshow(image_out)

    # Visualize augmented images

    # for i in range(3):
    #     image_in, image_out = train_set[i]
    #     image_in = realsr.inject_noise(image_in.unsqueeze(0), noise_loader)
    #     print(image_in.shape)
    #     imwrite(
    #         SAVE_DIR + 'data/output/%d_lr_scaled.png' % i,
    #         F.interpolate(image_in, scale_factor=scale, mode='bicubic')
    #     )
    #     imwrite(
    #         SAVE_DIR + 'data/output/%d_lr.png' % i,
    #         image_in
    #     )
    #     imwrite(
    #         SAVE_DIR + 'data/output/%d_hr.png' % i,
    #         image_out
    #     )


SAVE_DIR = ''  # ../drive/MyDrive/

train_dir = os.path.join(SAVE_DIR, 'data/Bakemonogatari/Bakemonogatari_train_HR')
valid_hr_dir = os.path.join(SAVE_DIR, 'data/Bakemonogatari/Bakemonogatari_valid_HR')
valid_lr_dir = os.path.join(SAVE_DIR, 'data/Bakemonogatari/Bakemonogatari_valid_LR')
kernel_dir = os.path.join(SAVE_DIR, 'data/SoulTaker/SoulTaker_train_kernel')
noise_dir  = os.path.join(SAVE_DIR, 'data/SoulTaker/SoulTaker_train_noise')

# Load datasets
train_batch_size = 128
valid_batch_size = 1  # Better leave it 1, otherwise many things won't work)

crop_size = 64        # Training crop HR size
scale = 2             # General SR upscaling parameter
extra_scale = 0.885   # Extra downscaling in training
aspect_ratio = 0.838  # Aspect ratio change (anamorphic encoding)

predict_res = (1920, 1080)  # Prediction resolution

train_set_size = 0
valid_set_size = 0

train_set = None
train_loader = None
valid_set = None
valid_loader = None
noise_set = None
noise_loader = None
kernel_storage = None
