import cv2
import albumentations as albu
from albumentations.augmentations.transforms import ImageCompression


def get_training_augmentation(crop_size: int):
    return albu.Compose([
        albu.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        albu.HorizontalFlip(p=0.5)
    ])


def get_validation_augmentation(crop_size: int):
    return albu.Compose([
        albu.PadIfNeeded(min_height=crop_size, min_width=crop_size, always_apply=True),
        albu.CenterCrop(height=crop_size, width=crop_size, always_apply=True)
    ])


def get_input_image_augmentation():
    return albu.Compose([
        albu.OneOf(
            [
                albu.Compose([
                    albu.Blur(blur_limit=3, p=1),
                    albu.IAASharpen(alpha=(0.2, 0.4), lightness=(0.9, 1.0), p=1)
                ], p=1),
                albu.Compose([
                    albu.GaussianBlur(blur_limit=(3, 3), p=1),
                    albu.IAASharpen(alpha=(0.2, 0.4), lightness=(0.9, 1.0), p=1)
                ], p=1),
                albu.Downscale(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_LINEAR, p=1)
            ],
            p=0.75
        ),
        albu.OneOf(
            [
                albu.IAAAdditiveGaussianNoise(scale=(0.005 * 255, 0.005 * 255), p=1),
                albu.GaussNoise(var_limit=(5.0, 20.0), p=1),
                albu.ImageCompression(quality_lower=98,
                                      compression_type=ImageCompression.ImageCompressionType.JPEG, p=1),
                albu.ImageCompression(quality_lower=98,
                                      compression_type=ImageCompression.ImageCompressionType.WEBP, p=1)
            ],
            p=1
        )
    ])
