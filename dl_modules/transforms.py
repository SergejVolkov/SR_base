import cv2
import albumentations as albu
# from albumentations.augmentations.transforms import ImageCompression


def get_training_transform(crop_size: int):
    return albu.Compose([
        albu.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=(-0.25, -0.25),
                                          contrast_limit=(0.3, 0.3)),
            albu.RandomBrightnessContrast(brightness_limit=(0.25, 0.25),
                                          contrast_limit=(-0.3, -0.3)),
        ], p=0.7),
        albu.HorizontalFlip(p=0.5)
    ])


def get_training_noise_transform(crop_width: int, crop_height: int):
    return albu.Compose([
        albu.RandomCrop(height=crop_height, width=crop_width, always_apply=True),
        albu.Flip(p=0.5)
    ])


def get_generate_noise_transform(width: int, height: int):
    return albu.Compose([
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True),
        albu.CenterCrop(height=height, width=width, always_apply=True)
    ])


def get_input_image_augmentation():
    return albu.Compose([
        albu.Downscale(scale_min=0.35, scale_max=0.45, interpolation=cv2.INTER_AREA, p=0.5)
    ])


def get_predict_transform(width: int, height: int):
    return albu.Compose([
        albu.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC, always_apply=True)
    ])

    # albu.Compose([
    #     albu.OneOf(
    #         [
    #             albu.Compose([
    #                 albu.Blur(blur_limit=3, p=1),
    #                 albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.9, 1.0), p=1)
    #             ], p=1),
    #             albu.Compose([
    #                 albu.GaussianBlur(blur_limit=(3, 3), p=1),
    #                 albu.IAASharpen(alpha=(0.2, 0.5), lightness=(0.9, 1.0), p=1)
    #             ], p=1),
    #             albu.Downscale(scale_min=0.4, scale_max=0.6, interpolation=cv2.INTER_AREA, p=1)
    #         ],
    #         p=0.5
    #     ),
    #     albu.OneOf(
    #         [
    #             albu.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255), p=1),
    #             albu.GaussNoise(var_limit=(5.0, 30.0), p=1),
    #             albu.ImageCompression(quality_lower=95,
    #                                   compression_type=ImageCompression.ImageCompressionType.JPEG, p=1)
    #         ],
    #         p=1
    #     )
    # ])
