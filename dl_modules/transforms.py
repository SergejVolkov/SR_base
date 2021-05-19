import os
import cv2
import numpy as np
import albumentations as albu
# import albumentations.augmentations.functional as F
from PIL import Image


def get_training_transform(crop_size: int, kernel_size: int, bg_prob: float):
    return albu.Compose([
        RandomEdgeCrop(height=crop_size, width=crop_size, kernel_size=kernel_size, bg_prob=bg_prob, always_apply=True),
        # albu.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
        # albu.OneOf([
        #     albu.RandomBrightnessContrast(brightness_limit=(-0.15, -0.15),
        #                                   contrast_limit=(0.2, 0.2)),
        #     albu.RandomBrightnessContrast(brightness_limit=(0.12, 0.12),
        #                                   contrast_limit=(-0.17, -0.17)),
        # ], p=0.66),
        albu.RandomRotate90(p=0.5),
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
        albu.ImageCompression(quality_lower=70, quality_upper=90, p=0.5)
        # albu.Downscale(scale_min=0.35, scale_max=0.45, interpolation=cv2.INTER_AREA, p=0.5)
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


class RandomEdgeCrop(albu.DualTransform):
    def __init__(self, height, width, kernel_size, bg_prob=0.01, always_apply=False, p=1.0):
        super(RandomEdgeCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.sobel_x = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=float)
        self.sobel_y = np.transpose(self.sobel_x, (1, 0))
        self.kernel_size = kernel_size
        self.default_prob = bg_prob
        self.tmp_dir = './tmp/'
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        # self.kernel = np.ones((self.kernel_size, self.kernel_size), np.float32) / (self.kernel_size ** 2)

    def apply(self, img, uid=None, **params):
        prob_map = None
        if uid is not None:
            filename = self.tmp_dir + uid[:-4] + '.png'
            if os.path.isfile(filename):
                prob_map = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0 + self.default_prob
        if prob_map is None:
            flt_img = img.astype(np.float32)
            edges = (cv2.filter2D(flt_img, -1, self.sobel_x) ** 2 +
                     cv2.filter2D(flt_img, -1, self.sobel_y) ** 2) ** (1 / 2.0)
            prob_map = cv2.GaussianBlur(edges, (self.kernel_size, self.kernel_size), 0)
            prob_map = cv2.cvtColor(prob_map, cv2.COLOR_RGB2GRAY)
            s_h, s_w, c_h, c_w = ccrop_params(self.height, self.width)
            prob_map = prob_map[c_h:prob_map.shape[0] - s_h, c_w:prob_map.shape[1] - s_w]
            prob_map *= 255.0 / prob_map.max()
            if uid is not None:
                pimage = Image.fromarray(np.round(prob_map).astype(np.uint8), mode='L')
                pimage.save(self.tmp_dir + uid[:-4] + '.png', 'PNG')
                # cv2.imwrite(self.tmp_dir + uid[:-4] + '.png', np.round(prob_map).astype(np.uint8))
        return prob_crop(img, prob_map, self.height, self.width)

    def get_params_dependent_on_targets(self, params):
        uid = None
        if 'uid' in params:
            uid = params['uid']
        return {"uid": uid}

    @property
    def targets_as_params(self):
        return ["image", "uid"]

    def get_transform_init_args_names(self):
        return "height", "width"


def ccrop_params(height: int, width: int) -> tuple:
    s_h = height // 2 + 1
    s_w = width // 2 + 1
    c_h = (height - 1) // 2
    c_w = (width - 1) // 2
    return s_h, s_w, c_h, c_w


def prob_crop(img, prob_map, height, width):
    prob_map /= np.sum(prob_map)

    p_h = prob_map.shape[0]
    p_w = prob_map.shape[1]

    coords = np.arange(0, p_w * p_h)
    prob_map = prob_map.flatten()
    coord = np.random.choice(coords, p=prob_map)
    x = coord % p_w
    y = coord // p_w
    img = img[y:y + height, x:x + width, :]
    return img
