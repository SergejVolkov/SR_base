import cv2
import numpy as np
import torch
import torch.tensor as Tensor
import torch.nn.functional as F

piece_count = 4


def imwrite(filename: str, image: Tensor):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    output = torch.clamp(image / 2 + 0.5, min=0, max=1)
    output = np.transpose(output.cpu().numpy(), (1, 2, 0)) * 255
    cv2.imwrite(filename, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def scale(image: Tensor, aspect_ratio: float=1.0,
          extra_scale: float=1.0, even_rounding: bool=False):
    if aspect_ratio != 1.0 or extra_scale != 1.0:
        unsq_dim = 0
        while len(image.shape) < 4:
            image = image.unsqueeze(0)
            unsq_dim += 1
        if even_rounding:
            size = even_round(image.shape[2] * extra_scale,
                              image.shape[3] * aspect_ratio * extra_scale)
        else:
            size = (int(round(image.shape[2] * extra_scale)),
                    int(round(image.shape[3] * aspect_ratio * extra_scale)))
        image = torch.clamp(F.interpolate(
            image, size=size, mode='bicubic', align_corners=True
        ), min=-1, max=1)
        while unsq_dim > 0:
            image = image.squeeze(0)
            unsq_dim -= 1
    return image


def even_round(*args):
    rounded = []
    for arg in args:
        rnd = int(arg)
        if rnd % 2 != 0:
            rnd += 1
        rounded.append(rnd)
    return rounded


def cut_image(image: Tensor) -> list:
    _, c, h, w = image.shape
    h //= piece_count
    w //= piece_count
    pieces = []
    for i in range(piece_count):
        for j in range(piece_count):
            pieces.append(image[:, :, i * h:(i + 1) * h,
                          j * w:(j + 1) * w])
    return pieces


def glue_image(pieces: list) -> Tensor:
    horiz = []
    for i in range(piece_count):
        horiz.append(torch.cat(pieces[i * piece_count:(i + 1) * piece_count], 3))
    image = torch.cat(horiz, 2)
    return image
