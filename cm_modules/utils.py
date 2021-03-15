import cv2
import numpy as np
import torch
import torch.tensor as Tensor
import torch.nn.functional as F

piece_count = 2


def imwrite(filename: str, image: Tensor):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    output = torch.clamp(image / 2 + 0.5, min=0, max=1)
    output = np.transpose(output.cpu().numpy(), (1, 2, 0)) * 255
    cv2.imwrite(filename, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def scale(image: Tensor, aspect_ratio: float=1.0, extra_scale: float=1.0):
    if aspect_ratio != 1.0 or extra_scale != 1.0:
        unsq_dim = 0
        while len(image.shape) < 4:
            image = image.unsqueeze(0)
            unsq_dim += 1
        image = torch.clamp(F.interpolate(
            image, scale_factor=(extra_scale, aspect_ratio * extra_scale),
            mode='bicubic', align_corners=True, recompute_scale_factor=False
        ), min=-1, max=1)
        while unsq_dim > 0:
            image = image.squeeze(0)
            unsq_dim -= 1
    return image


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
    # Temporary code
    horiz_1 = torch.cat((pieces[0], pieces[1]), 3)
    horiz_2 = torch.cat((pieces[2], pieces[3]), 3)
    image = torch.cat((horiz_1, horiz_2), 2)

    return image
