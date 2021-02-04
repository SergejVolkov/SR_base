import sys
import pyprind
import torch
import scripts.dataset as ds
import scripts.algorithm as algorithm
import torch.nn.functional as F


images_to_save = 3


def valid(net: torch.nn.Module, save_images=False,
          title="Valid") -> (int, float, list):
    criterion = algorithm.get_loss()
    metric = algorithm.get_metric()
    average_loss = 0.0
    accuracy = 0.0
    total = len(ds.valid_loader)

    iter_bar = pyprind.ProgBar(total, title=title, stream=sys.stdout)
    images = []

    with torch.no_grad():
        for data in ds.valid_loader:
            inputs, gt = data
            inputs = inputs.cuda()
            gt = gt.cuda()
            outputs = net(inputs)
            if save_images and len(images) < images_to_save:
                images.append(
                    torch.clamp(outputs.squeeze(0) / 2 + 0.5, min=0, max=1)
                )
            average_loss += criterion(outputs, gt).item()
            accuracy += metric(outputs, gt).item()
            iter_bar.update()
    iter_bar.update()
    return accuracy / total, average_loss / total, images


def get_static_images() -> list:
    images = []

    for data in ds.valid_loader:
        inputs, gt = data
        # Add LR sample
        images.append(
            torch.clamp(F.interpolate(
                inputs, scale_factor=(2, 2), mode='bicubic'
            ).squeeze(0) / 2 + 0.5, min=0, max=1)
        )
        # Add HR sample
        images.append(gt)
        if len(images) >= images_to_save * 2:
            break

    return images
