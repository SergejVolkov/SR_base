import sys
import pyprind
import torch
import training.dataset as ds
import training.algorithm as algorithm
import torch.nn.functional as F
import torch.tensor as Tensor


def valid(net: torch.nn.Module) -> (int, float, Tensor):
    criterion = algorithm.get_loss()
    metric = algorithm.get_metric()
    average_loss = 0.0
    accuracy = 0.0
    total = len(ds.valid_loader)

    iter_bar = pyprind.ProgBar(total, title="Valid", stream=sys.stdout)
    pred = None

    with torch.no_grad():
        for data in ds.valid_loader:
            images, gt = data
            images = images.cuda()
            gt = gt.cuda()
            outputs = net(images)
            if pred is None:
                pred = torch.clamp(torch.cat(
                    (F.interpolate(images, scale_factor=(2, 2), mode='bicubic'),
                     outputs, gt), 3
                ).squeeze(0) / 2 + 0.5, min=0, max=1)
            average_loss += criterion(outputs, gt).item()
            accuracy += metric(outputs, gt).item()
            iter_bar.update()
    iter_bar.update()
    return accuracy / total, average_loss / total, pred
