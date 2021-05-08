import sys
import pyprind
import numpy as np
import cv2
import torch
import torch.tensor as Tensor
import torch.nn.functional as F
import dl_modules.dataset as ds
import cm_modules.utils as utils
from cm_modules.utils import imwrite, convert_to_cv_float
from cm_modules.enhance import correct_colors


def predict(net: torch.nn.Module, device: torch.device,
            cut: bool=False, normalize: bool=False, ensemble: bool=False) -> None:
    net.eval()
    total = len(ds.predict_loader)
    iter_bar = pyprind.ProgBar(total, title="Predict", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in ds.predict_loader:
            downscaled, source = data
            source = source.to(device)
            output = None
            if ensemble:
                for _ in range(2):
                    for j in range(4):
                        if output is None:
                            output = process(net, source, cut)
                        else:
                            output += torch.rot90(
                                process(net, source, cut),
                                -j, (2, 3)
                            )
                        source = torch.rot90(source, 1, (2, 3))
                    source = torch.flip(source, (3, ))
                    output = torch.flip(output, (3, ))
                output /= 8.0
            else:
                output = process(net, source, cut)

            if normalize:
                path = ds.SAVE_DIR + 'data/output/' + ds.predict_set.ids[i][:-4] + '_sr_n.png'
                output = correct_colors(output, source)
            else:
                path = ds.SAVE_DIR + 'data/output/' + ds.predict_set.ids[i][:-4] + '_sr.png'
            imwrite(path, output)

            iter_bar.update()
            i += 1


def process(net: torch.nn.Module, source: Tensor, cut: bool) -> Tensor:
    if cut:
        pieces = utils.cut_image(source)
        out_pieces = []
        for piece in pieces:
            out_pieces.append(net(piece))
        output = utils.glue_image(out_pieces)
    else:
        output = net(source)
    return output


def predict_tb(net: torch.nn.Module, device: torch.device, count: int) -> list:
    net.eval()
    i = 0
    results = []

    with torch.no_grad():
        for data in ds.predict_loader:
            if i >= count:
                break
            downscaled, source = data
            source = source.to(device)
            results.append(torch.clamp(
                net(source).data[0, :, :, :] / 2 + 0.5, min=0, max=1)
            )
            i += 1
    return results
