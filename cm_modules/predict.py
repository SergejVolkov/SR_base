import sys
import pyprind
import torch
import dl_modules.dataset as ds
import cm_modules.utils as utils
from cm_modules.utils import imwrite


def predict(net: torch.nn.Module, device: torch.device, cut: bool=False) -> None:
    net.eval()
    total = len(ds.predict_loader)
    iter_bar = pyprind.ProgBar(total, title="Predict", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in ds.predict_loader:
            downscaled, source = data
            source = source.to(device)
            if cut:
                pieces = utils.cut_image(source)
                out_pieces = []
                for piece in pieces:
                    out_pieces.append(net(piece))
                output = utils.glue_image(out_pieces)
            else:
                output = net(source)
            imwrite(
                ds.SAVE_DIR + 'data/output/' + ds.predict_set.ids[i][:-4] + '_sr.png',
                output
            )
            iter_bar.update()
            i += 1


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
