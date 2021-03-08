import cv2
import sys
import pyprind
import torch
import numpy as np
import dl_modules.dataset as ds
from cm_modules.utils import imwrite


def predict(net: torch.nn.Module, device: torch.device) -> None:
    net.eval()

    dataset = ds.Dataset(ds.SAVE_DIR + 'data/predict', scale=ds.scale,
                         downscaling='none')
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds.valid_batch_size,
                                         shuffle=False, num_workers=0)
    total = len(loader)
    iter_bar = pyprind.ProgBar(total, title="Predict", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            source = source.to(device)
            imwrite(
                ds.SAVE_DIR + 'data/output/' + dataset.ids[i][:-4] + '_x2.png',
                net(source)
            )
            iter_bar.update()
            i += 1
