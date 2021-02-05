import cv2
import sys
import pyprind
import torch
import scripts.dataset as ds
import numpy as np


def predict(net: torch.nn.Module) -> None:
    dataset = ds.Dataset('data/predict', scale=2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds.valid_batch_size,
                                               shuffle=False, num_workers=0)
    total = len(loader)
    iter_bar = pyprind.ProgBar(total, title="Predict", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            # source = source.cuda()
            output = torch.clamp(net(source).squeeze(0) / 2 + 0.5, min=0, max=1)
            output = np.transpose(output.cpu().numpy(), (1, 2, 0)) * 255
            cv2.imwrite('data/output/%2d.png' % i, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            iter_bar.update()
            i += 1
    iter_bar.update()
