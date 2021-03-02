import os
import cv2
import sys
import pyprind
import torch
import numpy as np
import torchvision.transforms as transforms
import dl_modules.dataset as ds


def crop(folder: str, width: int, height: int) -> None:
    folder = ds.SAVE_DIR + 'data/' + folder
    if not os.path.isdir(folder):
        print('Folder "' + folder + '" does not exist!')
        return
    if width == 0 or height == 0:
        print('Please, specify valid crop resolution!')
        return
    if not os.path.isdir(folder + '/crop'):
        os.makedirs(folder + '/crop')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((height, width))
    ])
    dataset = ds.Dataset(folder, scale=ds.scale, normalization=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds.valid_batch_size,
                                         shuffle=False, num_workers=0)
    total = len(loader)
    iter_bar = pyprind.ProgBar(total, title="Crop", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            output = np.transpose(source.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255
            cv2.imwrite(folder + '/crop/' + dataset.ids[i], cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            iter_bar.update()
            i += 1
    iter_bar.update()
