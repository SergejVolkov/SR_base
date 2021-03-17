import os
import sys
import pyprind
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import dl_modules.dataset as ds
from cm_modules.utils import imwrite


def crop(folder: str, width: int, height: int) -> None:
    folder = ds.SAVE_DIR + 'data/' + folder
    if not os.path.isdir(folder):
        print('Folder "' + folder + '" does not exist!')
        return
    if width == 0 or height == 0:
        print('Please, specify valid crop resolution!')
        return

    transform = transforms.CenterCrop((height, width))
    dataset = ds.Dataset(folder, scale=ds.scale, downscaling='none')
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds.valid_batch_size,
                                         shuffle=False, num_workers=0)

    if not os.path.isdir(folder + '/crop'):
        os.makedirs(folder + '/crop')

    total = len(loader)
    iter_bar = pyprind.ProgBar(total, title="Crop", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            cropped = transform(source)
            hor_pad = (source.shape[3] - width) // 2
            ver_pad = (source.shape[2] - height) // 2
            letterbox = source - F.pad(cropped,
                                       [hor_pad, hor_pad, ver_pad, ver_pad],
                                       mode='constant', value=-1)
            if torch.mean(letterbox) < 0.012:
                imwrite(folder + '/crop/' + dataset.ids[i], cropped)
            iter_bar.update()
            i += 1
    iter_bar.update()
