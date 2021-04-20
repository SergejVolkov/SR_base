import os
import sys
import pyprind
import torch
import dl_modules.dataset as ds
import dl_modules.realsr as realsr
import dl_modules.transforms as trf
import cm_modules.utils as utils
from cm_modules.utils import imwrite


def generate(folder: str) -> None:
    if folder is None:
        folder = ds.valid_hr_dir
    else:
        folder = ds.SAVE_DIR + 'data/' + folder
    if not os.path.isdir(folder):
        print('Folder "' + folder + '" does not exist!')
        return

    # Load kernels and noise
    ds.train_batch_size = 1
    ds.kernel_train_dir = ds.kernel_valid_dir
    ds.noise_train_dir = ds.noise_valid_dir
    ds.init_data()
    print('%d kernels\n%d noise patches' % (len(ds.kernel_storage), len(ds.noise_set)))

    dataset = ds.Dataset(folder, scale=ds.scale,
                         augmentation=trf.get_input_image_augmentation(),
                         downscaling='kernel_even',
                         aspect_ratio=ds.aspect_ratio,
                         extra_scale=ds.extra_scale)
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds.valid_batch_size,
                                         shuffle=False, num_workers=0)

    if not os.path.isdir(folder + '/lr'):
        os.makedirs(folder + '/lr')

    total = len(loader)
    iter_bar = pyprind.ProgBar(total, title="Generate", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            ds.noise_set.transform = trf.get_generate_noise_transform(
                downscaled.shape[3], downscaled.shape[2]
            )
            downscaled = realsr.inject_noise(downscaled, ds.noise_set)
            downscaled = utils.scale(downscaled, 1 / ds.aspect_ratio, 1 / ds.extra_scale)
            imwrite(folder + '/lr/' + dataset.ids[i], downscaled)
            iter_bar.update()
            i += 1
    iter_bar.update()
