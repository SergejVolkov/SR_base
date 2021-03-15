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
    folder = ds.SAVE_DIR + 'data/' + folder
    if not os.path.isdir(folder):
        print('Folder "' + folder + '" does not exist!')
        return

    # Load kernels and noise
    ds.train_batch_size = 1
    ds.kernel_dir = os.path.join(ds.SAVE_DIR, 'data/SoulTaker/SoulTaker_valid_kernel')
    ds.noise_dir  = os.path.join(ds.SAVE_DIR, 'data/SoulTaker/SoulTaker_valid_noise')
    ds.init_data()
    print('%d kernels\n%d noise patches' % (len(ds.kernel_storage), len(ds.noise_set)))

    dataset = ds.Dataset(folder, scale=ds.scale,
                         augmentation=trf.get_input_image_augmentation(),
                         downscaling='kernel',
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
            downscaled = realsr.inject_noise(downscaled, ds.noise_loader)
            downscaled = utils.scale(downscaled, 1 / ds.aspect_ratio, 1 / ds.extra_scale)
            imwrite(folder + '/lr/' + dataset.ids[i], downscaled)
            iter_bar.update()
            i += 1
    iter_bar.update()
