import os
import cv2
import sys
import pyprind
import torch
import numpy as np
import torchvision.transforms as transforms
import dl_modules.dataset as ds
import dl_modules.realsr as realsr


def extract(folder: str, denoise_strength: int, window_size: int) -> None:
    folder = ds.SAVE_DIR + 'data/' + folder
    if not os.path.isdir(folder):
        print('Folder "' + folder + '" does not exist!')
        return

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ds.Dataset(folder, scale=ds.scale,
                         normalization=transform, downscaling='none')
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds.valid_batch_size,
                                         shuffle=False, num_workers=0)
    total = len(loader)
    if not os.path.isdir(folder + '/patches'):
        os.makedirs(folder + '/patches')
    iter_bar = pyprind.ProgBar(total, title="Extract", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            noisy = (np.transpose(source.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoisingColored(
                noisy, None, denoise_strength, denoise_strength, window_size, window_size * 3
            )
            extracted_noise = noisy - denoised
            cv2.imwrite(folder + '/patches/' + dataset.ids[i][:-4] + '_noise.png',
                        cv2.cvtColor(extracted_noise + realsr.noise_mean, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(folder + '/patches/' + dataset.ids[i][:-4] + '_denoised.png',
            #             cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
            # np.save(folder + '/patches/' + dataset.ids[i][:-4] + '_noise.npy', extracted_noise, False)
            iter_bar.update()
            i += 1
    iter_bar.update()
