import os
import cv2
import sys
import math
import pyprind
import torch
import numpy as np
import torchvision.transforms as transforms
import dl_modules.dataset as ds


def enhance_images(folder: str, denoise_strength: int,
                   window_size: int, contrast: int, kernel_size: int) -> None:
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
    if not os.path.isdir(folder + '/enhanced'):
        os.makedirs(folder + '/enhanced')
    iter_bar = pyprind.ProgBar(total, title="Enhance", stream=sys.stdout)
    i = 0

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            noisy = np.transpose(source.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255
            noisy = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
            enhanced = enhance(noisy, denoise_strength, window_size, contrast)
            cv2.imwrite(folder + '/enhanced/' + dataset.ids[i][:-4] + '_e.png', enhanced)
            iter_bar.update()
            i += 1
    iter_bar.update()


def enhance(image, denoise_strength: int=5, window_size: int=5, contrast: int=5, kernel_size: int=5):
    denoised = gentle_denoise(image, denoise_strength, window_size, kernel_size)
    equalized = auto_contrast(denoised, strength=contrast)
    # dithered = dither(denoised)
    return equalized


def dither(image, dither_strength: int=1):
    return np.clip(np.round(
        image.astype(np.float) + (np.random.rand(*image.shape) - 0.5) * dither_strength
    ), a_min=0, a_max=255).astype(np.uint8)


def gentle_denoise(noisy, denoise_strength, window_size, kernel_size):
    denoised = cv2.fastNlMeansDenoisingColored(
        noisy.astype(np.uint8), None, denoise_strength, denoise_strength, window_size, window_size * 3
    )
    noisy = noisy.astype(np.float32)
    extracted_noise = noisy - denoised.astype(np.float32)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        extracted_noise -= cv2.filter2D(extracted_noise, -1, kernel)
    denoised = noisy - extracted_noise
    return denoised


def auto_contrast(image, clip_hist_percent: int=1,
                  strength: int=5, saturation: int=64):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    if maximum_gray <= minimum_gray:
        maximum_gray = minimum_gray + 1

    # Calculate values
    alpha = 255 / (maximum_gray - minimum_gray)
    average_gray = (maximum_gray + minimum_gray) // 2

    correlation = 0.5
    contrast = int(round(saturate(
        alpha * strength,
        saturation
    )))
    brightness = int(round(saturate(
        correlation * strength * contrast * (127 - average_gray) / 128,
        saturation
    )))

    # print('\n', brightness, contrast)

    auto_result = apply_brightness_contrast(image, brightness, contrast)
    return auto_result


def saturate(x: float, threshhold: float):
    return sign(x) * threshhold * (1 - math.exp(-abs(x) / threshhold))


def sign(x: float):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    return -1


def apply_brightness_contrast(input_img, brightness: int=0, contrast: int=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
