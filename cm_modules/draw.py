import os
import sys
import cv2
import math
import pyprind
import random
import aggdraw
import torch
import numpy as np
import numpy.linalg as lng
import torchvision.transforms as transforms
import dl_modules.dataset as ds
from PIL import Image

w_min = 1.0
w_max = 10.0
line_colors = [
    ((255, 255, 255), 0.01),
    ((140, 140, 140), 0.02),
    ((153, 137, 76), 0.01),
    ((80, 80, 80), 0.06),
    ((40, 40, 40), 0.1),
    ((0, 0, 0), 0.8),
]


def random_palette(count: int, colors_per_image: int=3) -> list:
    palette = []
    for i in range(count * colors_per_image):
        palette.append(np.random.random((3, )))
    return palette


def extract_palette(folder: str, kernel_size: int=11, colors_per_image: int=3) -> list:
    folder = ds.SAVE_DIR + 'data/' + folder
    if not os.path.isdir(folder):
        print('Folder "' + folder + '" does not exist!')
        exit(0)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ds.Dataset(folder, scale=ds.scale,
                         normalization=transform, downscaling='none')
    loader = torch.utils.data.DataLoader(dataset, batch_size=ds.valid_batch_size,
                                         shuffle=False, num_workers=0)
    total = len(loader)
    iter_bar = pyprind.ProgBar(total, title="Palette", stream=sys.stdout)
    i = 0

    palette = []

    with torch.no_grad():
        for data in loader:
            downscaled, source = data
            image = np.transpose(source.squeeze(0).cpu().numpy(), (1, 2, 0))
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
                image = cv2.filter2D(image, -1, kernel)
            h, w, _ = image.shape
            for j in range(colors_per_image):
                color = image[random.randrange(0, h), random.randrange(0, w), :]
                palette.append(color)
            iter_bar.update()
            i += 1
    return palette


def draw_data(name: str, palette: list, resolution: tuple, sample_count: int,
              line_count: int, sample_id: int=0) -> None:
    save_dir = ds.SAVE_DIR + 'data/' + name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    iter_bar = pyprind.ProgBar(sample_count, title='Draw', stream=sys.stdout)

    for i in range(sample_id, sample_id + sample_count):
        image = np.ones((*resolution[::-1], 3), dtype=np.float)
        fill_colors(image, palette)
        image = Image.fromarray((image * 255).astype(np.uint8))
        generate_lines(image, line_count)
        image.save(save_dir + '/%05d.png' % i, "PNG")
        iter_bar.update()


def fill_colors(image: np.ndarray, palette: list) -> None:
    h, w, _ = image.shape
    colors = []
    col_count = 3
    threshold = 0.0015
    for i in range(col_count):
        colors.append(palette[random.randrange(0, len(palette))])
    period = random.uniform(math.pi, 20 * math.pi)
    period2 = random.uniform(math.pi, 20 * math.pi)
    phase = random.uniform(0, math.pi)
    solid = random.randrange(0, 2)
    for i in range(h):
        for j in range(w):
            ratio1 = (math.sin(j / (w / 10) * math.pi - phase * 2.0) +
                      math.sin(i / (w / 10) * math.pi + phase / 2.0) + 2.0) / 4.0
            ratio2 = (math.sin(j / w * period + phase + i / h * period2) +
                      math.sin(i / h * period + phase + j / w * period2) + 2.0) / 8.0
            image[i, j, :] = (1.0 - ratio2) * (colors[0] * ratio1 + colors[1] * (1.0 - ratio1)) + ratio2 * colors[2]
            if solid == 1:
                distance = []
                for k in range(col_count):
                    distance.append(lng.norm(colors[k] - image[i, j, :]))
                min_dst = min(distance)
                min_ind = []
                for k in range(col_count):
                    diff = np.abs(min_dst - distance[k])
                    if diff < threshold:
                        min_ind.append((k, diff))
                if len(min_ind) == 1:
                    image[i, j, :] = colors[min_ind[0][0]]
                else:
                    color = np.zeros(3, dtype=np.float)
                    norm = 0.0
                    for ind, diff in min_ind:
                        weight = threshold - diff
                        color += colors[ind] * weight
                        norm += weight
                    image[i, j, :] = color / norm


def generate_lines(image: Image, line_count: int=100) -> None:
    p = 0.0
    color_ind = 0
    for i in range(line_count):
        if p > line_colors[color_ind][1]:
            p -= line_colors[color_ind][1]
            color_ind += 1
        p += 1 / line_count
        color = line_colors[color_ind][0]
        if random.randrange(0, 40) == 0:
            width = random.uniform(w_min, w_max)
        elif random.randrange(0, 6) == 0:
            width = random.uniform(w_min, 3.0)
        else:
            width = 2.0
        generate_line(image, color, width)


def generate_line(image: Image, color: tuple, line_width: float) -> None:
    w, h = image.size
    extra_space = 150
    canvas = aggdraw.Draw(image)
    pen = aggdraw.Pen(color=color, width=int(round(line_width)))
    path = aggdraw.Path()
    if random.randrange(0, 4) == 0:
        path.moveto(random.randrange(0, w), random.randrange(0, h))
        path.curveto(random.randrange(0, w), random.randrange(0, h),
                     random.randrange(0, w), random.randrange(0, h),
                     random.randrange(0, w), random.randrange(0, h))
    else:
        if random.randrange(0, 1920 + 1080) < 1080:
            x0, y0 = random.randrange(0, 2) * (w - 1), random.randrange(0, h)
            if random.randrange(0, 6) == 0:
                x1, y1 = w - 1 - x0, random.randrange(0, h)
            else:
                x1, y1 = random.randrange(0, w), random.randrange(0, 2) * (h - 1)
        else:
            x0, y0 = random.randrange(0, w), random.randrange(0, 2) * (h - 1)
            if random.randrange(0, 6) == 0:
                x1, y1 = random.randrange(0, w), h - 1 - y0
            else:
                x1, y1 = random.randrange(0, 2) * (w - 1), random.randrange(0, h)
        left = min(x0, x1) - extra_space
        right = max(x0, x1) + extra_space
        top = min(y0, y1) - extra_space
        bottom = max(y0, y1) + extra_space

        path.moveto(x0, y0)
        path.curveto(random.randrange(left, right), random.randrange(top, bottom),
                     random.randrange(left, right), random.randrange(top, bottom),
                     x1, y1)
    canvas.path(path, path, pen)
    canvas.flush()

# def generate_lines(image: np.ndarray, line_count: int=100, v_max: float=4.0) -> None:
#     h, w, _ = image.shape
#     for i in range(line_count):
#         x_0 = random.randrange(0, w)
#         y_0 = random.randrange(0, h)
#         v_0 = random.uniform(v_max / 2.0, v_max)
#         phi_0 = random.uniform(0, 2 * math.pi)
#         if random.randint(0, 1) == 0:
#             line_width = w_min
#         else:
#             line_width = random.uniform(w_min, w_max)
#         generate_line(image, x_0, y_0, v_0, phi_0, line_width)
#         generate_line(image, x_0, y_0, -v_0, phi_0, line_width)
#
#
# def generate_line(image: np.ndarray, x: float, y: float, v: float, phi: float, line_width: float) -> None:
#     h, w, _ = image.shape
#     max_d_v = 0.001
#     max_d2_phi = 0.00004
#     part = 0.01
#     d_v = random.uniform(0, max_d_v)
#     d2_phi = random.uniform(-max_d2_phi, max_d2_phi)
#     d_phi = 0.0
#     iters = 0
#     max_iters = w / abs(v)
#     min_w = line_width * 2.0
#     max_w = w - min_w
#     min_h = min_w
#     max_h = h - min_h
#     while min_w <= x < max_w and min_h <= y < max_h:
#         x_1 = x + v * math.cos(phi)
#         y_1 = y + v * math.sin(phi)
#         if min_w <= x_1 < max_w and min_h <= y_1 < max_h:
#             draw_line(image, x, y, x_1, y_1, line_width)
#         else:
#             break
#         x = x_1
#         y = y_1
#         phi += d_phi
#         d_phi += d2_phi
#         d2_phi += random.uniform(-max_d2_phi * part, max_d2_phi * part)
#         v += d_v
#         d_v += random.uniform(-max_d_v * part, max_d_v * part)
#         iters += 1
#         if iters > max_iters:
#             break
#
#
# def draw_line(image: np.ndarray, x0: float, y0: float, x1: float, y1: float, width: float):
#     rr, cc, val = weighted_line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)), width)
#     val = np.stack([val for _ in range(3)], axis=-1)
#     image[rr, cc, :] *= (1.0 - val)
#     image[rr, cc, :] += val * line_color
#
#
# def trapez(y,y0,w):
#     return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)
#
#
# def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
#     # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
#     # If either of these cases are violated, do some switches.
#     if abs(c1-c0) < abs(r1-r0):
#         # Switch x and y, and switch again when returning.
#         xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
#         return (yy, xx, val)
#
#     # At this point we know that the distance in columns (x) is greater
#     # than that in rows (y). Possibly one more switch if c0 > c1.
#     if c0 > c1:
#         return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)
#
#     if c1 == c0:
#         return weighted_line(r1, c1, r0, c0 + 1, w, rmin=rmin, rmax=rmax)
#
#     # The following is now always < 1 in abs
#     slope = (r1-r0) / (c1-c0)
#
#     # Adjust weight by the slope
#     w *= np.sqrt(1+np.abs(slope)) / 2
#
#     # We write y as a function of x, because the slope is always <= 1
#     # (in absolute value)
#     x = np.arange(c0, c1+1, dtype=float)
#     y = x * slope + (c1*r0-c0*r1) / (c1-c0)
#
#     # Now instead of 2 values for y, we have 2*np.ceil(w/2).
#     # All values are 1 except the upmost and bottommost.
#     thickness = np.ceil(w/2)
#     yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
#     xx = np.repeat(x, yy.shape[1])
#     vals = trapez(yy, y.reshape(-1,1), w).flatten()
#
#     yy = yy.flatten()
#
#     # Exclude useless parts and those outside of the interval
#     # to avoid parts outside of the picture
#     mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))
#
#     return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])
