import cv2
import os
import numpy as np

INIT_SIZE = 24
FONT_WIDTH_FACTOR = 0.8

c_h, c_w = 128, 128
scale = 3
img_num = 7
max_row_w = 6
i_size = (1920, 1080)
start_index = len('The_SoulTaker_01_a_sr_')
font_size = 24
header_h = font_size * 3

coords = [(780, 966), (579, 762), (482, 497),
          (385, 968), (485, 1000), (890, 1680), (420, 1250)]

# coords = [(548, 1037), (482, 497), (442, 361), (170, 1220), (354, 630)]

img_dir = 'D:/Foma/Python/SR_base/data/output'
save_dir = 'D:/Foma/Documents/P&P/media/Projects/2 Term, Video Enhance CP2/images'

sc_h = int(c_h * scale)
sc_w = int(c_w * scale)

imgs = [name for name in os.listdir(img_dir) if
        name.lower().endswith('.png') or
        name.lower().endswith('.jpg') or
        name.lower().endswith('.jpeg') or
        name.lower().endswith('.gif') or
        name.lower().endswith('.bmp')]
imgs.sort()

img_num = len(imgs) // img_num

divide = 1
while (divide + 1) * img_num <= max_row_w:
    divide += 1

height = len(imgs) // img_num
if height % divide != 0:
    height = height // divide + 1
else:
    height = height // divide

out_img = np.ones((sc_h * height + header_h, sc_w * img_num * divide, 3), dtype=np.uint8) * 255

font = cv2.FONT_HERSHEY_COMPLEX
for i in range(img_num * divide):
    name = imgs[i % img_num][start_index:][:-4]
    cv2.putText(
        out_img, name,
        (i * sc_w + int(sc_w - len(name) * font_size * FONT_WIDTH_FACTOR) // 2, font_size * 2),
        font, font_size / INIT_SIZE, (0, 0, 0), 1, cv2.LINE_AA
    )

for i in range(len(imgs) // img_num):
    div_idx = i % height
    column = i // height
    for j in range(img_num):
        img = cv2.imread(img_dir + '/' + imgs[i * img_num + j])
        img = cv2.resize(img, i_size, interpolation=cv2.INTER_CUBIC)
        crop = img[coords[i][0]:coords[i][0] + c_h, coords[i][1]:coords[i][1] + c_w, :]
        crop = cv2.resize(crop, (sc_w, sc_h), interpolation=cv2.INTER_LANCZOS4)
        out_img[sc_h * div_idx + header_h:sc_h * (div_idx + 1) + header_h,
                sc_w * j + column * sc_w * img_num:sc_w * (j + 1) + column * sc_w * img_num, :] = crop


cv2.imwrite(save_dir + '/figure1.png', out_img)
