import os
import cv2
import sys
import pyprind
import torch
import random
import dl_modules.dataset as ds
# import numpy as np


def prepare(name: str, step: float,
            length: float=0, start: float=0, random_start: bool=False,
            episodes: bool=False, ep_start: int=0, ep_end: int=0,
            sample_id: int=0) -> None:
    save_dir = ds.SAVE_DIR + 'data/' + name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    bar_last_step = 0
    bar_init = False
    iter_bar = None

    for ep in range(ep_start, ep_end + 1):
        if episodes:
            video_path = ds.SAVE_DIR + 'data/video/' + name + '_%02d.mp4' % ep
        else:
            video_path = ds.SAVE_DIR + 'data/video/' + name + '.mp4'
        if not os.path.isfile(video_path):
            if episodes:
                print('No video file with the name "' + name + '_%02d"!' % ep)
            else:
                print('No video file with the name "' + name + '"!')
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if random_start:
            start_frame = int(round(start * fps))
            start_frame = random.randint(0, start_frame)
        else:
            start_frame = int(round(start * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        step_frame = int(round(step * fps))

        i = 0

        if length != 0:
            total = int(round(length * fps))
        else:
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) - start_frame
        if not bar_init:
            bar_last_step = total * (ep_end + 1 - ep_start)
            iter_bar = pyprind.ProgBar(bar_last_step, title='Prepare data', stream=sys.stdout)
            bar_init = True

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret or (length != 0 and i >= total):
                    break
                if i % step_frame == 0:
                    output = cv2.fastNlMeansDenoisingColored(frame, None, 1, 1, 5, 15)
                    cv2.imwrite(save_dir + '/%05d.png' % sample_id, output)
                    # cv2.imwrite(save_dir + '/%05d_raw.png' % (i // step), frame)
                    sample_id += 1
                i += 1
                iter_bar.update()
        cap.release()
    iter_bar.update(item_id=bar_last_step)
