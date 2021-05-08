import cv2
import sys
import pyprind
import torch
import numpy as np
import dl_modules.dataset as ds
import dl_modules.transforms as trf
import cm_modules.utils as utils
import skvideo.io as vio
from cm_modules.enhance import correct_colors
from cm_modules.utils import convert_to_cv_8bit


def inference(name: str, net: torch.nn.Module, device: torch.device,
              length: float=0, start: float=0, batch: int=1,
              cut: bool=False, normalize: bool=False, crf: int=17) -> None:
    net.eval()
    norm = ds.get_normalization()
    trn = trf.get_predict_transform(*ds.predict_res)

    cap = cv2.VideoCapture(ds.SAVE_DIR + 'data/video/' + name + '.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))

    w, h = ds.predict_res
    w *= ds.scale
    h *= ds.scale
    if normalize:
        path = ds.SAVE_DIR + 'data/output/' + name + '_sr_n.mp4'
    else:
        path = ds.SAVE_DIR + 'data/output/' + name + '_sr.mp4'
    out = vio.FFmpegWriter(path, inputdict={
        '-r': '%g' % fps,
    }, outputdict={
        '-vcodec': 'libx264',
        '-crf': '%d' % crf,
        '-tune': 'animation',
        '-preset': 'veryslow',
        '-r' : '%g' % fps
    })

    i = 0
    if length != 0:
        total = int(round(length * fps))
    else:
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    iter_bar = pyprind.ProgBar(total, title='Inference ' + name, stream=sys.stdout)

    frame_list = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or (length != 0 and i >= length * fps):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = norm(trn(image=frame)["image"]).to(device)
            frame_list.append(frame)
            if len(frame_list) == batch or i >= length * fps - 1:
                frames = torch.stack(frame_list)
                if cut:
                    pieces = utils.cut_image(frames)
                    out_pieces = []
                    for piece in pieces:
                        out_pieces.append(net(piece))
                    output = utils.glue_image(out_pieces)
                else:
                    output = net(frames)
                for j in range(len(frame_list)):
                    if normalize:
                        out_frame = correct_colors(output[j, :, :, :], frames[j, :, :, :])
                    else:
                        out_frame = output[j, :, :, :]
                    out.writeFrame(cv2.cvtColor(convert_to_cv_8bit(out_frame), cv2.COLOR_RGB2BGR))
                frame_list.clear()
            i += 1
            iter_bar.update()
    cap.release()
    out.close()
