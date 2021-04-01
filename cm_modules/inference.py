import cv2
import sys
import pyprind
import torch
import numpy as np
import dl_modules.dataset as ds
import dl_modules.transforms as trf
import cm_modules.utils as utils
from cm_modules.enhance import enhance
from cm_modules.utils import convert_to_cv_float


def inference(name: str, net: torch.nn.Module, device: torch.device,
              length: int=0, start: int=0, cut: bool=False, perform_enhance: bool=False) -> None:
    net.eval()

    cap = cv2.VideoCapture(ds.SAVE_DIR + 'data/video/' + name + '.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start * fps)
    norm = ds.get_normalization()
    trn = trf.get_predict_transform(*ds.predict_res)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = ds.predict_res
    w *= ds.scale
    h *= ds.scale
    if perform_enhance:
        path = ds.SAVE_DIR + 'data/output/' + name + '_sr_e.mp4'
    else:
        path = ds.SAVE_DIR + 'data/output/' + name + '_sr.mp4'
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    i = 0

    if length != 0:
        total = length * fps
    else:
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    iter_bar = pyprind.ProgBar(total, title='Inference', stream=sys.stdout)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or (length != 0 and i >= length * fps):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = norm(trn(image=frame)["image"]).to(device).unsqueeze(0)
            if cut:
                pieces = utils.cut_image(frame)
                out_pieces = []
                for piece in pieces:
                    out_pieces.append(net(piece))
                output = utils.glue_image(out_pieces)
            else:
                output = net(frame)
            output = convert_to_cv_float(output)
            if perform_enhance:
                output = enhance(output)
            out.write(output.astype(np.uint8))
            i += 1
            iter_bar.update()
    cap.release()
    out.release()
