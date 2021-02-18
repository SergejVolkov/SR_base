import cv2
import sys
import pyprind
import torch
import scripts.dataset as ds
import numpy as np


def inference(net: torch.nn.Module, device: torch.device, length: int=0) -> None:
    cap = cv2.VideoCapture(ds.SAVE_DIR + 'data/video/input.mp4')
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    iter_bar = pyprind.ProgBar(total, title='Predict', stream=sys.stdout)
    norm = ds.get_normalization()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(ds.SAVE_DIR + 'data/video/output.mp4', fourcc, fps, (w, h))
    i = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or (length != 0 and i >= length * fps):
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = norm(frame).to(device).unsqueeze(0)
            # pieces = ds.cut_image(frame)
            # out_pieces = []
            # for piece in pieces:
            #     out_pieces.append(torch.clamp(net(piece) / 2 + 0.5, min=0, max=1))
            # output = ds.glue_image(out_pieces).squeeze(0)
            output = torch.clamp(net(frame) / 2 + 0.5, min=0, max=1)
            output = np.transpose(output.cpu().numpy(), (1, 2, 0)) * 255
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            out.write(output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1
            iter_bar.update()
    iter_bar.update()
