# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image
import cv2
import pyflow


@dataclass
class OFlowEstimator:
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    def run(self, im1, im2):
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.

        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, self.alpha, self.ratio, self.minWidth, self.nOuterFPIterations, self.nInnerFPIterations,
            self.nSORIterations, self.colType)
        return u, v, im2W


def colorize(shape, flow):
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def run(name1, name2):
    im1 = np.array(Image.open(name1))
    im2 = np.array(Image.open(name2))

    oflow_estimator = OFlowEstimator()

    s = time.time()
    u, v, im2W = oflow_estimator.run(im1, im2)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    np.save('examples/outFlow.npy', flow)

    if args.viz:
        rgb = colorize(im1.shape, flow)
        cv2.imwrite('examples/outFlow_new.png', rgb)
        cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Demo for python wrapper of Coarse2Fine Optical Flow')
    parser.add_argument(
        '-viz', dest='viz', action='store_true',
        help='Visualize (i.e. save) output of flow.')
    args = parser.parse_args()

    movie_file = Path("kinyoubi.mp4")
    movie_file = Path("clear.mp4")

    prev_frame, current_frame = None, None
    cap = cv2.VideoCapture(str(movie_file))

    frames = []

    _, current_frame = cap.read()
    frames.append(current_frame)

    oflow_estimator = OFlowEstimator()

    cv2.namedWindow("current", cv2.WINDOW_NORMAL)
    cv2.namedWindow("concat", cv2.WINDOW_NORMAL)

    i = -1
    while True:
        result, new_frame = cap.read()
        i += 1
        print(f"{i=}")
        if not result:
            break

        frames.append(new_frame)
        # continue
        if len(frames) < 2:
            continue
        if len(frames) > 2:
            frames.pop(0)

        prev_frame, current_frame = frames[:2]

        cv2.imshow("current", current_frame)
        cv2.waitKey(100)
        print(f"{i=} {current_frame.shape=} {np.mean(current_frame.flatten())=}")


        s = time.time()
        u, v, im2W = oflow_estimator.run(prev_frame, current_frame)
        e = time.time()
        print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
            e - s, prev_frame.shape[0], prev_frame.shape[1], prev_frame.shape[2]))
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        rgb = colorize(prev_frame.shape, flow)
        concat = np.hstack((current_frame.copy(), rgb))
        cv2.imshow("concat", concat)
        cv2.waitKey(1)
