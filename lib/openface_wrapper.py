#!/usr/bin/env python2.7

from __future__ import print_function

import sys
sys.path.append("./openface/")

import openface
import cv2
import numpy as np
from lib.affine_transform import transformation_from_points


IMAGE_SIZE = 96
_ALIGN = openface.AlignDlib(
    "data/shape_predictor_68_face_landmarks.dat"
)
_NET = openface.TorchNeuralNet(
    "openface/models/openface/nn4.small2.v1.t7",
    imgDim=IMAGE_SIZE,
    cuda=False
)

ALIGNED_LANDMARKS = IMAGE_SIZE * \
    np.array(openface.align_dlib.MINMAX_TEMPLATE, dtype=np.float32)


def _normalize(image, image_size, face_box, landmarks):
    H = transformation_from_points(landmarks, ALIGNED_LANDMARKS)
    return cv2.warpAffine(image, H[:2], (image_size, image_size),
                          flags=cv2.INTER_AREA)


def align_face(image, face_box, image_size=IMAGE_SIZE):
    landmarks = np.asarray(
        _ALIGN.findLandmarks(image, face_box),
        dtype=np.float32
    )
    aligned_face = _normalize(image, image_size, face_box, landmarks)
    return aligned_face, landmarks


def hash_face(aligned_face):
    return _NET.forward(aligned_face)
