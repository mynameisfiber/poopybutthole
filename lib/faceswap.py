import cv2
import dlib
import numpy as np
from lib import openface_wrapper
from collections import namedtuple


PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

ImageInfo = namedtuple(
    'ImageInfo',
    'aligned_face face_box landmarks imghash'.split()
)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def extract_face_infos(image):
    rects = detector(image)
    if len(rects) == 0:
        raise NoFaces
    for face_box in rects:
        aligned_face, landmarks = openface_wrapper.align_face(image, face_box)
        imghash = openface_wrapper.hash_face(aligned_face)
        yield ImageInfo(aligned_face, face_box, landmarks, imghash)


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = tuple(point)
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255, 0.5))
    return im


def load_image(fd, max_size=None):
    data = np.fromstring(fd.read(), np.uint8)
    image = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_COLOR)
    if max_size is not None:
        img_size = image.shape[:-1]
        ratio = max(
            limit/float(img) for limit, img in zip(max_size, img_size)
        )
        if ratio < 1:
            new_size = map(int, (img * ratio for img in img_size))
            image = cv2.resize(image, tuple(new_size), interpolation=cv2.INTER_AREA)
    return image
