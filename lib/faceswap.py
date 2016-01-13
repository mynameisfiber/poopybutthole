import cv2
import dlib
import numpy as np
from PIL import Image

import transform


PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im, min_area=None):
    rects = detector(im, 1)
    if len(rects) == 0:
        raise NoFaces
    for face in rects:
        if face.area() > min_area:
            landmarks = np.array([[p.x, p.y] for p in predictor(im,face).parts()])
            yield face, landmarks

def normalize_landmarks(face, landmarks):
    scale = [float(face.width()), float(face.height())]
    nose = landmarks[transform.NOSE_POINTS] / scale
    eyes = landmarks[transform.LEFT_EYE_POINTS + transform.RIGHT_EYE_POINTS] / scale
    # angle of the nose
    nose_angle = np.arctan2(nose[3,1] - nose[1,1], nose[3,0] - nose[0,0])
    eye_angle = np.arctan2(eyes[6,1] - eyes[3,1], eyes[6,0] - eyes[3,0])
    # orientation and size of the nose
    left_nose = nose[4,:] - nose[6,:]
    left_nose /= np.linalg.norm(left_nose)
    right_nose = nose[7,:] - nose[6,:]
    right_nose /= np.linalg.norm(right_nose)
    orientation = np.cross(left_nose, right_nose)
    return [nose_angle, eye_angle, orientation]

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

def load_image(fd, resize=False):
    data = np.fromstring(fd.read(), np.uint8)
    image = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_COLOR)
    f = 1024.0 / image.shape[0]
    if resize and f < 1:
        image = cv2.resize(image, (0,0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return image

def process_image(fd, min_area=None, resize=False):
    image = load_image(fd, resize=resize)
    landmark_gen = get_landmarks(image, min_area=min_area)
    return image, landmark_gen
