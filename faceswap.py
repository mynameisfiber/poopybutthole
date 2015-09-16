import cv2
import dlib
import numpy as np

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

def read_im_and_landmarks(fname, min_area=None):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    landmark_gen = get_landmarks(im, min_area=min_area)
    return im, landmark_gen


if __name__ == "__main__":
    from scipy.spatial import cKDTree
    print "Making DB"
    landmarks_db = []
    db_files = ["images/micha_fb{}.jpg".format(i) for i in xrange(1,6)]
    db_files += ['cruz.png', 'trump.jpg', "images/abby_fb1.jpg"]
    for image in db_files:
        db_image, landmarks_gen = read_im_and_landmarks(image)
        landmarks_db.extend([(db_image, f, l) for f,l in landmarks_gen])
    data = [normalize_landmarks(f, l) for _, f,l in landmarks_db]
    database = cKDTree(data)

    print "Matching faces"
    image, landmarks_gen = read_im_and_landmarks("images/micha_fb6.jpg")
    for i, (face, landmarks) in enumerate(landmarks_gen):
        norm_landmarks = normalize_landmarks(face, landmarks)
        search = database.query(norm_landmarks)
        print search
        if search[0] < 2.0:
            closest_match = search[1]
            db_image, db_face, db_landmarks = landmarks_db[closest_match]
            image = transform.faceswap(db_image, db_landmarks, image, face, landmarks)
        else:
            closest_match = "NA"
        center = (face.center().x, face.center().y)
        cv2.putText(image, str(closest_match), center,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 255))
    cv2.imwrite("matches.jpg", image)
