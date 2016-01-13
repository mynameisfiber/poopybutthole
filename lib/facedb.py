from scipy.spatial import cKDTree
from lib import faceswap
from lib import transform
import cPickle
import os


class FaceDB(object):
    def __init__(self, data_dir):
        self.data = []
        self.lookup = {}
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        print "Loading FaceDB"
        self.lookup = {}
        for faceset in os.listdir(data_dir):
            faceset_path = os.path.join(data_dir, faceset)
            if not os.path.isdir(faceset_path):
                continue
            print "\tLoading set: ", faceset
            for image in os.listdir(faceset_path):
                image_path = os.path.join(faceset_path, image)
                # hack to check if the file is an image
                if '.' not in image or image.rsplit('.', 1)[1] not in ('jpg', 'jpeg', 'png'):
                    continue
                cache_path = os.path.join(faceset_path, "cache/", image + '.pkl')
                print "\t\tsample: ", image
                try:
                    d = cPickle.load(open(cache_path))
                except IOError:
                    db_image, landmarks_gen = faceswap.process_image(open(image_path))
                    d = [(image_path, f, l) for f, l in landmarks_gen]
                    cPickle.dump(d, open(cache_path, "w+"))
                self.data.extend(d)
            self.lookup[faceset] = cKDTree([faceswap.normalize_landmarks(f, l) for _, f,l in self.data])

    def find(self, face, landmarks, dataset):
        norm_landmarks = faceswap.normalize_landmarks(face, landmarks)
        score, idx = self.lookup[dataset].query(norm_landmarks)
        return (score, self.data[idx])

    def swap(self, image, face_landmarks, dataset, threshold=2.0):
        image = image.copy()
        for face, landmarks in face_landmarks:
            score, (db_image_path, db_face, db_landmarks) = self.find(face, landmarks, dataset)
            if score < threshold:
                db_image = faceswap.load_image(open(db_image_path))
                image = transform.faceswap(db_face, db_image, db_landmarks, image, face, landmarks)
        return image

