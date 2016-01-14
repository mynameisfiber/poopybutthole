from scipy.spatial import cKDTree
import cv2
from lib import faceswap
from lib import transform
from collections import defaultdict
import cPickle
import os


class FaceDB(object):
    def __init__(self, data_dir):
        self.image_lookup = {}
        self.hashes_kdtree = {}
        self._load_data(data_dir)

    def _parse_image(self, image, basename, db, image_path):
        try:
            os.makedirs(os.path.join("./database/", db, 'cache'))
        except:
            pass
        for i, info in enumerate(faceswap.extract_face_infos(image)):
            print "\t\tAdding face: ", i
            file_base = "./database/{}/cache/{}_{}".format(db, basename, i)
            data = {
                "imghash": info.imghash,
                "landmarks": info.landmarks,
                "face_box": info.face_box,
                "image_path": image_path,
                "id": image_path + str(i)
            }
            cPickle.dump(data, open(file_base + ".pkl", 'w+'))
            yield data['id'], data

    def _load_cache(self, cache_dir, cache_files):
        for item in cache_files:
            if item.endswith(".pkl"):
                fullpath = os.path.join(cache_dir, item)
                data = cPickle.load(open(fullpath))
                yield data['id'], data

    def _load_data(self, data_dir):
        data = defaultdict(dict)
        for dirpath, _, files in os.walk('./database/'):
            db = os.path.basename(dirpath)
            if db == 'cache':
                cache_db = dirpath.split(os.path.sep)[-2]
                print "Loading cache: ", cache_db
                data[cache_db].update(self._load_cache(dirpath, files))
            elif files:
                print "Looking at DB: ", db
                for filename in files:
                    if any(filename.endswith(f) for f in ('jpg', 'jpeg', 'png')):
                        fullpath = os.path.join(dirpath, filename)
                        image = faceswap.load_image(open(fullpath))
                        basename = os.path.splitext(filename)[0]
                        cache_loc = os.path.join(dirpath, "cache", basename + '_0.pkl')
                        if not os.path.exists(cache_loc):
                            print "\tAnalysing: ", filename
                            extracted_data = self._parse_image(image, basename, db, fullpath)
                            data[db].update(extracted_data)
        for db, raw_data in data.iteritems():
            if raw_data:
                print "Database {} has {} items".format(db, len(raw_data))
                db_items = raw_data.values()
                self.image_lookup[db] = db_items
                self.hashes_kdtree[db] = cKDTree([data['imghash'] for data in db_items])

    def find(self, imghash, dataset):
        score, idx = self.hashes_kdtree[dataset].query(imghash)
        return (score, self.image_lookup[dataset][idx])

    def swap(self, image_fd, dataset, threshold=2.0, max_size=(1024,1024)):
        target = faceswap.load_image(image_fd, max_size=max_size)
        face_info = faceswap.extract_face_infos(target)
        for info in face_info:
            score, db_image_data = self.find(info.imghash, dataset)
            if score < threshold:
                db_image = faceswap.load_image(open(db_image_data['image_path']))
                target = transform.faceswap(db_image, db_image_data, target, info)
        return target

