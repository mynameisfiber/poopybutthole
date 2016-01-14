from tornado import gen
from tornado import web
from tornado.web import asynchronous
import mmh3
import cv2

from lib import faceswap

from os import path
from cStringIO import StringIO


class PoopyButthole(web.RequestHandler):
    @asynchronous
    @gen.coroutine
    def post(self):
        image_set = self.get_argument("set", "politics")
        threshold = int(self.get_argument('threshold', 2.0))
        image_body = self.request.files['image'][0]['body']

        _id = mmh3.hash(str(image_set) + str(image_body) + str(threshold))
        image_path = "./tmp/{}.png".format(_id)

        if True or not path.exists(image_path):
            image_fd = StringIO(image_body)
            facedb = self.application.settings['facedb']
            new_image = facedb.swap(image_fd, image_set, threshold=threshold)
            cv2.imwrite(image_path, new_image)
        self.redirect("/poopy/image?id=" + str(_id))

class ShowImage(web.RequestHandler):
    def get(self):
        try:
            _id = int(self.get_argument("id"))
        except ValueError:
            self.set_status(404)
        else:
            try:
                image_path = "./tmp/{}.png".format(_id)
                self.set_header('Content-Type', "image/png")
                self.write(open(image_path).read())
            except IOError:
                self.set_status(404)
        return self.finish()
