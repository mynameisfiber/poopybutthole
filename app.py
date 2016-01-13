
from tornado import web
from tornado import options
from tornado import httpserver
from tornado import ioloop

from lib.facedb import FaceDB
from app.poopy import PoopyButthole, ShowImage

from os import path


options.define("port", default=8080, type=int, help="Port to serve on")
options.define("debug", default=False, type=bool, help="Debug Mode")


if __name__ == "__main__":
    options.parse_command_line()
    port = options.options.port
    debug = options.options.debug

    facedb = FaceDB('./database/')

    application = web.Application(
        [
            ("/poopy/butthole", PoopyButthole),
            ("/poopy/image", ShowImage),
        ],
        cookie_secret = "OMG0234234OMgsoooososospoopy@#423rfdsf@fewafewFAdsf4234@#",
        debug = debug,
        facedb = facedb,
        static_path = './static/',
    )

    print("Starting server on: {}".format(port))
    server = httpserver.HTTPServer(application)
    server.listen(port)
    ioloop.IOLoop.instance().start()
