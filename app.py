import os.path

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import os
from model import Pre
import pickle

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

netparams = 'train.params'
ids_synsets_name = 'ids_synsets'

f = open(ids_synsets_name,'rb')
ids_synsets = pickle.load(f)
f.close()


PP = Pre(netparams,ids_synsets[1],1)


def RemoveFile(dirhname):
    for root, dirs, files in os.walk(dirhname):
        for name in files:
            os.remove(os.path.join(root, name))

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html',imagename=None,classname=None)

class Update_Image(tornado.web.RequestHandler):
    def post(self):
        RemoveFile("./static/image/")
        img = self.request.files['file'][0]
        f = open("./static/image/"+img['filename'],'wb')
        f.write(img['body'])
        f.close()
        classname = PP.PreName("./static/image/"+img['filename']).lower()
        self.render('index.html',imagename="./static/image/"+img['filename'],classname = classname)

if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[(r'/', IndexHandler), (r'/Updata_Image', Update_Image)],
        template_path=os.path.join(os.path.dirname(__file__), "./templates"),
        static_path=os.path.join(os.path.dirname(__file__),'./static'),
        debug=True
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()