from flask import Flask
from werkzeug.utils import secure_filename
from flask import request
from functions import load_img, infer

application = Flask(__name__)


@application.route('/', methods=['POST'])
def img_infer():
    f = request.files['img']
    img_path = './data/' + secure_filename(f.filename)
    f.save(img_path)
    img = load_img(img_path)
    output = infer(img)
    return str(output)


if __name__ == '__main__':
    application.run()
