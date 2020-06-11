# -*- coding: UTF-8 -*-

from predict import *
from flask import Flask
from werkzeug.utils import secure_filename
from flask import request

application = Flask(__name__)


@application.route('/', methods=['POST'])
def infer():
    s = request.form['sign']
    sign = conf.get('key', 'key')
    if sign != s:
        return 'wrong sign'
    else:
        f = request.files['file']
        path = './pre_data/' + secure_filename(f.filename)
        f.save(path)
        data = pd.read_csv(path, encoding='gbk')['TEXT']
        output = get_prediction(data)
        return str(output)


if __name__ == '__main__':
    application.run()

