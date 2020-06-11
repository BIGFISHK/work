import cv2
import numpy as np
import paddle.fluid as fluid


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.array(img).astype('float32')
    img = np.reshape(img, (1, 28, 28))
    img = img / 255.
    img = img[np.newaxis, :]
    return img


def infer(img):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname='./model/',
                                                                                          executor=exe)
    label = exe.run(inference_program, feed={feed_target_names[0]: img}, fetch_list=fetch_targets)
    output = np.argmax(label)
    return output