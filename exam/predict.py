# -*- coding: UTF-8 -*-

from train import *
import numpy as np


def get_prediction(in_sentences):
    input_examples = [InputExample(text_a=x, label='其它') for x in in_sentences]
    input_features = convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)
    predict_input_fn = input_fn_builder(features=input_features, seq_length=max_seq_length, is_training=False,
                                        drop_remainder=False)
    result = estimator.predict(predict_input_fn)
    return [label_list[np.argmax(prediction['probabilities'])] for
            prediction in result]


# if __name__ == '__main__':
#     pred_sentences = pd.read_csv('./input/test.csv', encoding='gbk')['TEXT']
#     predictions = get_prediction(pred_sentences)
#     print(predictions)
